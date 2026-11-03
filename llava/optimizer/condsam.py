"""Empirical Kronecker-factor conditioning for SAM perturbations only."""

import math
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .sam import SAM


@contextmanager
def _fp32_matmul():
    """Full fp32 matmuls: bmm/baddbmm would otherwise use TF32 when allow_tf32 is set
    globally (HF's tf32 flag), unlike the per-layer matrix-vector products they replace."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class _Group:
    """The Kronecker factors of one size d and device, stacked so one bmm serves them all.

    slots[j] = (layer index, 'A' or 'B'); stats/mats are (k, d, d), vecs (k, d). The layer
    states hold the views mat_views[j] / vec_views[j], so updates must stay in place.
    """
    def __init__(self, d, device, slots):
        k = len(slots)
        self.slots = slots
        self.layer_index = torch.tensor([i for i, _ in slots], device=device)
        self.stats = torch.zeros(k, d, d, device=device, dtype=torch.float32)
        self.mats = torch.zeros(k, d, d, device=device, dtype=torch.float32)
        self.vecs = torch.zeros(k, d, device=device, dtype=torch.float32)
        self.mat_views = list(self.mats.unbind(0))
        self.vec_views = list(self.vecs.unbind(0))
        self.stat_views = list(self.stats.unbind(0))
        # Rows of the stack holding A (resp. B) factors, and the layers they belong to.
        self.rows = {}
        for which in 'AB':
            rows = [j for j, (_, w) in enumerate(slots) if w == which]
            if rows:
                rows = torch.tensor(rows, device=device)
                self.rows[which] = (rows, self.layer_index[rows])


class _Layout:
    """Per-layer statistics and factors as views into a few stacked tensors (one per size).

    Built once per set of conditioned layers; it replaces ~10^4 small per-layer kernels of
    the factor update and the perturbation with a few batched ones.
    """
    def __init__(self, layers, matrix_shape):
        self.params = [p for p, _ in layers]
        self.index = {p: i for i, p in enumerate(self.params)}
        device = self.params[0].device
        self.counts = torch.zeros(len(self.params), device=device, dtype=torch.float64)
        self.count_views = list(self.counts.unbind(0))
        by_size = defaultdict(list)
        for i, p in enumerate(self.params):
            out_dim, in_dim = matrix_shape(p)
            by_size[(in_dim, p.device)].append((i, 'A'))
            by_size[(out_dim, p.device)].append((i, 'B'))
        self.groups = [_Group(d, dev, slots) for (d, dev), slots in by_size.items()]
        self.slot = {}
        for group in self.groups:
            for j, (i, which) in enumerate(group.slots):
                self.slot[(i, which)] = (group, j)
        # x^T A x and ||A||_F^2 per layer and factor, refreshed whenever the factors change.
        self.quad = {w: torch.zeros(len(self.params), device=device) for w in 'AB'}
        self.sqnorm = {w: torch.zeros(len(self.params), device=device) for w in 'AB'}
        self.ready = set()  # layer indices whose state is the stacked views

    def stats(self, i):
        """(A statistics, B statistics, row count) of layer i: the hooks accumulate into these."""
        (a, ja), (b, jb) = self.slot[(i, 'A')], self.slot[(i, 'B')]
        return a.stat_views[ja], b.stat_views[jb], self.count_views[i]


class CondSAM(SAM):
    STATS_CHUNK_BYTES = 256 * 1024**2

    def __init__(self, params, rho=0.05, beta=0.95, eps=1e-8, power_iters=1, update_interval=1):
        super().__init__(params, rho=rho, eps=eps)
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError('CondSAM beta must be in [0, 1)')
        self._validate_power_iters(power_iters)
        self._validate_positive_int(update_interval, 'update_interval')
        self.options.update(beta=beta, power_iters=power_iters, update_interval=update_interval)
        self._distributed = False
        self._process_group = None
        self._stats_buffers = {}
        self._observe_calls = 0
        self._layer_cache = None
        self._layout = None
        self._checked_coverage = False

    @staticmethod
    def _validate_positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'CondSAM {name} must be a positive integer')

    @classmethod
    def _validate_power_iters(cls, value):
        cls._validate_positive_int(value, 'power_iters')

    @staticmethod
    def _matrix_shape(p):
        # ZeRO-3 partitions leave p.shape == (0,); ds_shape keeps the full shape.
        shape = getattr(p, 'ds_shape', p.shape)
        return shape[0], math.prod(shape) // shape[0]

    def _finite_many(self, values, name):
        if not values:
            return
        invalid = torch.stack([~torch.isfinite(value).all() for value in values]).any().to(torch.int32)
        if self._distributed:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self._process_group)
        if invalid.item():
            raise FloatingPointError(f'CondSAM non-finite {name}')

    def _finite_flags(self, checks):
        """checks: [(name, tensor)]. One reduction and one host sync for all of them."""
        invalid = torch.stack([~torch.isfinite(value).all() for _, value in checks]).to(torch.int32)
        if self._distributed:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self._process_group)
        for (name, _), bad in zip(checks, invalid.tolist()):
            if bad:
                raise FloatingPointError(f'CondSAM non-finite {name}')

    @staticmethod
    def _outer_sum(rows, out=None, alpha=1.0, exact_in_tf32=False):
        """out += alpha * rows^T rows, accumulated in fp32.

        exact_in_tf32: the rows hold bf16/fp16 values, which TF32 represents exactly, so the
        products run on TF32 tensor cores without rounding (sums stay fp32), whatever the
        global torch.backends.cuda.matmul.allow_tf32 says.
        """
        result = out if out is not None else rows.new_zeros((rows.shape[-1], rows.shape[-1]), dtype=torch.float32)
        tf32 = exact_in_tf32 and rows.is_cuda
        previous = torch.backends.cuda.matmul.allow_tf32
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            for chunk in rows.split(262144):
                chunk = chunk.float()
                result.addmm_(chunk.T, chunk, alpha=alpha)
        finally:
            if tf32:
                torch.backends.cuda.matmul.allow_tf32 = previous
        return result

    @torch.no_grad()
    def _accumulate(self, module, activation, grad, batch_size, observation):
        a_sum, b_sum, count = observation
        low_precision = (torch.bfloat16, torch.float16)
        a_low, b_low = activation.dtype in low_precision, grad.dtype in low_precision
        if isinstance(module, torch.nn.Linear):
            a = activation.reshape(-1, activation.shape[-1])
            b = grad.reshape(-1, grad.shape[-1])
            self._outer_sum(a, out=a_sum, exact_in_tf32=a_low)
            # (batch_size * g)^T (batch_size * g), scaled after the product so g stays exact.
            self._outer_sum(b, out=b_sum, alpha=batch_size ** 2, exact_in_tf32=b_low)
            count.add_(a.shape[0])
            return
        # Target ~256 MiB of patch/derivative data per chunk; reshape can copy.
        # The sums and row count are identical to the per-image definition.
        positions = grad.shape[-2] * grad.shape[-1]
        features = a_sum.shape[0] + b_sum.shape[0]
        chunk_size = max(1, self.STATS_CHUNK_BYTES // (4 * positions * features))
        for start in range(0, activation.shape[0], chunk_size):
            image = activation[start:start+chunk_size].float()
            derivative = grad[start:start+chunk_size]
            padding = module.padding
            if module.padding_mode != 'zeros' or isinstance(padding, str):
                image = F.pad(image, module._reversed_padding_repeated_twice,
                              mode=module.padding_mode if module.padding_mode != 'zeros' else 'constant')
                padding = 0
            if isinstance(padding, int):
                padding = (padding, padding)
            if any(padding):
                image = F.pad(image, (padding[1], padding[1], padding[0], padding[0]))
            # F.unfold's CUDA implementation launches im2col once per image.
            # Strided views plus one materialization process the whole chunk.
            kernel_h, kernel_w = module.kernel_size
            dilation_h, dilation_w = module.dilation
            patches = image.unfold(2, dilation_h * (kernel_h - 1) + 1, module.stride[0])
            patches = patches.unfold(3, dilation_w * (kernel_w - 1) + 1, module.stride[1])
            patches = patches[..., ::dilation_h, ::dilation_w]
            a = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, a_sum.shape[0])
            b = derivative.flatten(2).transpose(1, 2).reshape(-1, b_sum.shape[0])
            self._outer_sum(a, out=a_sum, exact_in_tf32=a_low)
            self._outer_sum(b, out=b_sum, alpha=batch_size ** 2, exact_in_tf32=b_low)
            count.add_(a.shape[0])

    def _layers(self, model):
        owned = {id(p) for p in self.params}
        # nn.MultiheadAttention applies out_proj.weight inside F.multi_head_attention_forward
        # without calling the module, so a hook never sees its input: leave it to SAM.
        unobservable = {id(m.out_proj) for m in model.modules()
                        if isinstance(m, torch.nn.MultiheadAttention)}
        # Shared weights with unsupported owners (e.g. tied embeddings) use SAM.
        owners = {}
        for module in model.modules():
            if id(module) in unobservable:
                continue
            p = getattr(module, 'weight', None)
            if isinstance(p, torch.nn.Parameter) and id(p) in owned:
                owners.setdefault(p, []).append(module)
        return [(p, modules) for p, modules in owners.items()
                if p.requires_grad and all(isinstance(m, torch.nn.Linear) or
                    (isinstance(m, torch.nn.Conv2d) and m.groups == 1) for m in modules)]

    def _sync_layout(self):
        """Move factors that are not the layout's views (first use, load_state_dict) into it."""
        layout, moved = self._layout, False
        for i, p in enumerate(layout.params):
            state = self.state.get(p)
            if not state or 'A' not in state:
                continue
            for which, vec in (('A', 'x'), ('B', 'y')):
                group, j = layout.slot[(i, which)]
                if state[which] is not group.mat_views[j]:
                    group.mats[j].copy_(state[which])
                    group.vecs[j].copy_(state[vec])
                    state[which], state[vec] = group.mat_views[j], group.vec_views[j]
                    moved = True
            layout.ready.add(i)
        if moved:
            self._refresh_factor_summaries()

    @torch.no_grad()
    def _refresh_factor_summaries(self):
        """x^T A x and ||A||_F^2 (and for B, y) per layer, for the perturbation."""
        layout = self._layout
        with _fp32_matmul():
            for group in layout.groups:
                sqnorm = group.mats.square().sum((1, 2))
                quad = torch.bmm(torch.bmm(group.vecs.unsqueeze(1), group.mats),
                                 group.vecs.unsqueeze(2)).view(-1)
                for which, (rows, index) in group.rows.items():
                    layout.quad[which].index_copy_(0, index, quad[rows])
                    layout.sqnorm[which].index_copy_(0, index, sqnorm[rows])

    @torch.no_grad()
    def _update_factors(self):
        layout = self._layout
        if self._distributed:
            # A few large all-reduces (one per factor size) instead of one per layer.
            works = [dist.all_reduce(group.stats, group=self._process_group, async_op=True)
                     for group in layout.groups]
            works.append(dist.all_reduce(layout.counts, group=self._process_group, async_op=True))
            for work in works:
                work.wait()
        power_iters = self.options['power_iters']
        self._validate_power_iters(power_iters)
        self._sync_layout()
        counts = layout.counts
        new = [i for i in range(len(layout.params)) if i not in layout.ready]
        if new:
            # First observation: identity factors and random unit vectors (eager, once per
            # layer, in the same order and with the same RNG draws as per-layer code).
            observed = (counts[new] > 0).tolist()
            for i, seen in zip(new, observed):
                if not seen:
                    continue
                p = layout.params[i]
                state = self.state[p]
                for which in 'AB':
                    group, j = layout.slot[(i, which)]
                    group.mats[j].copy_(torch.eye(group.mats.shape[-1], device=p.device))
                for which, key in (('A', 'x'), ('B', 'y')):
                    group, j = layout.slot[(i, which)]
                    v = torch.randn(group.vecs.shape[-1], device=p.device, dtype=torch.float32)
                    v /= v.norm().clamp_min(torch.finfo(v.dtype).tiny)
                    if self._distributed:
                        # None is the default group, which get_global_rank does not accept.
                        src = dist.get_global_rank(self._process_group or dist.group.WORLD, 0)
                        dist.broadcast(v, src=src, group=self._process_group)
                    group.vecs[j].copy_(v)
                for which, key in (('A', 'x'), ('B', 'y')):
                    group, j = layout.slot[(i, which)]
                    state[which], state[key] = group.mat_views[j], group.vec_views[j]
                layout.ready.add(i)
        active = counts > 0
        weight = active.float() * (1 - self.options['beta'])
        denominator = counts.clamp_min(1).float()
        tiny = torch.finfo(torch.float64).tiny
        checks = []
        with _fp32_matmul():
            for group in layout.groups:
                index = group.layer_index
                # Statistics that are not finite reach the factor through the EMA (also with
                # weight 0), so checking ||factor|| afterwards covers them without a copy.
                group.mats.lerp_(group.stats / denominator[index].view(-1, 1, 1), weight[index].view(-1, 1, 1))
                keep = active[index].unsqueeze(1)
                for _ in range(power_iters):
                    v = torch.bmm(group.mats, group.vecs.unsqueeze(2)).squeeze(2)
                    norm = torch.linalg.vector_norm(v.double(), dim=1)
                    checks.append(('power iteration', norm))
                    candidate = (v.double() / norm.clamp_min(tiny).unsqueeze(1)).float()
                    group.vecs.copy_(torch.where(keep & (norm > 0).unsqueeze(1), candidate, group.vecs))
        self._refresh_factor_summaries()
        checks = [('factor statistics', layout.sqnorm[w]) for w in 'AB'] + checks
        self._finite_flags(checks)
        if not self._checked_coverage:
            # A module run through .forward() or a functional call skips hooks and would
            # silently keep the plain SAM direction.
            self._checked_coverage = True
            missing = len(layout.params) - len(layout.ready)
            if missing:
                warnings.warn(f'CondSAM: {missing} of {len(layout.params)} conditioned layers got no '
                              'statistics in the first update (hooks not called?); they use the SAM direction')

    @staticmethod
    def _direction(gradient, state, eps):
        if 'A' not in state:
            return gradient.float().clone()
        g = gradient.float().reshape(gradient.shape[0], -1)
        a, b, x, y = (state[k] for k in ('A', 'B', 'x', 'y'))
        s = y @ g @ x
        denominator = s + torch.where(s >= 0, eps, -eps)
        spectral = (x @ a @ x) * (y @ b @ y)
        direction = (spectral.square() / (2 * denominator)) * torch.outer(y, x)
        direction -= (a.square().sum() * b.square().sum() /
                      (2 * g.numel() * (g.square().sum() + eps))) * g
        direction = torch.where(torch.count_nonzero(direction) == 0, g, direction)
        return direction.reshape_as(gradient)
    @contextmanager
    def observe(self, model, batch_size, distributed=None):
        """Collect first-pass statistics, including DataParallel replicas.

        The caller controls the forward, loss, backward and parameter selection.
        Unsupported/functional layers have no factors and retain the SAM direction.
        `distributed=True` all-reduces the factors over the default process group
        for wrappers that are not DDP (e.g. a DeepSpeed engine's inner module).
        With update_interval k only every k-th call collects statistics and updates
        the factors (K-FAC style); the others reuse the current factors.
        """
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        collect = self._observe_calls % self.options.get('update_interval', 1) == 0
        self._observe_calls += 1
        if not collect:
            yield
            return
        is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self._process_group = model.process_group if is_ddp else None
        if distributed is None:
            distributed = is_ddp
        self._distributed = bool(distributed) and dist.is_initialized() and \
            dist.get_world_size(self._process_group) > 1
        handles, observations = [], {}
        lock = threading.Lock()
        # The module tree does not change during training; walking a 7B model per call is slow.
        if self._layer_cache is None or self._layer_cache[0] is not model:
            self._layer_cache = (model, self._layers(model))
            self._layout = None
        layers = self._layer_cache[1]
        if not layers:
            yield
            return
        if self._layout is None:
            self._layout = _Layout(layers, self._matrix_shape)
            self._stats_buffers = {(p, p.device): self._layout.stats(i) for i, p in enumerate(self._layout.params)}
        layout = self._layout
        for group in layout.groups:
            group.stats.zero_()
        layout.counts.zero_()
        for (p, device), buffers in self._stats_buffers.items():
            if device != p.device:  # DataParallel replicas
                for value in buffers:
                    value.zero_()

        def hook(p):
            def forward(module, inputs, output):
                if not torch.is_tensor(output) or not output.requires_grad:
                    return
                activation = inputs[0].detach().clone()
                key = (p, activation.device)
                with lock:
                    if key not in observations:
                        buffers = self._stats_buffers.get(key)
                        if buffers is None:
                            out_dim, in_dim = self._matrix_shape(p)
                            buffers = (
                                torch.zeros(in_dim, in_dim, device=activation.device, dtype=torch.float32),
                                torch.zeros(out_dim, out_dim, device=activation.device, dtype=torch.float32),
                                torch.zeros((), device=activation.device, dtype=torch.float64),
                            )
                            self._stats_buffers[key] = buffers
                        observations[key] = buffers

                @torch.no_grad()
                def backward(grad):
                    self._accumulate(module, activation, grad.detach(), batch_size, observations[key])
                    return grad

                with lock:
                    handles.append(output.register_hook(backward))
            return forward

        try:
            for p, modules in layers:
                for module in modules:
                    handles.append(module.register_forward_hook(hook(p)))
            yield
        finally:
            for handle in handles:
                handle.remove()
        # The statistics on the parameter's own device are the layout's views (zeroed
        # above); DataParallel replicas add into them. Every DDP worker reduces the same
        # stacked tensors, used locally or not.
        for (p, device), buffers in observations.items():
            if device != p.device:
                for total, value in zip(self._stats_buffers[(p, p.device)], buffers):
                    total.add_(value.to(p.device))
        self._update_factors()

    def _directions(self):
        gradients = self._gradients()
        layout, eps = self._layout, self.options['eps']
        if layout is None:  # factors loaded but never observed: per-layer formula
            return {p: self._direction(g, self.state[p], eps) for p, g in gradients.items()}
        self._sync_layout()
        directions, buckets = {}, defaultdict(list)
        for p, g in gradients.items():
            i = layout.index.get(p)
            if i is None or i not in layout.ready:
                directions[p] = g.float()
            else:
                buckets[(tuple(g.shape), g.device)].append((p, i))
        # Same formula as _direction, batched over the layers of one gradient shape.
        for (shape, device), members in buckets.items():
            index = torch.tensor([i for _, i in members], device=device)
            g = torch.stack([gradients[p].float().reshape(shape[0], -1) for p, _ in members])
            x = torch.stack([self.state[p]['x'] for p, _ in members])
            y = torch.stack([self.state[p]['y'] for p, _ in members])
            with _fp32_matmul():
                s = torch.bmm(torch.bmm(y.unsqueeze(1), g), x.unsqueeze(2)).view(-1)
            denominator = s + torch.where(s >= 0, eps, -eps)
            spectral = layout.quad['A'][index] * layout.quad['B'][index]
            direction = (spectral.square() / (2 * denominator)).view(-1, 1, 1) * (y.unsqueeze(2) * x.unsqueeze(1))
            norms = layout.sqnorm['A'][index] * layout.sqnorm['B'][index]
            direction -= (norms / (2 * g[0].numel() * (g.square().sum((1, 2)) + eps))).view(-1, 1, 1) * g
            empty = direction.flatten(1).count_nonzero(1) == 0
            direction = torch.where(empty.view(-1, 1, 1), g, direction)
            for k, (p, _) in enumerate(members):
                directions[p] = direction[k].view(shape)
        return {p: directions[p] for p in gradients}

    def state_dict(self):
        # Copy each layer's factors on its own: deep-copying the stacked views would save
        # every stack once per layer.
        stacked = self.state
        self.state = defaultdict(dict, {p: {k: v.clone() if torch.is_tensor(v) else v for k, v in st.items()}
                                        for p, st in stacked.items()})
        try:
            return super().state_dict()
        finally:
            self.state = stacked

    def load_state_dict(self, checkpoint):
        super().load_state_dict(checkpoint)
        self._validate_power_iters(self.options['power_iters'])
        self._stats_buffers.clear()
        self._layout = None
        for state in self.state.values():
            for key in ('A', 'B', 'x', 'y'):
                if key in state:
                    state[key] = state[key].float()
