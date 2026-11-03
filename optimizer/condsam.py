"""Empirical Kronecker-factor conditioning for SAM perturbations only."""

import math
import threading
import warnings
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .gram import gram_accumulate, normalize_into
from .sam import SAM


class CondSAM(SAM):
    STATS_CHUNK_BYTES = 1024**3
    CUDA_GRAPHS = True
    TRITON_GRAM = True
    SIDE_STREAM = True

    def __init__(self, params, rho=0.05, beta=0.95, eps=1e-8, power_iters=1, stats_every=1):
        super().__init__(params, rho=rho, eps=eps)
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError('CondSAM beta must be in [0, 1)')
        self._validate_power_iters(power_iters)
        if isinstance(stats_every, bool) or not isinstance(stats_every, int) or stats_every < 1:
            raise ValueError('CondSAM stats_every must be a positive integer')
        self.options.update(beta=beta, power_iters=power_iters, stats_every=stats_every)
        self._distributed = False
        self._process_group = None
        self._stats_buffers = {}
        self._observations = 0
        self._update_graph = None
        self._perturb_graph = None
        self._grad_source = None
        self._warned_unobserved = False
        self._side_streams = {}

    @staticmethod
    def _validate_power_iters(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('CondSAM power_iters must be a positive integer')

    @staticmethod
    def _invalid(values):
        # x * 0 is 0 for finite x and NaN otherwise, so each norm is 0 or NaN:
        # a few multi-tensor kernels instead of one reduction per tensor.
        norms = torch._foreach_norm(torch._foreach_mul(values, 0.))
        return (~torch.isfinite(torch.stack(norms))).any().to(torch.int32)

    def _finite_many(self, values, name):
        if not values:
            return
        invalid = self._invalid(values)
        if self._distributed:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self._process_group)
        if invalid.item():
            raise FloatingPointError(f'CondSAM non-finite {name}')

    @classmethod
    def _outer_sum(cls, rows, out=None, alpha=1.):
        result = out if out is not None else rows.new_zeros((rows.shape[-1], rows.shape[-1]), dtype=torch.float32)
        return gram_accumulate(rows, result, alpha, use_triton=cls.TRITON_GRAM)

    @torch.no_grad()
    def _accumulate(self, module, activation, grad, batch_size, observation):
        a_sum, b_sum, count = observation
        if isinstance(module, torch.nn.Linear):
            a = activation.reshape(-1, activation.shape[-1])
            b = grad.reshape(-1, grad.shape[-1])
            self._outer_sum(a, out=a_sum)
            self._outer_sum(b, out=b_sum, alpha=batch_size ** 2)
            count.add_(a.shape[0])
            return
        # Output side: one pass over every position (a view for channels_last)
        b = grad.permute(0, 2, 3, 1).reshape(-1, b_sum.shape[0])
        self._outer_sum(b, out=b_sum, alpha=batch_size ** 2)
        count.add_(b.shape[0])
        padding = module.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        if (module.kernel_size == (1, 1) and not isinstance(padding, str) and not any(padding)):
            # A 1x1 patch is the (strided) input itself: no im2col copy
            image = activation[:, :, ::module.stride[0], ::module.stride[1]]
            self._outer_sum(image.permute(0, 2, 3, 1).reshape(-1, a_sum.shape[0]), out=a_sum)
            return
        # Materialize ~STATS_CHUNK_BYTES of patches at a time; the sum is the
        # same as over the whole batch.
        positions = grad.shape[-2] * grad.shape[-1]
        chunk_size = max(1, self.STATS_CHUNK_BYTES // (activation.element_size() * positions * a_sum.shape[0]))
        for start in range(0, activation.shape[0], chunk_size):
            image = activation[start:start+chunk_size]
            chunk_padding = padding
            if module.padding_mode != 'zeros' or isinstance(padding, str):
                image = F.pad(image, module._reversed_padding_repeated_twice,
                              mode=module.padding_mode if module.padding_mode != 'zeros' else 'constant')
                chunk_padding = (0, 0)
            if any(chunk_padding):
                image = F.pad(image, (chunk_padding[1], chunk_padding[1], chunk_padding[0], chunk_padding[0]))
            # F.unfold's CUDA implementation launches im2col once per image.
            # Strided views plus one materialization process the whole chunk.
            kernel_h, kernel_w = module.kernel_size
            dilation_h, dilation_w = module.dilation
            patches = image.unfold(2, dilation_h * (kernel_h - 1) + 1, module.stride[0])
            patches = patches.unfold(3, dilation_w * (kernel_w - 1) + 1, module.stride[1])
            patches = patches[..., ::dilation_h, ::dilation_w]
            self._outer_sum(patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, a_sum.shape[0]), out=a_sum)

    def _layers(self, model):
        cached = getattr(self, '_layers_cache', None)
        if cached is not None and cached[0] is model:
            return cached[1]
        self._layers_cache = (model, self._find_layers(model))
        return self._layers_cache[1]

    def _find_layers(self, model):
        owned = {id(p) for p in self.params}
        # Shared weights with unsupported owners (e.g. tied embeddings) use SAM.
        owners = {}
        for module in model.modules():
            p = getattr(module, 'weight', None)
            if isinstance(p, torch.nn.Parameter) and id(p) in owned:
                owners.setdefault(p, []).append(module)
        return [(p, modules) for p, modules in owners.items()
                if p.requires_grad and all(isinstance(m, torch.nn.Linear) or
                    (isinstance(m, torch.nn.Conv2d) and m.groups == 1) for m in modules)]

    @torch.no_grad()
    def _update_factors(self, observations):
        if self._distributed:
            for values in observations.values():
                for value in values:
                    dist.all_reduce(value, group=self._process_group)
        self._validate_power_iters(self.options['power_iters'])
        ready, fresh = {}, {}
        for p, values in observations.items():
            if 'A' in self.state[p]:
                ready[p] = values
            # This synchronization is needed only until a layer is first observed.
            elif values[2].item() > 0:
                fresh[p] = values
            elif not self._warned_unobserved:
                warnings.warn('CondSAM: a conditioned layer produced no statistics (does its module '
                              'run through __call__, so that hooks fire?); it keeps the SAM direction')
                self._warned_unobserved = True
        if fresh:
            self._finite_many([t for values in fresh.values() for t in values], 'factor statistics')
            for p, values in fresh.items():
                self._init_state(p, self.state[p], values[0].shape[0], values[1].shape[0])
            # First update runs eagerly; it also warms up cuBLAS for capture
            self._finite_many(self._ema_and_power(fresh), 'power iteration')
        if not ready:
            return
        if self._graph_ready(ready):
            flags = torch.stack(self._run_update(ready))
            if self._distributed:
                dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self._process_group)
            stats_invalid, power_invalid = flags.tolist()
            if stats_invalid:
                raise FloatingPointError('CondSAM non-finite factor statistics')
            if power_invalid:
                raise FloatingPointError('CondSAM non-finite power iteration')
        else:
            self._finite_many([t for values in ready.values() for t in values], 'factor statistics')
            self._finite_many(self._ema_and_power(ready), 'power iteration')

    def _init_state(self, p, state, in_dim, out_dim):
        state['A'] = torch.eye(in_dim, device=p.device, dtype=torch.float32)
        state['B'] = torch.eye(out_dim, device=p.device, dtype=torch.float32)
        for key, size in [('x', in_dim), ('y', out_dim)]:
            v = torch.randn(size, device=p.device, dtype=torch.float32)
            v /= v.norm().clamp_min(torch.finfo(v.dtype).tiny)
            if self._distributed:
                src = dist.get_global_rank(self._process_group, 0)
                dist.broadcast(v, src=src, group=self._process_group)
            state[key] = v
        # Factor-only terms of the direction, refreshed with the factors
        state['spectral'] = torch.zeros((), device=p.device, dtype=torch.float32)
        state['ab'] = torch.zeros((), device=p.device, dtype=torch.float32)

    @staticmethod
    def _refresh_terms(state):
        a, b, x, y = (state[k] for k in ('A', 'B', 'x', 'y'))
        state['spectral'].copy_((x @ a @ x) * (y @ b @ y))
        state['ab'].copy_(a.square().sum() * b.square().sum())

    def _ema_and_power(self, observations):
        """EMA of the factors and power iteration, in place; returns the vector norms."""
        beta, power_iters = self.options['beta'], self.options['power_iters']
        norms = []
        for p, (a_sum, b_sum, count) in observations.items():
            state = self.state[p]
            active = count > 0
            weight = active.float() * (1 - beta)
            for key, total in [('A', a_sum), ('B', b_sum)]:
                # In place: the sums are zeroed before the next collection
                state[key].lerp_(total.div_(count.clamp_min(1)), weight)
            for matrix, vector in [('A', 'x'), ('B', 'y')]:
                for _ in range(power_iters):
                    v = state[matrix] @ state[vector]
                    norm = torch.linalg.vector_norm(v, dtype=torch.float64)
                    # A float32 vector is finite iff its float64 norm is
                    norms.append(norm)
                    normalize_into(state[vector], v, norm, active)
            self._refresh_terms(state)
        return norms

    def _graph_ready(self, observations):
        return self.CUDA_GRAPHS and all(t.is_cuda for values in observations.values() for t in values)

    def _run_update(self, observations):
        """Replay the checks and the update as one CUDA graph: the same kernels in
        the same order, without thousands of small launches. Recaptured if any
        tensor changes. Returns (statistics invalid, power iteration invalid)."""
        tensors = [t for p, values in observations.items()
                   for t in (*values, *(self.state[p][k] for k in ('A', 'B', 'x', 'y', 'spectral', 'ab')))]
        key = (tuple(t.data_ptr() for t in tensors), self.options['beta'], self.options['power_iters'])
        if self._update_graph is None or self._update_graph[0] != key:
            graph = torch.cuda.CUDAGraph()
            # Capture records without running; the replay below does the update once.
            # thread_local: data-loader threads may keep using CUDA meanwhile.
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                stats_invalid = self._invalid([t for values in observations.values() for t in values])
                power_invalid = self._invalid(self._ema_and_power(observations))
            self._update_graph = (key, graph, (stats_invalid, power_invalid))
        _, graph, flags = self._update_graph
        graph.replay()
        return flags

    @staticmethod
    def _direction(gradient, state, eps):
        if 'A' not in state:
            return gradient.float().clone()
        g = gradient.float().reshape(gradient.shape[0], -1)
        x, y = state['x'], state['y']
        s = y @ g @ x
        denominator = s + torch.where(s >= 0, eps, -eps)
        # spectral = (xᵀAx)(yᵀBy) and ab = ‖A‖²‖B‖², cached by _refresh_terms
        direction = (state['spectral'].square() / (2 * denominator)) * torch.outer(y, x)
        direction -= (state['ab'] / (2 * g.numel() * (g.square().sum() + eps))) * g
        direction = torch.where(torch.count_nonzero(direction) == 0, g, direction)
        return direction.reshape_as(gradient)
    @contextmanager
    def observe(self, model, batch_size):
        """Collect first-pass statistics, including DataParallel replicas.

        The caller controls the forward, loss, backward and parameter selection.
        Unsupported/functional layers have no factors and retain the SAM direction.
        """
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        # Between refreshes the factors and power vectors are reused as they are
        self._observations += 1
        if (self._observations - 1) % self.options['stats_every']:
            yield
            return
        is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self._process_group = model.process_group if is_ddp else None
        self._distributed = is_ddp and dist.get_world_size(self._process_group) > 1
        handles, observations, used = [], {}, set()
        lock = threading.Lock()
        layers = self._layers(model)
        for buffers in self._stats_buffers.values():
            for value in buffers:
                value.zero_()

        def hook(p):
            def forward(module, inputs, output):
                if not torch.is_tensor(output) or not output.requires_grad:
                    return
                # No copy: the input stays alive until backward through this
                # reference, and its version counter (shared by detach) shows
                # any in-place change made in between.
                activation = inputs[0].detach()
                version = activation._version
                key = (p, activation.device)
                with lock:
                    if key not in observations:
                        buffers = self._stats_buffers.get(key)
                        if buffers is None:
                            out_dim, in_dim = p.shape[0], p.numel() // p.shape[0]
                            buffers = (
                                torch.zeros(in_dim, in_dim, device=activation.device, dtype=torch.float32),
                                torch.zeros(out_dim, out_dim, device=activation.device, dtype=torch.float32),
                                torch.zeros((), device=activation.device, dtype=torch.float64),
                            )
                            self._stats_buffers[key] = buffers
                        observations[key] = buffers

                @torch.no_grad()
                def backward(grad):
                    if activation._version != version:
                        raise RuntimeError(f'CondSAM: the input of {type(module).__name__} was modified '
                                           'in place before backward')
                    derivative = grad.detach()
                    if not (self.SIDE_STREAM and derivative.is_cuda):
                        self._accumulate(module, activation, derivative, batch_size, observations[key])
                        return grad
                    # Overlap the statistics with the rest of backward: same work,
                    # on a second stream; the totals are awaited after backward.
                    main = torch.cuda.current_stream(derivative.device)
                    with lock:
                        side = self._side_streams.get(derivative.device)
                        if side is None:
                            side = self._side_streams[derivative.device] = torch.cuda.Stream(derivative.device)
                        used.add((side, derivative.device))
                    side.wait_stream(main)
                    with torch.cuda.stream(side):
                        self._accumulate(module, activation, derivative, batch_size, observations[key])
                    activation.record_stream(side)
                    derivative.record_stream(side)
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
            for side, device in used:
                torch.cuda.current_stream(device).wait_stream(side)
        pooled = {}
        # Keep the parameter order identical on every DDP worker, even if a
        # layer was unused locally. The buffers on the parameter's own device
        # (zeroed above) are the totals; DataParallel replicas add into them.
        for p, _ in layers:
            key = (p, p.device)
            if key not in self._stats_buffers:
                out_dim, in_dim = p.shape[0], p.numel() // p.shape[0]
                self._stats_buffers[key] = (
                    torch.zeros(in_dim, in_dim, device=p.device, dtype=torch.float32),
                    torch.zeros(out_dim, out_dim, device=p.device, dtype=torch.float32),
                    torch.zeros((), device=p.device, dtype=torch.float64),
                )
            pooled[p] = self._stats_buffers[key]
        for (p, device), buffers in observations.items():
            if device != p.device:
                for total, value in zip(pooled[p], buffers):
                    total.add_(value.to(p.device))
        self._update_factors(pooled)

    @torch.no_grad()
    def perturbations(self):
        if self.options['rho'] != 0 and self._graphable():
            return self._graphed_perturbations()
        return super().perturbations()

    def _gradients(self):
        if self._grad_source is not None:
            # Graph capture: private static copies of the gradients
            return dict(self._grad_source)
        return super()._gradients()

    def _graphable(self):
        return (self.CUDA_GRAPHS
                and all(p.grad is None or (p.grad.is_cuda and not p.grad.is_sparse) for p in self.params)
                and any(p.grad is not None for p in self.params))

    def _graphed_perturbations(self):
        """SAM.perturbations as one CUDA graph, with its four checks read back
        in a single synchronization."""
        params = [p for p in self.params if p.grad is not None]
        key = (tuple(id(p) for p in params), tuple((p.grad.dtype, p.grad.shape) for p in params),
               self.options['rho'], self.options['eps'], self._graph_key())
        if self._perturb_graph is None or self._perturb_graph[0] != key:
            buffers = {p: torch.empty(p.grad.shape, device=p.grad.device,
                                      dtype=torch.float64 if p.grad.dtype == torch.float64 else torch.float32)
                       for p in params}
            graph = torch.cuda.CUDAGraph()
            self._grad_source = buffers
            try:
                # Capture records without running; the replay below computes once
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    directions = self._directions()
                    valid = torch.stack([torch.isfinite(d).all() for d in directions.values()]).all()
                    norm = self._normalizer(directions)
                    scale = self.options['rho'] / (norm + self.options['eps'])
                    deltas = {p: (d.double() * scale).to(p.dtype) for p, d in directions.items()}
                    fits = torch.stack([torch.isfinite(d).all() for d in deltas.values()]).all()
                    flags = torch.stack([valid, torch.isfinite(norm), norm == 0, fits])
            finally:
                self._grad_source = None
            self._perturb_graph = (key, graph, buffers, deltas, flags)
        _, graph, buffers, deltas, flags = self._perturb_graph
        torch._foreach_copy_(list(buffers.values()), [p.grad for p in buffers])
        graph.replay()
        valid, finite_norm, zero, fits = flags.tolist()
        if not valid:
            raise FloatingPointError('Non-finite SAM perturbation direction')
        if not finite_norm:
            raise FloatingPointError('Non-finite SAM normalization')
        if zero:
            return {}
        if not fits:
            raise FloatingPointError('SAM perturbation overflows the parameter dtype')
        # Graph-owned outputs, overwritten by the next call
        return dict(deltas)

    def _graph_key(self):
        """State that the captured directions depend on (tensor addresses, branches)."""
        return tuple(tuple(t.data_ptr() for k, t in sorted(self.state[p].items()) if torch.is_tensor(t))
                     for p in self.params)

    def _directions(self):
        return {p: self._direction(g, self.state[p], self.options['eps'])
                for p, g in self._gradients().items()}

    def load_state_dict(self, checkpoint):
        super().load_state_dict(checkpoint)
        self._validate_power_iters(self.options['power_iters'])
        self._stats_buffers.clear()
        self._update_graph = None
        self._perturb_graph = None
        for p in self.params:
            state = self.state[p]
            if 'A' in state and 'spectral' not in state:   # checkpoint from before the cache
                state['spectral'] = torch.zeros((), device=p.device, dtype=torch.float32)
                state['ab'] = torch.zeros((), device=p.device, dtype=torch.float32)
                self._refresh_terms(state)
        for state in self.state.values():
            for key in ('A', 'B', 'x', 'y'):
                if key in state:
                    state[key] = state[key].float()
