"""Empirical Kronecker-factor conditioning for SAM perturbations only."""

import math
import threading
import warnings
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .sam import SAM


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

    @torch.no_grad()
    def _update_factors(self, observations):
        if self._distributed:
            # Issued together so NCCL overlaps them; one blocking call per tensor was serial.
            works = [dist.all_reduce(value, group=self._process_group, async_op=True)
                     for values in observations.values() for value in values]
            for work in works:
                work.wait()
        self._finite_many([value for values in observations.values() for value in values], 'factor statistics')
        power_values = []
        for p, (a_sum, b_sum, count) in observations.items():
            power_iters = self.options['power_iters']
            self._validate_power_iters(power_iters)
            state = self.state[p]
            first_observation = 'A' not in state
            if first_observation:
                # This synchronization is needed only on initial observation.
                if count.item() == 0:
                    continue
                state['A'] = torch.eye(a_sum.shape[0], device=p.device, dtype=torch.float32)
                state['B'] = torch.eye(b_sum.shape[0], device=p.device, dtype=torch.float32)
                for key, size in [('x', a_sum.shape[0]), ('y', b_sum.shape[0])]:
                    v = torch.randn(size, device=p.device, dtype=torch.float32)
                    v /= v.norm().clamp_min(torch.finfo(v.dtype).tiny)
                    if self._distributed:
                        # None is the default group, which get_global_rank does not accept.
                        group = self._process_group or dist.group.WORLD
                        src = dist.get_global_rank(group, 0)
                        dist.broadcast(v, src=src, group=self._process_group)
                    state[key] = v
            beta = self.options['beta']
            active = count > 0
            weight = active.float() * (1 - beta)
            for key, total in [('A', a_sum), ('B', b_sum)]:
                state[key].lerp_(total / count.clamp_min(1), weight)
            for matrix, vector in [('A', 'x'), ('B', 'y')]:
                for _ in range(power_iters):
                    v = state[matrix] @ state[vector]
                    power_values.append(v)
                    norm = torch.linalg.vector_norm(v.double())
                    candidate = (v.double() / norm.clamp_min(torch.finfo(torch.float64).tiny)).float()
                    state[vector] = torch.where(active & (norm > 0), candidate, state[vector])
        self._finite_many(power_values, 'power iteration')
        if not self._checked_coverage:
            # A module run through .forward() or a functional call skips hooks and would
            # silently keep the plain SAM direction.
            self._checked_coverage = True
            missing = sum('A' not in self.state[p] for p in observations)
            if missing:
                warnings.warn(f'CondSAM: {missing} of {len(observations)} conditioned layers got no '
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
        layers = self._layer_cache[1]
        for buffers in self._stats_buffers.values():
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
        pooled = {}
        # Keep the parameter order identical on every DDP worker, even if a
        # layer was unused locally. The statistics on the parameter's own device
        # are the pooled ones (zeroed above); DataParallel replicas add into them.
        for p, _ in layers:
            key = (p, p.device)
            if key not in self._stats_buffers:
                out_dim, in_dim = self._matrix_shape(p)
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

    def _directions(self):
        return {p: self._direction(g, self.state[p], self.options['eps'])
                for p, g in self._gradients().items()}

    def load_state_dict(self, checkpoint):
        super().load_state_dict(checkpoint)
        self._validate_power_iters(self.options['power_iters'])
        self._stats_buffers.clear()
        for state in self.state.values():
            for key in ('A', 'B', 'x', 'y'):
                if key in state:
                    state[key] = state[key].float()
