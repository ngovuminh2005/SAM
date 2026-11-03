"""Empirical Kronecker-factor conditioning for SAM perturbations only."""

import math
import threading
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .sam import SAM


class CondSAM(SAM):
    STATS_CHUNK_BYTES = 256 * 1024**2

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

    @staticmethod
    def _validate_power_iters(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('CondSAM power_iters must be a positive integer')

    def _finite_many(self, values, name):
        if not values:
            return
        # x * 0 is 0 for finite x and NaN otherwise, so each norm is 0 or NaN:
        # a few multi-tensor kernels instead of one reduction per tensor.
        norms = torch._foreach_norm(torch._foreach_mul(values, 0.))
        invalid = (~torch.isfinite(torch.stack(norms))).any().to(torch.int32)
        if self._distributed:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self._process_group)
        if invalid.item():
            raise FloatingPointError(f'CondSAM non-finite {name}')

    @staticmethod
    def _outer_sum(rows, out=None, alpha=1.):
        result = out if out is not None else rows.new_zeros((rows.shape[-1], rows.shape[-1]), dtype=torch.float32)
        # bf16/fp16 values are exact in TF32, so TF32 products equal fp32 ones
        # (accumulation stays fp32); this GEMM dominates CondSAM's cost.
        exact_in_tf32 = rows.dtype in (torch.bfloat16, torch.float16)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = previous or exact_in_tf32
        try:
            for chunk in rows.split(262144):
                chunk = chunk.float()
                result.addmm_(chunk.T, chunk, alpha=alpha)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        return result

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
        # Target ~256 MiB of patch/derivative data per chunk; reshape can copy.
        # The sums and row count are identical to the per-image definition.
        positions = grad.shape[-2] * grad.shape[-1]
        features = a_sum.shape[0] + b_sum.shape[0]
        chunk_size = max(1, self.STATS_CHUNK_BYTES // (4 * positions * features))
        for start in range(0, activation.shape[0], chunk_size):
            image = activation[start:start+chunk_size]
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
            self._outer_sum(a, out=a_sum)
            self._outer_sum(b, out=b_sum, alpha=batch_size ** 2)
            count.add_(a.shape[0])

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
                        src = dist.get_global_rank(self._process_group, 0)
                        dist.broadcast(v, src=src, group=self._process_group)
                    state[key] = v
            beta = self.options['beta']
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
                    power_values.append(norm)
                    candidate = (v.double() / norm.clamp_min(torch.finfo(torch.float64).tiny)).float()
                    state[vector] = torch.where(active & (norm > 0), candidate, state[vector])
        self._finite_many(power_values, 'power iteration')

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
        handles, observations = [], {}
        lock = threading.Lock()
        layers = self._layers(model)
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
