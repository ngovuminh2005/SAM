"""CondSAM perturbations followed by Adam with coupled L2 weight decay."""
import math

import torch

from .adam_sam import _adam_l2_step
from .condsam import CondSAM


class CondSAM_adam(CondSAM):
    """Reuse CondSAM's factors/two passes; update restored weights with Adam.

    ``beta`` and ``eps`` control CondSAM factors. ``betas`` and ``adam_eps``
    control Adam independently. Adam moments and factors share the optimizer
    state dictionary so checkpointing and LR/rho scheduling remain supported.
    """

    def __init__(self, params, lr=1e-3, weight_decay=5e-4, rho=0.01,
                 beta=0.95, eps=1e-8, power_iters=1,
                 betas=(0.9, 0.999), adam_eps=1e-8):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError('Adam lr must be finite and nonnegative')
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('Adam weight_decay must be finite and nonnegative')
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Adam betas must be in [0, 1)')
        if not math.isfinite(adam_eps) or adam_eps <= 0:
            raise ValueError('Adam epsilon must be finite and positive')
        super().__init__(params, lr=lr, momentum=0, weight_decay=weight_decay,
                         nesterov=False, rho=rho, beta=beta, eps=eps,
                         power_iters=power_iters)
        self.defaults.update(betas=betas, adam_eps=adam_eps)
        for group in self.param_groups:
            group.setdefault('betas', betas)
            group.setdefault('adam_eps', adam_eps)
            for p in group['params']:
                self.state[p].pop('momentum_buffer', None)

    @torch.no_grad()
    def descent_step(self):
        for group in self.param_groups:
            params, grads, first, second, steps = [], [], [], [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('CondSAM_adam does not support sparse gradients')
                state = self.state[p]
                # Factor statistics may already exist before the first Adam step.
                if 'exp_avg' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                params.append(p)
                grads.append(p.grad)
                first.append(state['exp_avg'])
                second.append(state['exp_avg_sq'])
                steps.append(state['step'])
            _adam_l2_step(
                params, grads, first, second, steps,
                beta1=group['betas'][0], beta2=group['betas'][1],
                lr=group['lr'], weight_decay=group['weight_decay'],
                eps=group['adam_eps'], maximize=False)
