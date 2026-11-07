"""CondSAM2: SAM's perturbation plus CondSAM1's.

    delta = sam_rho * g / ‖g‖  +  rho * E / ‖E‖      (E: CondSAM1 direction)
    L(W + delta) ≈ L(W) + sam_rho ‖g‖ + rho ⟨g, E⟩ / ‖E‖

CondSAM's E is nearly orthogonal to g, so on its own it barely raises the loss; the SAM
term restores SAM's sharpness penalty, and the CondSAM term is added on top of it.
"""

import math

import torch

from .condsam1 import CondSAM1


class CondSAM2(CondSAM1):
    def __init__(self, params, sam_rho=0.05, **options):
        super().__init__(params, **options)
        if not math.isfinite(sam_rho) or sam_rho < 0:
            raise ValueError('CondSAM2 sam_rho must be finite and nonnegative')
        self.options.update(sam_rho=sam_rho)

    @torch.no_grad()
    def perturbations(self):
        deltas = super().perturbations()
        sam_rho = self.options.get('sam_rho', 0.)
        params = [p for p in self.params if p.grad is not None]
        if sam_rho == 0 or not params:
            return deltas
        grads = [p.grad.float() for p in params]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        scale = sam_rho / (norm + self.options['eps'])
        combined = {}
        for p, g in zip(params, grads):
            step = (g * scale).to(p.dtype)
            combined[p] = deltas[p] + step if p in deltas else step
        return combined
