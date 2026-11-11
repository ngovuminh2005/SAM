"""CondSAM3: CondSAM with a log knee on layers whose perturbation norm is an outlier.

CondSAM's penalty is rho * sum_l R_l / ||E||, with <G_l, E_l> = R_well of layer l. The
norm ||E_l|| ~ ||A||^2 ||B||^2 / (2 y^T G x) is often set by one or two layers whose
y^T G x is near zero, so ||E|| -- and with it the penalty strength -- swings from step
to step, and that layer takes almost all of rho. CondSAM3 compresses only those layers:

    r_l = ||E_l|| / median_l ||E_l||          (over the conditioned layers)
    f(r) = r                   if r <= tau   (layer left exactly as in CondSAM)
    f(r) = tau (1 + ln(r/tau)) if r >  tau   (continuous, slope 1 at tau, no cap)
    E_l <- E_l * f(r_l) / r_l

then normalizes globally as CondSAM. Each layer is only scaled by a positive factor, so
the penalty is still a positively weighted sum of the layers' R_well. Large tau recovers
CondSAM; small tau spreads rho over more layers.
"""

import math

import torch

from .condsam import CondSAM


class CondSAM3(CondSAM):
    def __init__(self, params, tau=10.0, **options):
        super().__init__(params, **options)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('CondSAM3 tau must be finite and positive')
        self.options.update(tau=tau)

    @staticmethod
    def knee(ratio, tau):
        """Per-layer scale f(r)/r: 1 up to tau, log-compressed above."""
        over = ratio.clamp_min(tau)
        return torch.where(ratio > tau, tau * (1 + torch.log(over / tau)) / over, torch.ones_like(ratio))

    def _directions(self):
        directions = super()._directions()
        conditioned = [p for p in directions if 'A' in self.state[p]]
        if not conditioned:
            return directions
        tau = self.options['tau']
        norms = torch.stack([torch.linalg.vector_norm(directions[p], dtype=torch.float32) for p in conditioned])
        ratio = norms / norms.median().clamp_min(torch.finfo(torch.float32).tiny)
        for p, scale in zip(conditioned, self.knee(ratio, tau).unbind()):
            directions[p] = directions[p] * scale.to(directions[p].dtype)
        return directions

    def _graph_key(self):
        # tau is baked into the captured perturbation graph
        return (super()._graph_key(), self.options['tau'])
