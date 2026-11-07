"""CondSAM1: CondSAM with the perturbation radius shared across layers as in SAM.

Inside each conditioned layer the direction is CondSAM's E_direct, unchanged. CondSAM
normalizes all layers together, so a layer's share of rho scales with
‖A‖₂²‖B‖₂² / (yᵀGx) and a few layers with small yᵀGx take almost all of it. CondSAM1
first rescales each layer's E_direct to that layer's gradient norm, so the layers share
rho as in SAM (in proportion to ‖g‖); the global normalization is then the same as SAM's.
"""

import torch

from .condsam import CondSAM


class CondSAM1(CondSAM):
    @staticmethod
    def _direction(gradient, state, eps):
        direction = CondSAM._direction(gradient, state, eps)
        if 'A' not in state:
            return direction
        g = gradient.float()
        return direction * (g.norm() / direction.norm().clamp_min(torch.finfo(torch.float32).tiny))
