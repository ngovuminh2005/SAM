"""CondSAM with log10(d_in * d_out + 1) replacing d_in * d_out."""

import math

import torch

from .condsam import CondSAM


class CondSam2(CondSAM):
    """Change only the perturbation direction; inherit factor updates and SGD.

    Use the full CondSAM direction, replacing only the dimension divisor in
    the gradient term with log10(d_in * d_out + 1), which is positive
    even for scalar 1x1 weights. Parameters without CondSAM factors retain
    the SAM fallback.
    """

    @staticmethod
    def _direction(gradient, state, eps):
        if 'A' not in state:
            return gradient.float().clone()
        g = gradient.float().reshape(gradient.shape[0], -1)
        dimension_divisor = math.log10(g.numel() + 1)
        a, b, x, y = (state[k] for k in ('A', 'B', 'x', 'y'))
        s = y @ g @ x
        denominator = s + torch.where(s >= 0, eps, -eps)
        spectral = (x @ a @ x) * (y @ b @ y)
        direction = (spectral.square() / (2 * denominator)) * torch.outer(y, x)
        direction -= (a.square().sum() * b.square().sum() /
                      (2 * dimension_divisor * (g.square().sum() + eps))) * g
        direction = torch.where(torch.count_nonzero(direction) == 0, g, direction)
        return direction.reshape_as(gradient)
