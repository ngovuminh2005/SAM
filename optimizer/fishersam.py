"""Diagonal gradient-squared Fisher approximation from the imported baseline."""

import math

from .sam import SAM


class FisherSAM(SAM):
    def __init__(self, params, rho=0.05, eps=1e-12, eta=1.0):
        super().__init__(params, rho=rho, eps=eps)
        if not math.isfinite(eta) or eta < 0:
            raise ValueError('eta must be finite and nonnegative')
        self.options['eta'] = eta

    def _directions(self):
        return {p: g / (1 + self.options['eta'] * g.square())
                for p, g in self._gradients().items()}

    def _normalizer(self, directions):
        gradients = self._gradients()
        return sum((gradients[p].double() * d.double()).sum()
                   for p, d in directions.items()).sqrt()
