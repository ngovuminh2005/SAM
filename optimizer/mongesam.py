"""Monge perturbation: rho * g / (||g|| * sqrt(1 + ||g||**2))."""

from .sam import SAM


class MongeSAM(SAM):
    def _normalizer(self, directions):
        norm = self._norm(directions.values())
        return norm * (1 + norm.square()).sqrt()
