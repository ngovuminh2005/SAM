"""Adaptive perturbation: rho * weight**2 * gradient / ||abs(weight)*gradient||."""

from .sam import SAM


class ASAM(SAM):
    def _directions(self):
        return {p: p.detach().to(g.dtype).square() * g for p, g in self._gradients().items()}

    def _normalizer(self, directions):
        return self._norm(p.detach().to(g.dtype).abs() * g
                          for p, g in self._gradients().items())
