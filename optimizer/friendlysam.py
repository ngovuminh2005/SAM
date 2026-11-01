"""FriendlySAM's gradient-noise direction, with its own EMA state."""

import math

from .sam import SAM


class FriendlySAM(SAM):
    def __init__(self, params, rho=0.05, eps=1e-12, sigma=1.0, lmbda=0.95):
        super().__init__(params, rho=rho, eps=eps)
        if not math.isfinite(sigma) or sigma < 0 or not 0 <= lmbda < 1:
            raise ValueError('sigma must be nonnegative; lmbda must be in [0, 1)')
        self.options.update(sigma=sigma, lmbda=lmbda)

    def _directions(self):
        directions = {}
        for p, g in self._gradients().items():
            state = self.state[p]
            if 'ema' not in state:
                directions[p] = g
                state['ema'] = g.clone()
            else:
                directions[p] = g - self.options['sigma'] * state['ema']
                state['ema'].mul_(self.options['lmbda']).add_(g, alpha=1 - self.options['lmbda'])
        return directions
