"""SAM computes deltas only; the training loop owns parameter updates."""

import copy
import math
from collections import defaultdict
from contextlib import nullcontext

import torch


class SAM:
    def __init__(self, params, rho=0.05, eps=1e-12):
        if not math.isfinite(rho) or rho < 0:
            raise ValueError('rho must be finite and nonnegative')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        self.params = tuple(params)
        if not self.params or len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Pass a nonempty, unique selection of parameters')
        self.options = dict(rho=rho, eps=eps)
        self.state = defaultdict(dict)

    def observe(self, model, batch_size, distributed=None):
        """Optional first-pass statistics collection; never runs the model."""
        return nullcontext()

    def _gradients(self):
        gradients = {}
        for p in self.params:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise ValueError('SAM policies require dense gradients')
            dtype = torch.float64 if p.grad.dtype == torch.float64 else torch.float32
            gradients[p] = p.grad.detach().to(dtype).clone()
        return gradients

    def _directions(self):
        return self._gradients()

    @staticmethod
    def _norm(values):
        return torch.stack([torch.linalg.vector_norm(v, dtype=torch.float64)
                            for v in values]).norm()

    def _normalizer(self, directions):
        return self._norm(directions.values())

    @torch.no_grad()
    def perturbations(self):
        """Return {parameter: delta}, without changing parameters or gradients.

        Normalization includes only the parameters supplied by the caller.
        Stateful policies update their own statistics once per invocation.
        """
        if self.options['rho'] == 0:
            return {}
        directions = self._directions()
        if not directions:
            return {}
        valid = torch.stack([torch.isfinite(d).all() for d in directions.values()]).all()
        if not valid.item():
            raise FloatingPointError('Non-finite SAM perturbation direction')
        norm = self._normalizer(directions)
        if not torch.isfinite(norm).item():
            raise FloatingPointError('Non-finite SAM normalization')
        if norm.item() == 0:
            return {}
        scale = self.options['rho'] / (norm + self.options['eps'])
        deltas = {p: (d.double() * scale).to(p.dtype) for p, d in directions.items()}
        if not torch.stack([torch.isfinite(d).all() for d in deltas.values()]).all().item():
            raise FloatingPointError('SAM perturbation overflows the parameter dtype')
        return deltas

    def state_dict(self):
        return {
            'policy': type(self).__name__,
            'options': copy.deepcopy(self.options),
            'shapes': [tuple(p.shape) for p in self.params],
            'state': [copy.deepcopy(self.state[p]) for p in self.params],
        }

    def load_state_dict(self, checkpoint):
        if checkpoint['policy'] != type(self).__name__:
            raise ValueError('Perturbation policy does not match the checkpoint')
        if checkpoint['shapes'] != [tuple(p.shape) for p in self.params]:
            raise ValueError('Perturbation parameter selection does not match the checkpoint')
        if len(checkpoint['state']) != len(self.params):
            raise ValueError('Incorrect number of perturbation states')
        self.options = copy.deepcopy(checkpoint['options'])
        self.state.clear()
        for p, state in zip(self.params, checkpoint['state']):
            self.state[p] = {k: v.detach().to(p.device).clone() if torch.is_tensor(v)
                             else copy.deepcopy(v) for k, v in state.items()}
