"""Training-loop helpers for perturbation policies (ported from ffcv-imagenet / wise-ft)."""

import re
from contextlib import contextmanager

import torch


def select_perturbation_parameters(model, include=None, exclude=None):
    """Select trainable parameters by name; excluded parameters still train."""
    include = re.compile(include) if include else None
    exclude = re.compile(exclude) if exclude else None
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    selected = [(name, p) for name, p in model.named_parameters()
                if p.requires_grad and (include is None or include.search(name))
                and (exclude is None or not exclude.search(name))]
    if not selected:
        raise ValueError('SAM parameter filters matched no trainable parameters')
    return selected


@contextmanager
def apply_perturbations(deltas):
    """Restore exact original weights even if backward fails or gradients vanish."""
    originals = {}
    try:
        with torch.no_grad():
            for p, delta in deltas.items():
                originals[p] = p.detach().clone()
                p.add_(delta)
        yield
    finally:
        with torch.no_grad():
            for p, original in originals.items():
                p.copy_(original)


@contextmanager
def freeze_batchnorm_statistics(model):
    """Use batch statistics without updating running stats on the perturbed pass."""
    states = [(m, m.track_running_stats) for m in model.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in states:
            module.track_running_stats = False
        yield
    finally:
        for module, tracking in states:
            module.track_running_stats = tracking
