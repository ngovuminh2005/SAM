"""Optimizer construction, parameter selection and the fine-tuning step."""

import math
import re
from contextlib import contextmanager, nullcontext

import torch

from optimizer import POLICIES


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


def build_optimizer_and_policy(model, args):
    """Keep the descent optimizer and perturbation policy independent."""
    params = [p for p in model.parameters() if p.requires_grad]
    name = getattr(args, 'optimizer', 'adamw').lower()
    optimizers = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
    if name not in optimizers:
        raise ValueError('Unknown base optimizer: ' + name)
    kwargs = dict(lr=args.lr, weight_decay=args.wd)
    if name == 'sgd':
        kwargs.update(momentum=getattr(args, 'momentum', 0.9),
                      nesterov=getattr(args, 'nesterov', False))
    optimizer = optimizers[name](params, **kwargs)
    method = getattr(args, 'sam', 'none').lower()
    if method == 'none':
        return optimizer, None, []
    if method not in POLICIES:
        raise ValueError('Unknown perturbation policy: ' + method)
    selected = select_perturbation_parameters(
        model, getattr(args, 'sam_include', None), getattr(args, 'sam_exclude', None))
    options = dict(rho=getattr(args, 'sam_rho', 0.05))
    if method == 'condsam':
        options.update(beta=getattr(args, 'condsam_beta', 0.95),
                       eps=getattr(args, 'condsam_eps', 1e-8),
                       power_iters=getattr(args, 'condsam_power_iters', 1))
    elif method == 'friendlysam':
        options.update(sigma=getattr(args, 'friendlysam_sigma', 1.0),
                       lmbda=getattr(args, 'friendlysam_lambda', 0.95))
    elif method == 'fishersam':
        options.update(eta=getattr(args, 'fishersam_eta', 1.0))
    policy = POLICIES[method]([p for _, p in selected], **options)
    return optimizer, policy, [name for name, _ in selected]


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
    states = [(m, m.track_running_stats) for m in model.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in states:
            module.track_running_stats = False
        yield
    finally:
        for module, tracking in states:
            module.track_running_stats = tracking


def train_batch(model, inputs, labels, loss_fn, optimizer, policy=None, grad_clip=1.0):
    """One descent step. Only this function runs the model and optimizer.step()."""
    if not math.isfinite(grad_clip) or grad_clip < 0:
        raise ValueError('grad_clip must be finite and nonnegative')
    use_sam = policy is not None and policy.options['rho'] > 0
    optimizer.zero_grad()
    try:
        observation = policy.observe(model, labels.shape[0]) if use_sam else nullcontext()
        with observation:
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()

        if use_sam:
            deltas = policy.perturbations()
            if deltas:
                optimizer.zero_grad()
                with apply_perturbations(deltas), freeze_batchnorm_statistics(model):
                    loss_fn(model(inputs), labels).backward()

        if grad_clip:
            params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
    except BaseException:
        optimizer.zero_grad()
        raise
    return loss.detach()
