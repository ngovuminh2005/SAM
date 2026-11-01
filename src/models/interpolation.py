"""WiSE-FT weight interpolation, independent of the training entry point."""

import torch


def merge_weights(alpha, theta_0, theta_1, fishers=None, fisher_floor=1e-8):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    if theta_0.keys() != theta_1.keys():
        raise ValueError('Reference and fine-tuned checkpoint keys differ')
    merged = {}
    for key, initial in theta_0.items():
        final = theta_1[key]
        if initial.shape != final.shape:
            raise ValueError('Checkpoint shape mismatch: ' + key)
        if not initial.is_floating_point():
            merged[key] = initial if alpha < 0.5 else final
        elif alpha == 0:
            merged[key] = initial
        elif alpha == 1:
            merged[key] = final
        elif fishers is None:
            merged[key] = (1 - alpha) * initial + alpha * final
        else:
            fisher_0, fisher_1 = fishers
            if (key in fisher_0) != (key in fisher_1):
                raise ValueError('Fisher coverage differs: ' + key)
            ones = torch.ones_like(initial)
            f0 = torch.maximum(fisher_0.get(key, ones).to(initial), fisher_floor * ones)
            f1 = torch.maximum(fisher_1.get(key, ones).to(initial), fisher_floor * ones)
            c0, c1 = (1 - alpha) * f0, alpha * f1
            merged[key] = (c0 * initial + c1 * final) / (c0 + c1)
    return merged
