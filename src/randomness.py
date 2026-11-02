"""Seed the RNG streams used by data loading, augmentation and SAM policies."""

import random

import numpy as np
import torch


def seed_everything(seed=42):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
