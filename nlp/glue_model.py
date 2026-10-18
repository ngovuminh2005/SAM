"""HuggingFace BERT wrapper compatible with vision SAM optimizers (forward(batch_dict), setBatchNormTracking)."""

from __future__ import annotations

import torch
from torch import nn


class BertGlueWrapper(nn.Module):
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf = hf_model

    def forward(self, batch: dict) -> torch.Tensor:
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = self.hf(**inputs, labels=None)
        return out.logits

    def setBatchNormTracking(self, track_running_stats: bool) -> None:
        del track_running_stats
        return
