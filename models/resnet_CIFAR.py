# Backward-compatible import shim.
# ResNetCIFAR is now implemented in models/resnet.py to avoid duplicated code paths.
from .resnet import ResNetCIFAR

__all__ = ["ResNetCIFAR"]
