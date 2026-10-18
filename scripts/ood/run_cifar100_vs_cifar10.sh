#!/usr/bin/env bash
set -euo pipefail
# Edit values here. One or more existing trained vision run directories.
RUN_DIRS=() # e.g. ("logs/CIFAR10_CondSAM_resnet_rho0.05_lr0.1")
CHECKPOINT="final.model"
SETTING="configs/ood/cifar100_vs_cifar10.json"
GPU_ID="0"
DEVICE="auto" # auto | cuda | cpu
MAX_EXAMPLES=0 # full dataset; positive values are diagnostic subsets only
WANDB_PROJECT="CondSAM"
WANDB_MODE="online"
DATA_ROOT="" # empty uses <project>/datasets
if (( $# )); then RUN_DIRS=("$@"); fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
