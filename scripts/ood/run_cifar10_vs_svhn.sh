#!/usr/bin/env bash
set -euo pipefail
RUN_DIRS=()
CHECKPOINT="final.model"
SETTING="configs/ood/cifar10_vs_svhn.json"
GPU_ID="0"
DEVICE="auto"
MAX_EXAMPLES=0
WANDB_PROJECT="CondSAM"
WANDB_MODE="online"
DATA_ROOT=""
if (( $# )); then RUN_DIRS=("$@"); fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
