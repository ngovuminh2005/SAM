#!/usr/bin/env bash
set -euo pipefail
export SAM_CONSOLE_EVAL_ONLY=1 # Print training metrics/time only on eval epochs.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"
export PATH="${PROJECT_ROOT}/env/bin:${PATH}"
export SAM_DATASET_ROOT="${SAM_DATASET_ROOT:-${PROJECT_ROOT}/datasets}"
export WANDB_PROJECT="CondSAM"
export WANDB_MODE="${WANDB_MODE:-online}"


# Usage:
#   bash scripts/run_adamw_variants.sh [config_path] [log_prefix]
# Example:
#   bash scripts/run_adamw_variants.sh configs/ImageNet_ViT.ini imagenet_vit

CONFIG_PATH=${1:-configs/ImageNet_ViT.ini}
LOG_PREFIX=${2:-adamw_variants}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_$RANDOM}

# Select GPU by index (0 or 1). Override with: GPU_ID=1 bash ...
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_variant() {
    local name="$1"
    shift
    local run_name="${LOG_PREFIX}_${name}"
    if [[ -n "${RUN_SUFFIX}" ]]; then
        run_name="${run_name}_${RUN_SUFFIX}"
    fi
    echo "[RUN] ${name}"
    run_training python -m torch.distributed.run main.py \
        --ifile "${CONFIG_PATH}" \
        --logSubDir "${run_name}" \
        "$@"
}

run_variant adamw --optimizer AdamW
run_variant adamw_sam --optimizer AdamW_SAM
run_variant adamw_msam --optimizer AdamW_MSAM
run_variant adamw_friendlysam --optimizer AdamW_FriendlySAM
