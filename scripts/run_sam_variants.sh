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
#   bash scripts/run_sam_variants.sh [config_path] [log_prefix] [extra_args...]
# Runs SGD, SAM-family baselines, and OrthoSAM on one config.
# Log folder names are built in main.py as:
#   <dataset>_<optimizer>_<model>_rho<rho>_lr<lr>
#   e.g. CIFAR100_ASAM_resnet_rho0.5_lr0.1 (dataset comes from the config's dataset= field).
# Example:
#   bash scripts/run_sam_variants.sh configs/CIFAR100_WRN16_4.ini cifar100_wrn
#   bash scripts/run_sam_variants.sh configs/CIFAR100_VGG16.ini cifar100_vgg16
#   bash scripts/run_sam_variants.sh configs/CIFAR100_PyramidNet110.ini cifar100_pyramidnet110
#   bash scripts/run_sam_variants.sh configs/CIFAR10_WRN16_4.ini cifar10_wrn
#   bash scripts/run_sam_variants.sh configs/CIFAR10_VGG16.ini cifar10_vgg16
#   bash scripts/run_sam_variants.sh configs/CIFAR10_PyramidNet110.ini cifar10_pyramidnet110
#   bash scripts/run_sam_variants.sh configs/ImageNet_WRN16_4.ini imagenet_wrn16_4
#   bash scripts/run_sam_variants.sh configs/ImageNet_VGG16.ini imagenet_vgg16
#   bash scripts/run_sam_variants.sh configs/ImageNet_PyramidNet50.ini imagenet_pyramidnet50

CONFIG_PATH=${1:-configs/CIFAR100_DenseNet121.ini}
LOG_PREFIX=${2:-sam_variants}
EXTRA_ARGS=("${@:3}")

# First segment of config basename matches dataset in our configs (CIFAR100_..., ImageNet_...).
CONFIG_BASENAME=$(basename "${CONFIG_PATH}" .ini)
DATASET_HINT="${CONFIG_BASENAME%%_*}"

# Select GPU by index (0 or 1). Override with: GPU_ID=1 bash ...
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

declare -a PASSED_VARIANTS=()
declare -a FAILED_VARIANTS=()

pick_master_port() {
    python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

run_variant() {
    local name="$1"
    shift
    local master_port="${MASTER_PORT:-}"
    if [[ -z "${master_port}" ]]; then
        master_port="$(pick_master_port)"
    fi
    echo "[RUN] ${name} (dataset=${DATASET_HINT}, logPrefix=${LOG_PREFIX}, master_port=${master_port})"
    if run_training python -m torch.distributed.run \
        --master_port "${master_port}" \
        main.py \
        --ifile "${CONFIG_PATH}" \
        --logSubDir "${LOG_PREFIX}" \
        "$@" \
        "${EXTRA_ARGS[@]}"; then
        PASSED_VARIANTS+=("${name}")
        echo "[OK] ${name}"
    else
        local exit_code=$?
        FAILED_VARIANTS+=("${name}:${exit_code}")
        echo "[FAIL] ${name} (exit=${exit_code})"
    fi
}

run_variant sgd --optimizer SGD
run_variant sam --optimizer SAM
run_variant orthosam --optimizer OrthoSAM
run_variant asam --optimizer ASAM
run_variant msam --optimizer MSAM
run_variant esam --optimizer ESAM
run_variant friendlysam --optimizer FriendlySAM
run_variant fishersam --optimizer FisherSAM
run_variant mongesam --optimizer MongeSAM
run_variant gsam --optimizer GSAM

echo "[SUMMARY] passed=${#PASSED_VARIANTS[@]} failed=${#FAILED_VARIANTS[@]}"
if [[ ${#PASSED_VARIANTS[@]} -gt 0 ]]; then
    echo "[SUMMARY] passed variants: ${PASSED_VARIANTS[*]}"
fi
if [[ ${#FAILED_VARIANTS[@]} -gt 0 ]]; then
    echo "[SUMMARY] failed variants: ${FAILED_VARIANTS[*]}"
    exit 1
fi
