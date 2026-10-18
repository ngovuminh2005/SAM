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
#   bash scripts/run_orthosam_rho.sh [config_path] [log_prefix] [extra_args...]
# Example:
#   bash scripts/run_orthosam_rho.sh configs/CIFAR100_WRN16_4.ini orthosam_rho
#   bash scripts/run_orthosam_rho.sh configs/CIFAR100_ResNet50.ini orthosam_rho --epochs 50
#
# Optional env override for rho sweep values:
#   RHO_VALUES="0.025 0.05 0.075 0.1 0.15 0.2"
# Optional env override for learning rate:
#   ORTHOSAM_LR=0.05

CONFIG_PATH=${1:-configs/CIFAR10_WRN28_10.ini}
LOG_PREFIX=${2:-run_orthosam_rho_sweep}
EXTRA_ARGS=("${@:3}")

GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ -n "${RHO_VALUES:-}" ]]; then
    read -r -a RHO_LIST <<< "${RHO_VALUES}"
else
    RHO_LIST=(0.035 0.045 0.05 0.06 0.075 0.1 0.15 0.2 0.5 2)
fi

ORTHOSAM_LR=${ORTHOSAM_LR:-}

declare -a PASSED_SWEEPS=()
declare -a FAILED_SWEEPS=()

pick_master_port() {
    python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

sanitize_value() {
    echo "$1" | tr '.' 'p' | tr '-' 'm'
}

run_case() {
    local rho="$1"
    local tag="rho$(sanitize_value "${rho}")"
    local run_name="${LOG_PREFIX}_${tag}"
    local master_port="${MASTER_PORT:-}"

    if [[ -z "${master_port}" ]]; then
        master_port="$(pick_master_port)"
    fi

    echo "[RUN] OrthoSAM rho=${rho} (master_port=${master_port})"
    cmd=(
        python -m torch.distributed.run
        --master_port "${master_port}"
        main.py
        --ifile "${CONFIG_PATH}"
        --logSubDir "${run_name}"
        --optimizer OrthoSAM
        --samRho "${rho}"
    )

    if [[ -n "${ORTHOSAM_LR}" ]]; then
        cmd+=(--learningRate "${ORTHOSAM_LR}")
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    if run_training "${cmd[@]}"; then
        PASSED_SWEEPS+=("${rho}")
        echo "[OK] rho=${rho}"
    else
        local exit_code=$?
        FAILED_SWEEPS+=("${rho}:${exit_code}")
        echo "[FAIL] rho=${rho} (exit=${exit_code})"
    fi
}

for rho in "${RHO_LIST[@]}"; do
    run_case "${rho}"
done

echo "[SUMMARY] passed=${#PASSED_SWEEPS[@]} failed=${#FAILED_SWEEPS[@]}"
if [[ ${#PASSED_SWEEPS[@]} -gt 0 ]]; then
    echo "[SUMMARY] passed rho: ${PASSED_SWEEPS[*]}"
fi
if [[ ${#FAILED_SWEEPS[@]} -gt 0 ]]; then
    echo "[SUMMARY] failed rho: ${FAILED_SWEEPS[*]}"
    exit 1
fi
