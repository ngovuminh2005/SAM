#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EPOCHS="${EPOCHS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"
CONDSAM_RHO=1.5
CONDSAM_BETA=0.99

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"

PYTHON="${PYTHON:-${CONDA_PREFIX:-${PROJECT_ROOT}/env}/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
CONFIG="configs/CIFAR10_PyramidNet110.ini"
DRY_RUN="${DRY_RUN:-0}"
OPTIMIZERS=(SGD SAM CondSAM)

export WANDB_PROJECT="test_rwell"
export WANDB_RUN_GROUP="pyramid_cifar10"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAM_DATASET_ROOT="${DATA_ROOT}"
export SAM_CONSOLE_EVAL_ONLY=1

STAMP="$(date -u +%Y%m%dT%H%M%S_%NZ)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/logs/test_rwell_${STAMP}_$$}"
[[ "${RUN_ROOT}" == /* ]] || RUN_ROOT="${PROJECT_ROOT}/${RUN_ROOT}"
[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 2; }
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2; exit 2; }

IFS=',' read -r -a gpu_ids <<< "${GPU_ID}"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] && (( NPROC_PER_NODE == ${#gpu_ids[@]} )) || {
    echo "NPROC_PER_NODE must match the number of indices in GPU_ID" >&2; exit 2;
}
for value in "${EPOCHS}" "${EVAL_EVERY}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Epoch settings must be positive integers" >&2; exit 2; }
done

if [[ "${DRY_RUN}" != 1 ]]; then mkdir -p "${RUN_ROOT}"; fi
passed=0
failures=0
for optimizer in "${OPTIMIZERS[@]}"; do
    enabled=1
    extra_args=()
    if [[ "${optimizer}" == CondSAM ]]; then
        extra_args=(--rho "${CONDSAM_RHO}" --condsamRho "${CONDSAM_RHO}" --condsamBeta "${CONDSAM_BETA}")
    fi
    export WANDB_NAME="test_rwell_PyramidNet_CIFAR10_${optimizer}__seed42"
    cmd=(env "RWELL_KFAC=${enabled}" "RWELL_FULLH=${enabled}"
         "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"
         main.py --ifile "${CONFIG}" --optimizer "${optimizer}"
         --logDir "${RUN_ROOT}/${optimizer}" --contin False --truncate False --rndSeed False
         --epochs "${EPOCHS}" --evalEvery "${EVAL_EVERY}" "${extra_args[@]}")
    echo "[TRAIN] optimizer=${optimizer} rwell_kfac=${enabled} rwell_fullH=${enabled} wandb=${WANDB_PROJECT}/${WANDB_NAME}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi
    if run_training "${cmd[@]}"; then
        passed=$((passed + 1))
    else
        status=$?
        failures=$((failures + 1))
        echo "[FAIL] ${optimizer} exit=${status}" >&2
    fi
done

echo "[OUTPUT] ${RUN_ROOT}"
if [[ "${DRY_RUN}" != 1 ]]; then
    echo "[SUMMARY] trained=${passed} failures=${failures}"
fi
if (( failures )); then exit 1; fi
