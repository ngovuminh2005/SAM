#!/usr/bin/env bash
set -euo pipefail
export SAM_CONSOLE_EVAL_ONLY=1


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"

LOG_PREFIX="${1:-condsam}"
RHO="${2:-${CONDSAM_RHO:-0.5}}"
BETA="${3:-${CONDSAM_BETA:-0.995}}"
POWER_ITERS="${4:-${CONDSAM_POWER_ITERS:-5}}"
sanitize_value() { echo "$1" | tr '.' 'p' | tr '-' 'm'; }
SETTING_TAG="rho$(sanitize_value "${RHO}")_beta$(sanitize_value "${BETA}")_power${POWER_ITERS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/logs/${LOG_PREFIX}_${SETTING_TAG}}"
EPOCHS="${EPOCHS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"
GPU_ID="${GPU_ID:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
CONDSAM_EPS="${CONDSAM_EPS:-1e-8}"
REPEATS="${REPEATS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-CondSAM_table1}"
WANDB_MODE="${WANDB_MODE:-online}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/env/bin/python}"

CONFIGS=(
    configs/CIFAR10_ResNet50.ini
    configs/CIFAR10_WRN16_4.ini
    configs/CIFAR10_PyramidNet110.ini
    configs/CIFAR10_WRN28_10.ini
    configs/CIFAR10_DenseNet121.ini
    configs/CIFAR100_WRN16_4.ini
    configs/CIFAR100_PyramidNet110.ini
    configs/CIFAR100_WRN28_10.ini
    configs/CIFAR100_DenseNet121.ini
)

CONFIGS=(
    configs/CIFAR100_ResNet50.ini
)

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAM_DATASET_ROOT="${DATASET_ROOT}"
export WANDB_PROJECT WANDB_MODE
export WANDB_RUN_GROUP="${LOG_PREFIX}"

[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 2; }
for config in "${CONFIGS[@]}"; do
    [[ -f "${config}" ]] || { echo "Missing config ${config}" >&2; exit 2; }
done

command -v flock >/dev/null || { echo "Missing flock (util-linux)" >&2; exit 2; }
lock_tag="${GPU_ID//[^a-zA-Z0-9_-]/_}"
exec {gpu_lock_fd}>"/tmp/condsam-${UID}-gpu-${lock_tag}.lock"
flock -n "${gpu_lock_fd}" || {
    echo "Another sweep holds GPU ${GPU_ID}; wait for it to finish." >&2
    exit 2
}
if [[ "${ALLOW_SHARED_GPU}" == 0 ]] && command -v nvidia-smi >/dev/null; then
    if ! gpu_pids=$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits); then
        echo "Cannot inspect GPU ${GPU_ID}; check GPU_ID before running." >&2
        exit 2
    fi
    if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
        echo "GPU ${GPU_ID} already has compute processes: ${gpu_pids}." >&2
        echo "Finish existing runs first, or explicitly set ALLOW_SHARED_GPU=1." >&2
        exit 2
    fi
fi

run_sweep() {
    local rho="$1" beta="$2" power_iters="$3" output_root="$4"
    local passed=0
    local failed=()
    mkdir -p "${output_root}"
    for config in "${CONFIGS[@]}"; do
        config_tag="${config##*/}"
        config_tag="${config_tag%.*}"
        for repeat in $(seq 1 "${REPEATS}"); do
        run_name="${LOG_PREFIX}_${config_tag}_rho${rho}_beta${beta}_power${power_iters}_repeat${repeat}"
        export WANDB_NAME="${run_name}"
        cmd=("${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}"
            main.py --ifile "${config}" --optimizer CondSAM
            --condsamRho "${rho}" --condsamBeta "${beta}"
            --condsamEps "${CONDSAM_EPS}" --condsamPowerIters "${power_iters}"
            --epochs "${EPOCHS}" --evalEvery "${EVAL_EVERY}"
            --rndSeed False --logDir "${output_root}")
        echo "[RUN] ${run_name} config=${config} output=${output_root}"
        if run_training "${cmd[@]}"; then
            ((passed+=1))
            echo "[OK] ${run_name}"
        else
            status=$?
            failed+=("${run_name}:${status}")
            echo "[FAIL] ${run_name} exit=${status}"
        fi
        done
    done

    echo "[SUMMARY] passed=${passed} failed=${#failed[@]} output=${output_root}"
    if (( ${#failed[@]} )); then
        echo "[SUMMARY] failed runs: ${failed[*]}"
        return 1
    fi
}

run_sweep "${RHO}" "${BETA}" "${POWER_ITERS}" "${OUTPUT_ROOT}"
