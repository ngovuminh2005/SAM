#!/usr/bin/env bash
set -euo pipefail

# Edit settings here, then run: bash scripts/run_imagenet.sh [extra main.py args...]
# Preview without starting training: DRY_RUN=1 bash scripts/run_imagenet.sh
GPU_ID="${GPU_ID:-0}" # Comma-separated GPU indices.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" # One process per selected GPU.
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-ImageNet}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet}"
export WANDB_MODE="${WANDB_MODE:-online}"

CONFIG_PATH="${CONFIG_PATH:-configs/ImageNet_ResNet50.ini}"
OPTIMIZER="${OPTIMIZER:-CondSAM}"
EPOCHS="${EPOCHS:-100}"
EVAL_EVERY="${EVAL_EVERY:-20}"
TOP_K=(1 5) # All listed metrics are logged together. Use () to disable.

# Empty values use the selected .ini configuration.
BATCH_SIZE="${BATCH_SIZE:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
MOMENTUM="${MOMENTUM:-}"
RHO="${RHO:-}" # Overrides the selected optimizer's perturbation radius.
CONDSAM_BETA="${CONDSAM_BETA:-}"
CONDSAM_EPS="${CONDSAM_EPS:-}"
CONDSAM_POWER_ITERS="${CONDSAM_POWER_ITERS:-}"
EXTRA_ARGS=() # Additional main.py arguments; command-line arguments take precedence.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"
PYTHON="${PYTHON:-${CONDA_PREFIX:-${PROJECT_ROOT}/env}/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${SAM_DATASET_ROOT:-${PROJECT_ROOT}/datasets}}"
DRY_RUN="${DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAM_DATASET_ROOT="${DATASET_ROOT}"
export SAM_CONSOLE_EVAL_ONLY=1

[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Missing config: ${CONFIG_PATH}" >&2; exit 2; }
IFS=',' read -r -a gpu_ids <<< "${GPU_ID}"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] && (( NPROC_PER_NODE == ${#gpu_ids[@]} )) || {
    echo "NPROC_PER_NODE must match the number of indices in GPU_ID" >&2; exit 2;
}
for value in "${EPOCHS}" "${EVAL_EVERY}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Epoch settings must be positive integers" >&2; exit 2; }
done
for k in "${TOP_K[@]}"; do
    [[ "${k}" =~ ^[1-9][0-9]*$ ]] && (( k <= 1000 )) || {
        echo "ImageNet top-k values must be integers from 1 to 1000" >&2; exit 2;
    }
done

# Map RHO to the method-specific option so the .ini value cannot override it.
rho_option=""
case "${OPTIMIZER}" in
    CondSAM|CondSam1|CondSam2|CondSAM_adam) rho_option=--condsamRho ;;
    SAM) rho_option=--samRho ;;
    ASAM) rho_option=--asamRho ;;
    MSAM) rho_option=--msamRho ;;
    ESAM) rho_option=--esamRho ;;
    lookSAM) rho_option=--looksamRho ;;
    FriendlySAM) rho_option=--friendlysamRho ;;
    FisherSAM) rho_option=--fishersamRho ;;
    MongeSAM) rho_option=--mongesamRho ;;
    GSAM) rho_option=--gsamRho ;;
    BayesianSAM) rho_option=--bayesiansamRho ;;
    AdamW_SAM) rho_option=--adamwSamRho ;;
    Adam_SAM) rho_option=--adamSamRho ;;
    AdamW_MSAM) rho_option=--adamwMsamRho ;;
    AdamW_FriendlySAM|Adam_FriendlySAM) rho_option=--adamwFriendlysamRho ;;
    SGD|Adam|AdamW) ;;
    *) echo "Unknown optimizer: ${OPTIMIZER}" >&2; exit 2 ;;
esac

cmd=("${PYTHON}" -m torch.distributed.run --standalone
     --nproc_per_node "${NPROC_PER_NODE}" main.py --ifile "${CONFIG_PATH}"
     --optimizer "${OPTIMIZER}" --epochs "${EPOCHS}" --evalEvery "${EVAL_EVERY}"
     --topK "${TOP_K[@]}")
add_override() {
    if [[ -n "$2" ]]; then cmd+=("$1" "$2"); fi
}
add_override --batchSize "${BATCH_SIZE}"
add_override --learningRate "${LEARNING_RATE}"
add_override --weightDecay "${WEIGHT_DECAY}"
add_override --momentum "${MOMENTUM}"
if [[ -n "${rho_option}" ]]; then add_override "${rho_option}" "${RHO}"; fi
case "${OPTIMIZER}" in
    CondSAM|CondSam1|CondSam2|CondSAM_adam)
        add_override --condsamBeta "${CONDSAM_BETA}"
        add_override --condsamEps "${CONDSAM_EPS}"
        add_override --condsamPowerIters "${CONDSAM_POWER_ITERS}"
        ;;
esac
cmd+=("${EXTRA_ARGS[@]}" "$@")

if [[ "${DRY_RUN}" == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

# Hold a separate lock for every selected GPU until training exits.
command -v flock >/dev/null || { echo "Missing flock (util-linux)" >&2; exit 2; }
gpu_lock_fds=()
for gpu in "${gpu_ids[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "GPU_ID must contain numeric GPU indices" >&2; exit 2; }
    exec {gpu_lock_fd}>"/tmp/condsam-${UID}-gpu-${gpu}.lock"
    gpu_lock_fds+=("${gpu_lock_fd}")
    flock -n "${gpu_lock_fd}" || { echo "Another run holds GPU ${gpu}" >&2; exit 2; }
done
if [[ "${ALLOW_SHARED_GPU}" == 0 ]] && command -v nvidia-smi >/dev/null; then
    gpu_pids=$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits) || {
        echo "Cannot inspect GPUs ${GPU_ID}" >&2; exit 2;
    }
    [[ -z "${gpu_pids//[[:space:]]/}" ]] || {
        echo "GPUs ${GPU_ID} already have compute processes; wait or set ALLOW_SHARED_GPU=1" >&2; exit 2;
    }
fi

echo "[TRAIN] config=${CONFIG_PATH} optimizer=${OPTIMIZER} GPUs=${GPU_ID} wandb=${WANDB_PROJECT}"
run_training "${cmd[@]}"
