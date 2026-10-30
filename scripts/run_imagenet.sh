#!/usr/bin/env bash
set -euo pipefail

GPU_ID="0"
NPROC_PER_NODE=1
ALLOW_SHARED_GPU=0
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="ImageNet"
export WANDB_RUN_GROUP="imagenet"
export WANDB_MODE="online"

CONFIG_PATH="configs/ImageNet_ResNet50.ini"
OPTIMIZER="CondSAM"
EPOCHS=100
EVAL_EVERY=20
TOP_K=(1 5)

BATCH_SIZE=""
LEARNING_RATE=""
WEIGHT_DECAY=""
MOMENTUM=""
RHO_LIST=(0.1 0.15 0.5 1 1.5)
CONDSAM_BETA_LIST=(0.99)
CONDSAM_EPS=1e-8
CONDSAM_POWER_ITERS=5
EXTRA_ARGS=()

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

base_cmd=("${PYTHON}" -m torch.distributed.run --standalone
     --nproc_per_node "${NPROC_PER_NODE}" main.py --ifile "${CONFIG_PATH}"
     --optimizer "${OPTIMIZER}" --epochs "${EPOCHS}" --evalEvery "${EVAL_EVERY}"
     --topK "${TOP_K[@]}")
add_override() {
    if [[ -n "$2" ]]; then base_cmd+=("$1" "$2"); fi
}
add_override --batchSize "${BATCH_SIZE}"
add_override --learningRate "${LEARNING_RATE}"
add_override --weightDecay "${WEIGHT_DECAY}"
add_override --momentum "${MOMENTUM}"
case "${OPTIMIZER}" in
    CondSAM|CondSam1|CondSam2|CondSAM_adam)
        add_override --condsamEps "${CONDSAM_EPS}"
        add_override --condsamPowerIters "${CONDSAM_POWER_ITERS}"
        ;;
esac

if [[ "${DRY_RUN}" != 1 ]]; then
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
fi

rho_values=("${RHO_LIST[@]}")
beta_values=("")
if [[ -z "${rho_option}" ]] || (( ${#rho_values[@]} == 0 )); then rho_values=(""); fi
case "${OPTIMIZER}" in
    CondSAM|CondSam1|CondSam2|CondSAM_adam)
        if (( ${#CONDSAM_BETA_LIST[@]} )); then beta_values=("${CONDSAM_BETA_LIST[@]}"); fi
        ;;
esac

passed=0
failed=0
for rho in "${rho_values[@]}"; do
for beta in "${beta_values[@]}"; do
    cmd=("${base_cmd[@]}")
    if [[ -n "${rho}" ]]; then cmd+=("${rho_option}" "${rho}"); fi
    if [[ -n "${beta}" ]]; then cmd+=(--condsamBeta "${beta}"); fi
    cmd+=("${EXTRA_ARGS[@]}" "$@")
    run_tag="${OPTIMIZER}_rho${rho:-config}_beta${beta:-config}"
    export WANDB_NAME="${WANDB_RUN_GROUP}_${run_tag}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi
    echo "[TRAIN] config=${CONFIG_PATH} optimizer=${OPTIMIZER} rho=${rho:-config} beta=${beta:-config} GPUs=${GPU_ID} wandb=${WANDB_PROJECT}/${WANDB_NAME}"
    if run_training "${cmd[@]}"; then
        passed=$((passed + 1))
    else
        status=$?
        failed=$((failed + 1))
        echo "[FAIL] ${run_tag} exit=${status}" >&2
    fi
done
done

if [[ "${DRY_RUN}" != 1 ]]; then
    echo "[SUMMARY] passed=${passed} failed=${failed}"
fi
if (( failed )); then exit 1; fi
