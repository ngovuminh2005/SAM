#!/usr/bin/env bash
set -euo pipefail
export SAM_CONSOLE_EVAL_ONLY=1 # Print training metrics/time only on eval epochs.

# Usage: scripts/run_condsam_rho.sh [config.ini] [log_prefix] [extra main.py args...]
# Edit CONFIG_PATH, RHO_LIST, CONDSAM_BETA_LIST, CONDSAM_POWER_ITERS_LIST,
# EPOCHS and REPEATS below before running.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"
# Edit sweep values directly here.
CONFIG_PATH="configs/TinyImageNet_ResNet50.ini"
LOG_PREFIX="condsam_rho_sweep"
EPOCHS=200
EVAL_EVERY=20
GPU_ID="0"
NPROC_PER_NODE=1
ALLOW_SHARED_GPU=0 # Set to 1 only when deliberately sharing the GPU.
CONDSAM_BETA_LIST=(0)
CONDSAM_EPS=1e-8
CONDSAM_POWER_ITERS_LIST=(5) # Positive integers.
#RHO_LIST=(0.05 0.1 0.15 0.5 0.01 0.045)
RHO_LIST=(1 1.5)
REPEATS=1
WANDB_PROJECT="CondSAM"
WANDB_MODE="online" # Automatically falls back to offline if not logged in.
DATASET_ROOT="${PROJECT_ROOT}/datasets"
PYTHON="${PROJECT_ROOT}/env/bin/python"
EXTRA_ARGS=()
# Optional positional config/prefix and extra CLI arguments remain supported.
CONFIG_PATH=${1:-${CONFIG_PATH}}
LOG_PREFIX=${2:-${LOG_PREFIX}}
if (( $# > 2 )); then EXTRA_ARGS=("${@:3}"); fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAM_DATASET_ROOT="${DATASET_ROOT}"
export WANDB_PROJECT WANDB_MODE
export WANDB_RUN_GROUP="${LOG_PREFIX}"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Missing config ${CONFIG_PATH}" >&2; exit 2; }
CONFIG_TAG="$(basename "${CONFIG_PATH}")"
CONFIG_TAG="${CONFIG_TAG%.*}"

# Hold the lock for the whole sweep, including children. Repeating this command
# in another terminal must not silently start another sweep on the same GPU.
command -v flock >/dev/null || { echo "Missing flock (util-linux)" >&2; exit 2; }
lock_tag="${GPU_ID//[^a-zA-Z0-9_-]/_}"
exec {gpu_lock_fd}>"/tmp/condsam-${UID}-gpu-${lock_tag}.lock"
flock -n "${gpu_lock_fd}" || {
    echo "Another CondSAM sweep holds GPU ${GPU_ID}; wait for it to finish." >&2
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

sanitize_value() { echo "$1" | tr '.' 'p' | tr '-' 'm'; }
pick_master_port() {
    "${PYTHON}" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

passed=0
failed=()
for repeat in $(seq 1 "${REPEATS}"); do
for rho in "${RHO_LIST[@]}"; do
for beta in "${CONDSAM_BETA_LIST[@]}"; do
for power_iters in "${CONDSAM_POWER_ITERS_LIST[@]}"; do
    tag="rho$(sanitize_value "${rho}")"
    beta_tag="beta$(sanitize_value "${beta}")"
    power_tag="power$(sanitize_value "${power_iters}")"
    run_name="${LOG_PREFIX}_${CONFIG_TAG}_${tag}_${beta_tag}_${power_tag}_repeat${repeat}"
    export WANDB_NAME="${run_name}"
    port=${MASTER_PORT:-$(pick_master_port)}
    cmd=("${PYTHON}" -m torch.distributed.run --standalone --master_port "${port}"
         --nproc_per_node "${NPROC_PER_NODE}" main.py --ifile "${CONFIG_PATH}"
         --optimizer CondSAM --condsamRho "${rho}" --condsamBeta "${beta}"
         --condsamEps "${CONDSAM_EPS}" --condsamPowerIters "${power_iters}"
         --epochs "${EPOCHS}" --evalEvery "${EVAL_EVERY}" --rndSeed False --logSubDir "${run_name}")
    cmd+=("${EXTRA_ARGS[@]}")
    echo "[RUN] CondSAM rho=${rho} beta=${beta} power_iters=${power_iters} config=${CONFIG_PATH} epochs=${EPOCHS}"
    if run_training "${cmd[@]}"; then
        ((passed+=1))
        echo "[OK] ${run_name}"
    else
        status=$?
        failed+=("rho=${rho},beta=${beta},power_iters=${power_iters}:${status}")
        echo "[FAIL] ${run_name} exit=${status}"
    fi
done
done
done

done

echo "[SUMMARY] passed=${passed} failed=${#failed[@]}"
if (( ${#failed[@]} )); then
    echo "[SUMMARY] failed runs: ${failed[*]}"
    exit 1
fi
