#!/usr/bin/env bash
set -euo pipefail
export SAM_CONSOLE_EVAL_ONLY=1


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"

CONFIGS=(configs/CIFAR10_WRN16_4.ini configs/CIFAR100_WRN16_4.ini)
NOISE_RATES=(0.2 0.4 0.8)
LABEL_NOISE_SEED=1337
OPTIMIZERS=(SGD SAM ASAM ESAM GSAM FisherSAM MSAM FriendlySAM MongeSAM)
GPU_ID="0"
NPROC_PER_NODE=1
ALLOW_SHARED_GPU=0
LOG_PREFIX="baselines"
WANDB_PROJECT="CondSAM_robustness"
WANDB_MODE="online"
PYTHON="${PYTHON:-${CONDA_PREFIX:-${PROJECT_ROOT}/env}/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
EXTRA_ARGS=()

if (( $# > 0 )) && [[ "$1" != --* ]]; then
    CONFIGS=("$1")
    LOG_PREFIX=${2:-${LOG_PREFIX}}
    if (( $# > 2 )); then EXTRA_ARGS=("${@:3}"); fi
else
    while (( $# )); do
        case "$1" in
            --dataset)
                (( $# >= 2 )) || { echo "Missing value for --dataset" >&2; exit 2; }
                case "${2^^}" in
                    CIFAR10) CONFIGS=(configs/CIFAR10_WRN16_4.ini) ;;
                    CIFAR100) CONFIGS=(configs/CIFAR100_WRN16_4.ini) ;;
                    *) echo "Unsupported dataset '$2' (use CIFAR10 or CIFAR100)" >&2; exit 2 ;;
                esac
                shift 2
                ;;
            --config)
                (( $# >= 2 )) || { echo "Missing value for --config" >&2; exit 2; }
                CONFIGS=("$2")
                shift 2
                ;;
            --noise-rates)
                (( $# >= 2 )) || { echo "Missing value for --noise-rates" >&2; exit 2; }
                IFS=',' read -r -a NOISE_RATES <<< "$2"
                shift 2
                ;;
            --log-prefix)
                (( $# >= 2 )) || { echo "Missing value for --log-prefix" >&2; exit 2; }
                LOG_PREFIX="$2"
                shift 2
                ;;
            --master-port)
                (( $# >= 2 )) || { echo "Missing value for --master-port" >&2; exit 2; }
                export MASTER_PORT="$2"
                shift 2
                ;;
            --)
                shift
                EXTRA_ARGS=("$@")
                break
                ;;
            --help|-h)
                sed -n '1,8p' "${BASH_SOURCE[0]}"
                exit 0
                ;;
            *) echo "Unknown option: $1" >&2; exit 2 ;;
        esac
    done
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAM_DATASET_ROOT="${DATA_ROOT}"
export WANDB_PROJECT WANDB_MODE
STAMP="$(date -u +%Y%m%dT%H%M%S_%NZ)"
export WANDB_RUN_GROUP="${LOG_PREFIX}_${STAMP}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/logs/robustness_${STAMP}_$$}"
[[ "${RUN_ROOT}" == /* ]] || RUN_ROOT="${PROJECT_ROOT}/${RUN_ROOT}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2; exit 2; }
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
for config in "${CONFIGS[@]}"; do
    cfg_tag="$(basename "${config}" .ini)"
    for rate in "${NOISE_RATES[@]}"; do
        export LABEL_NOISE_RATE="${rate}" LABEL_NOISE_SEED
        rate_tag="$(echo "${rate}" | tr '.' 'p')"
        for optimizer in "${OPTIMIZERS[@]}"; do
            run_name="${LOG_PREFIX}_${cfg_tag}_${optimizer}_noise${rate_tag}_seed42"
            export WANDB_NAME="${run_name}"
            port=${MASTER_PORT:-$(pick_master_port)}
            cmd=("${PYTHON}" -m torch.distributed.run --standalone --master_port "${port}"
                 --nproc_per_node "${NPROC_PER_NODE}" main.py --ifile "${config}")
            cmd+=("${EXTRA_ARGS[@]}")
            cmd+=(--optimizer "${optimizer}" --rndSeed False
                  --logDir "${RUN_ROOT}/${run_name}" --contin False --truncate False)
            echo "[RUN] ${optimizer} config=${cfg_tag} noise=${rate} seed=42 label_noise_seed=${LABEL_NOISE_SEED}"
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
done

echo "[OUTPUT] ${RUN_ROOT}"
echo "[SUMMARY] passed=${passed} failed=${#failed[@]}"
if (( ${#failed[@]} )); then
    echo "[SUMMARY] failed runs: ${failed[*]}"
    exit 1
fi
