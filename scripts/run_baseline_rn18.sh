#!/usr/bin/env bash
# ResNet-18 ImageNet, 88 epochs (rn18_configs/rn18_88_epochs.yaml, standard linear LR decay):
# no SAM and every SAM baseline this repo implements except CondSAM (see
# scripts/run_condsam_rn18.sh), back to back, each at the one rho and method-specific
# setting below. Added 2026-11-14.
#
#   scripts/run_baseline_rn18.sh                       # run all 6 jobs
#   DRY_RUN=1 scripts/run_baseline_rn18.sh             # print the commands only
#   scripts/run_baseline_rn18.sh --wandb.mode=disabled  # extra args go to every run
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# =============================== GPUS =================================
GPUS_PER_NODE=1              # GPU processes on this node; > 1 enables DDP
GPU_IDS=(0)                  # which GPUs, e.g. (0 1 2 3); one id per process
# ======================================================================

# ============================== BASELINES ==============================
EPOCHS=88                    # any; 16 24 32 40 56 88 reproduce rn18_configs/ exactly
SEED=42

# One run per method, in this order; rho/eta/sigma/lambda match the rho table
# (no_sam ignores its rho).
BASELINE_METHODS=(no_sam sam  asam friendlysam fishersam mongesam)
BASELINE_RHOS=(0      0.05 1.0  0.075       0.05      0.05)
FRIENDLYSAM_SIGMA=1.0
FRIENDLYSAM_LAMBDA=0.95      # "lmbda" in the rho table
FISHERSAM_ETA=1.0
# ======================================================================

# ============================== FIXED =================================
BATCH_SIZE=1024              # per GPU, as in rn18_configs
# rn18_configs use lr 0.5 for 1 GPU x 1024; scale linearly with the global batch.
LR=$(awk -v b="${BATCH_SIZE}" -v n="${GPUS_PER_NODE}" 'BEGIN { print 0.5 * b * n / 1024 }')
# Loader workers per GPU process: all CPUs but two, split across the GPUs.
NUM_WORKERS=auto
DATA_DIR="${PROJECT_ROOT}/data"
DATA_TAG=500_0.50_90         # <max side>_<JPEG fraction>_<quality>; optional: 500_1.0_90
IN_MEMORY=auto               # 1 caches the train file in RAM; auto = 1 if it fits in 80% of RAM
LOG_DIR="${PROJECT_ROOT}/logs"
WANDB_PROJECT=ffcv-imagenet
WANDB_GROUP="rn18-baseline-$(date +%Y%m%d-%H%M%S)"   # all runs of this sweep share a group
# ======================================================================

(( ${#BASELINE_METHODS[@]} == ${#BASELINE_RHOS[@]} )) || {
    echo "BASELINE_METHODS and BASELINE_RHOS must have the same length" >&2; exit 2; }
for method in "${BASELINE_METHODS[@]}"; do
    [[ -f "sam_configs/${method}.yaml" ]] || { echo "Unknown method ${method}" >&2; exit 2; }
    [[ "${method}" != condsam ]] || { echo "CondSAM is excluded; use scripts/run_condsam_rn18.sh" >&2; exit 2; }
done
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be a positive integer, got ${EPOCHS}" >&2; exit 2; }

(( ${#GPU_IDS[@]} == GPUS_PER_NODE )) || {
    echo "GPU_IDS lists ${#GPU_IDS[@]} GPU(s) but GPUS_PER_NODE=${GPUS_PER_NODE}" >&2; exit 2; }
num_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
for id in "${GPU_IDS[@]}"; do
    [[ "${id}" =~ ^[0-9]+$ ]] && (( id < num_gpus )) || {
        echo "GPU id ${id} not found (nvidia-smi lists ${num_gpus} GPU(s))" >&2; exit 2; }
done
# DDP ranks 0..N-1 map onto these ids in order (train_imagenet.py is single-node).
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
DISTRIBUTED=$(( GPUS_PER_NODE > 1 ? 1 : 0 ))

if [[ "${NUM_WORKERS}" == auto ]]; then
    NUM_WORKERS=$(( ($(nproc) - 2) / GPUS_PER_NODE ))
    (( NUM_WORKERS >= 1 )) || NUM_WORKERS=1
fi

# ./env if it exists, otherwise the active environment.
if [[ -x "${PROJECT_ROOT}/env/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/env/bin/python"
else
    PYTHON="$(command -v python || true)"
    [[ -n "${PYTHON}" ]] || { echo "No ./env and no python on PATH; see README Environment" >&2; exit 2; }
fi
# Same as what `conda env config vars` sets for ./env; keeps ~/.local packages out.
export PYTHONNOUSERSITE=1

for file in "${DATA_DIR}/train_${DATA_TAG}.ffcv" "${DATA_DIR}/val_${DATA_TAG}.ffcv"; do
    [[ -f "${file}" ]] || { echo "Missing ${file}" >&2; exit 2; }
done
if [[ "${IN_MEMORY}" == auto ]]; then
    train_bytes=$(stat -c %s "${DATA_DIR}/train_${DATA_TAG}.ffcv")
    ram_bytes=$(( $(awk '/^MemTotal:/ { print $2 }' /proc/meminfo) * 1024 ))
    IN_MEMORY=$(( train_bytes * 10 < ram_bytes * 8 ? 1 : 0 ))
fi

# One line of extra flags per method, from the BASELINES block above.
baseline_extra() {   # method
    case "$1" in
        friendlysam) echo "--sam.friendlysam_sigma=${FRIENDLYSAM_SIGMA} --sam.friendlysam_lambda=${FRIENDLYSAM_LAMBDA}" ;;
        fishersam) echo "--sam.fishersam_eta=${FISHERSAM_ETA}" ;;
        *) echo "" ;;
    esac
}

# rn18_configs differ only in epochs and the 160->192px ramp, which all
# follow start = E - (E/4 + 1), end = E - (E/8 + 1); the LR schedule
# stretches with E, so every epoch count is a complete schedule.
start_ramp=$(( EPOCHS - (EPOCHS / 4 + 1) ))
end_ramp=$(( EPOCHS - (EPOCHS / 8 + 1) ))

echo "Baselines ${WANDB_GROUP}: ${#BASELINE_METHODS[@]} run(s) on GPU(s) ${CUDA_VISIBLE_DEVICES} (lr ${LR}, ${NUM_WORKERS} workers/GPU, in_memory=${IN_MEMORY}, ${PYTHON})"
failed=()
for i in "${!BASELINE_METHODS[@]}"; do
    method="${BASELINE_METHODS[$i]}"
    rho="${BASELINE_RHOS[$i]}"
    extra="$(baseline_extra "${method}")"
    name="${method}-rho${rho}-s${SEED}"
    [[ "${method}" == no_sam ]] && name="no_sam-s${SEED}"
    command=(
        "${PYTHON}" train_imagenet.py
        --config-file rn18_configs/rn18_88_epochs.yaml
        --config-file "sam_configs/${method}.yaml"
        --training.epochs="${EPOCHS}"
        --resolution.start_ramp="${start_ramp}"
        --resolution.end_ramp="${end_ramp}"
        --data.train_dataset="${DATA_DIR}/train_${DATA_TAG}.ffcv"
        --data.val_dataset="${DATA_DIR}/val_${DATA_TAG}.ffcv"
        --data.num_workers="${NUM_WORKERS}"
        --data.in_memory="${IN_MEMORY}"
        --training.distributed="${DISTRIBUTED}"
        --dist.world_size="${GPUS_PER_NODE}"
        --training.batch_size="${BATCH_SIZE}"
        --training.seed="${SEED}"
        --lr.lr="${LR}"
        --sam.rho="${rho}"
        --logging.folder="${LOG_DIR}"
        --logging.log_level=1
        --wandb.mode=online
        --wandb.project="${WANDB_PROJECT}"
        --wandb.group="${WANDB_GROUP}"
        --wandb.name="${name}"
    )
    [[ -n "${extra}" ]] && read -ra extra_args <<< "${extra}" && command+=("${extra_args[@]}")
    command+=("$@")

    echo "=== [$((i + 1))/${#BASELINE_METHODS[@]}] ${name}: method=${method} rho=${rho} ${extra}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${command[@]}"; printf '\n'
        continue
    fi
    "${command[@]}" || failed+=("$((i + 1))")
done

if (( ${#failed[@]} )); then
    echo "Failed runs: ${failed[*]}" >&2
    exit 1
fi
echo "Baselines ${WANDB_GROUP} done"
