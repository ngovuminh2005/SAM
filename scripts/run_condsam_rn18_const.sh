#!/usr/bin/env bash
# ResNet-18 ImageNet, 88 epochs: CondSAM (original) with the LR schedule of scripts/run_condsam_rn18.sh
# up to W&B epoch 80, exactly cosine, then held constant at that lr for the last epochs
# (cosine_tail with power 0). Same settings and W&B project as scripts/run_condsam_rn18.sh
# otherwise. Added 2026-11-15.
#
#   scripts/run_condsam_rn18_const.sh                 # run the sweep
#   DRY_RUN=1 scripts/run_condsam_rn18_const.sh       # print the commands only
#   scripts/run_condsam_rn18_const.sh --wandb.mode=disabled   # extra args go to every run
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# =============================== GPUS =================================
GPUS_PER_NODE=1              # GPU processes on this node; > 1 enables DDP
GPU_IDS=(0)                  # which GPUs, e.g. (0 1 2 3); one id per process
# ======================================================================

# ============================== SWEEP =================================
EPOCHS=(88)                  # any; the ramp and LR schedule stretch with it
METHODS=(condsam)
RHOS=(0.05 0.1 0.15)         # perturbation radius

# Only used by the matching method:
FRIENDLYSAM_SIGMAS=(1.0)
FRIENDLYSAM_LAMBDAS=(0.9)
FISHERSAM_ETAS=(1.0)
CONDSAM_BETAS=(0.99)
CONDSAM_POWER_ITERS=(5)
CONDSAM_STATS_EVERY=(1)      # refresh factors every N steps; >1 is faster, 1 is exact
CONDSAM_TAUS=(10)            # condsam3: knee at tau x the median layer norm; larger = closer to condsam

# LR / regularization, each a list to sweep (defaults = rn18_88_epochs_condsam.yaml):
LR_REFS=(0.5)                # peak lr at a global batch of 1024, scaled linearly below
LR_PEAK_EPOCHS=(2)           # end of linear warm-up
WEIGHT_DECAYS=(5e-5)
LR_SCHEDULE=cosine_tail      # cosine_tail, cosine (warm-up + cosine decay to 0), cyclic (linear decay) or step
# cosine_tail only: exactly cosine through W&B epoch TAIL_EPOCH, then from epoch TAIL_EPOCH+1
# lr = cosine(TAIL_EPOCH) * (1 - u)^power, u: 0 -> 1 over the remaining epochs (continuous).
# One run per power and rho; power 2 ~ plain cosine.
TAIL_EPOCH=80
TAIL_POWERS=(0)              # 0 holds the lr constant after TAIL_EPOCH; 2 ~ cosine, 4 decays faster
LABEL_SMOOTHING=0.1

SEEDS=(42)
# ======================================================================

# ============================== FIXED =================================
BATCH_SIZE=1024              # per GPU, as in rn18_configs
CONFIG=rn18_configs/rn18_88_epochs_condsam.yaml
# Loader workers per GPU process: all CPUs but two, split across the GPUs.
NUM_WORKERS=auto
DATA_DIR="${PROJECT_ROOT}/data"
DATA_TAG=500_0.50_90         # <max side>_<JPEG fraction>_<quality>; optional: 500_1.0_90
IN_MEMORY=auto               # 1 caches the train file in RAM; auto = 1 if it fits in 80% of RAM
LOG_DIR="${PROJECT_ROOT}/logs"
WANDB_PROJECT=rn18-ffcv-imagenet
WANDB_GROUP="rn18-condsam-cosconst-$(date +%Y%m%d-%H%M%S)"     # all runs of this sweep share a group
# ======================================================================

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

# One line of extra flags per method-specific combination.
method_variants() {
    case "$1" in
        friendlysam)
            for s in "${FRIENDLYSAM_SIGMAS[@]}"; do for l in "${FRIENDLYSAM_LAMBDAS[@]}"; do
                echo "--sam.friendlysam_sigma=${s} --sam.friendlysam_lambda=${l}"
            done; done ;;
        fishersam)
            for e in "${FISHERSAM_ETAS[@]}"; do echo "--sam.fishersam_eta=${e}"; done ;;
        condsam|condsam1|condsam2)
            for b in "${CONDSAM_BETAS[@]}"; do for p in "${CONDSAM_POWER_ITERS[@]}"; do
                for e in "${CONDSAM_STATS_EVERY[@]}"; do
                    echo "--sam.condsam_beta=${b} --sam.condsam_power_iters=${p} --sam.condsam_stats_every=${e}"
                done
            done; done ;;
        condsam3)
            for b in "${CONDSAM_BETAS[@]}"; do for p in "${CONDSAM_POWER_ITERS[@]}"; do
                for e in "${CONDSAM_STATS_EVERY[@]}"; do for t in "${CONDSAM_TAUS[@]}"; do
                    echo "--sam.condsam_beta=${b} --sam.condsam_power_iters=${p} --sam.condsam_stats_every=${e} --sam.condsam_tau=${t}"
                done; done
            done; done ;;
        *) echo "" ;;
    esac
}

# Short W&B name, e.g. condsam-rho0.1-cosine-lr0.5-pk2-wd5e-5-s42 (lr at batch 1024)
run_name() {    # method rho seed lr_ref peak wd tail
    local schedule="${LR_SCHEDULE}"
    [[ "${schedule}" == cosine_tail ]] && schedule="costail${TAIL_EPOCH}p$7"
    [[ "${LR_SCHEDULE}" == cosine_tail && "$7" == 0 ]] && schedule="cosconst${TAIL_EPOCH}"
    local tail="${schedule}-lr$4-pk$5-wd$6-s$3"
    if [[ "$2" == none ]]; then echo "$1-${tail}"; else echo "$1-rho$2-${tail}"; fi
}

for method in "${METHODS[@]}"; do
    [[ -f "sam_configs/${method}.yaml" ]] || { echo "Unknown method ${method}" >&2; exit 2; }
done

# Methods vary fastest, so each rho is compared across methods before moving on.
runs=()
for epochs in "${EPOCHS[@]}"; do
    [[ "${epochs}" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive integers, got ${epochs}" >&2; exit 2; }
    for i_rho in "${!RHOS[@]}"; do
        for method in "${METHODS[@]}"; do
            rho="${RHOS[$i_rho]}"
            if [[ "${method}" == no_sam ]]; then
                (( i_rho == 0 )) || continue    # no_sam ignores rho: run it once
                rho=none
            fi
            while IFS= read -r extra; do
                tails=("${TAIL_POWERS[@]}")
                [[ "${LR_SCHEDULE}" == cosine_tail ]] || tails=(none)     # other schedules: one run
                for lr_ref in "${LR_REFS[@]}"; do for peak in "${LR_PEAK_EPOCHS[@]}"; do
                    for wd in "${WEIGHT_DECAYS[@]}"; do for tl in "${tails[@]}"; do for seed in "${SEEDS[@]}"; do
                        runs+=("${epochs}|${method}|${rho}|${seed}|${lr_ref}|${peak}|${wd}|${tl}|${extra}")
                    done; done; done
                done; done
            done < <(method_variants "${method}")
        done
    done
done

echo "Sweep ${WANDB_GROUP}: ${#runs[@]} run(s) on GPU(s) ${CUDA_VISIBLE_DEVICES} (${CONFIG}, ${NUM_WORKERS} workers/GPU, in_memory=${IN_MEMORY}, ${PYTHON})"
failed=()
for i in "${!runs[@]}"; do
    IFS='|' read -r epochs method rho seed lr_ref peak wd tl extra <<< "${runs[$i]}"
    name="$(run_name "${method}" "${rho}" "${seed}" "${lr_ref}" "${peak}" "${wd}" "${tl}")"
    # Peak lr is given for a global batch of 1024; scale linearly with this run's batch.
    lr=$(awk -v r="${lr_ref}" -v b="${BATCH_SIZE}" -v n="${GPUS_PER_NODE}" 'BEGIN { print r * b * n / 1024 }')
    # rn18_configs differ only in epochs and the 160->192px ramp, which all
    # follow start = E - (E/4 + 1), end = E - (E/8 + 1); the LR schedule
    # stretches with E, so every epoch count is a complete schedule.
    start_ramp=$(( epochs - (epochs / 4 + 1) ))
    end_ramp=$(( epochs - (epochs / 8 + 1) ))
    command=(
        "${PYTHON}" train_imagenet.py
        --config-file "${CONFIG}"
        --config-file "sam_configs/${method}.yaml"
        --training.epochs="${epochs}"
        --resolution.start_ramp="${start_ramp}"
        --resolution.end_ramp="${end_ramp}"
        --data.train_dataset="${DATA_DIR}/train_${DATA_TAG}.ffcv"
        --data.val_dataset="${DATA_DIR}/val_${DATA_TAG}.ffcv"
        --data.num_workers="${NUM_WORKERS}"
        --data.in_memory="${IN_MEMORY}"
        --training.distributed="${DISTRIBUTED}"
        --dist.world_size="${GPUS_PER_NODE}"
        --training.batch_size="${BATCH_SIZE}"
        --training.seed="${seed}"
        --lr.lr="${lr}"
        --lr.lr_peak_epoch="${peak}"
        --lr.lr_schedule_type="${LR_SCHEDULE}"
        --training.weight_decay="${wd}"
        --training.label_smoothing="${LABEL_SMOOTHING}"
        --logging.folder="${LOG_DIR}"
        --logging.log_level=1
        --wandb.mode=online
        --wandb.project="${WANDB_PROJECT}"
        --wandb.group="${WANDB_GROUP}"
        --wandb.name="${name}"
    )
    [[ "${rho}" != none ]] && command+=(--sam.rho="${rho}")
    # W&B epochs are 1-based: tail_epoch (0-based) = TAIL_EPOCH keeps epochs 1..TAIL_EPOCH cosine
    [[ "${tl}" != none ]] && command+=(--lr.tail_epoch="${TAIL_EPOCH}" --lr.tail_power="${tl}")
    [[ -n "${extra}" ]] && read -ra extra_args <<< "${extra}" && command+=("${extra_args[@]}")
    command+=("$@")

    echo "=== [$((i + 1))/${#runs[@]}] ${name}: epochs=${epochs} method=${method} rho=${rho} lr=${lr} peak=${peak} wd=${wd} tail=${tl} seed=${seed} ${extra}"
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
echo "Sweep ${WANDB_GROUP} done"
