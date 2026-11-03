#!/usr/bin/env bash
# Stage 2 sweep (one GPU or multi-GPU DeepSpeed) over VFM types and SAM-style
# policies, logged to W&B. Edit the SWEEP block; every combination runs in turn.
#
#   bash scripts/v1_5/train_sweep.sh                       # run the sweep
#   DRY_RUN=1 bash scripts/v1_5/train_sweep.sh             # print the commands only
#   bash scripts/v1_5/train_sweep.sh --wandb_mode offline  # extra args go to every run
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# =============================== GPUS =================================
GPUS_PER_NODE=1              # processes on this node; > 1 runs data parallel (DeepSpeed)
GPU_IDS=(0)                  # which GPUs, one id per process
# ======================================================================

# ============================== SWEEP =================================
VFM_TYPES=(covft)            # covft fullft lora bias vpt (configs/vfm/)
METHODS=(no_sam sam)         # no_sam sam asam condsam friendlysam fishersam mongesam (configs/sam/)
RHOS=(0.01 0.05)             # perturbation radius (not used by no_sam)

# Only used by the matching method:
FRIENDLYSAM_SIGMAS=(1.0)
FRIENDLYSAM_LAMBDAS=(0.9)
FISHERSAM_ETAS=(1.0)
CONDSAM_BETAS=(0.95)
CONDSAM_POWER_ITERS=(1)

SEEDS=(42)
# ======================================================================

# ============================== FIXED =================================
GLOBAL_BATCH=128             # LLaVA-1.5 / paper recipe
PER_DEVICE_BATCH=4           # lower to 2 if a run goes out of memory
SAM_INCLUDE=vision_tower     # regex of parameters to perturb (the LLM still descends at the perturbed point)
# ZeRO-3 on several 80GB GPUs; one GPU also offloads the optimizer state to CPU RAM.
if (( GPUS_PER_NODE > 1 )); then DS_CONFIG=./scripts/zero3.json; else DS_CONFIG=./scripts/zero3_offload_optimizer.json; fi
MASTER_PORT=29600
OUTPUT_ROOT=./checkpoints/sweeps
WANDB_PROJECT=covft
WANDB_GROUP="sweep-$(date +%Y%m%d-%H%M%S)"     # all runs of this sweep share a group
# ======================================================================

(( ${#GPU_IDS[@]} == GPUS_PER_NODE )) || {
    echo "GPU_IDS lists ${#GPU_IDS[@]} GPU(s) but GPUS_PER_NODE=${GPUS_PER_NODE}" >&2; exit 2; }
num_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
for id in "${GPU_IDS[@]}"; do
    [[ "${id}" =~ ^[0-9]+$ ]] && (( id < num_gpus )) || {
        echo "GPU id ${id} not found (nvidia-smi lists ${num_gpus} GPU(s))" >&2; exit 2; }
done
(( GLOBAL_BATCH % (PER_DEVICE_BATCH * GPUS_PER_NODE) == 0 )) || {
    echo "GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by ${PER_DEVICE_BATCH} x ${GPUS_PER_NODE} GPU(s)" >&2; exit 2; }
GRAD_ACCUM=$(( GLOBAL_BATCH / (PER_DEVICE_BATCH * GPUS_PER_NODE) ))
INCLUDE="localhost:$(IFS=,; echo "${GPU_IDS[*]}")"

# ./env if it exists, otherwise the active environment.
if [[ -x "${PROJECT_ROOT}/env/bin/deepspeed" ]]; then
    DEEPSPEED="${PROJECT_ROOT}/env/bin/deepspeed"
    export CUDA_HOME="${CUDA_HOME:-${PROJECT_ROOT}/env}"
    export PATH="${PROJECT_ROOT}/env/bin:${PATH}"
else
    DEEPSPEED="$(command -v deepspeed || true)"
    [[ -n "${DEEPSPEED}" ]] || { echo "No ./env and no deepspeed on PATH; see README Environment" >&2; exit 2; }
fi
export PYTHONNOUSERSITE=1

# One line of extra flags per method-specific combination.
method_variants() {
    case "$1" in
        friendlysam)
            for s in "${FRIENDLYSAM_SIGMAS[@]}"; do for l in "${FRIENDLYSAM_LAMBDAS[@]}"; do
                echo "--sam_friendlysam_sigma ${s} --sam_friendlysam_lambda ${l}"
            done; done ;;
        fishersam)
            for e in "${FISHERSAM_ETAS[@]}"; do echo "--sam_fishersam_eta ${e}"; done ;;
        condsam)
            for b in "${CONDSAM_BETAS[@]}"; do for p in "${CONDSAM_POWER_ITERS[@]}"; do
                echo "--sam_condsam_beta ${b} --sam_condsam_power_iters ${p}"
            done; done ;;
        *) echo "" ;;
    esac
}

for vfm in "${VFM_TYPES[@]}"; do
    [[ -f "configs/vfm/${vfm}.yaml" ]] || { echo "Unknown VFM type ${vfm}" >&2; exit 2; }
done
for method in "${METHODS[@]}"; do
    [[ -f "configs/sam/${method}.yaml" ]] || { echo "Unknown method ${method}" >&2; exit 2; }
done

# Methods vary fastest, so each rho is compared across methods before moving on.
runs=()
for vfm in "${VFM_TYPES[@]}"; do
    for i_rho in "${!RHOS[@]}"; do
        for method in "${METHODS[@]}"; do
            rho="${RHOS[$i_rho]}"
            if [[ "${method}" == no_sam ]]; then
                (( i_rho == 0 )) || continue    # no_sam ignores rho: run it once
                rho=none
            fi
            while IFS= read -r extra; do
                for seed in "${SEEDS[@]}"; do
                    runs+=("${vfm}|${method}|${rho}|${seed}|${extra}")
                done
            done < <(method_variants "${method}")
        done
    done
done

echo "Sweep ${WANDB_GROUP}: ${#runs[@]} run(s) on ${INCLUDE} (batch ${PER_DEVICE_BATCH} x ${GPUS_PER_NODE} GPU(s) x accum ${GRAD_ACCUM} = ${GLOBAL_BATCH}, ${DS_CONFIG})"
failed=()
for i in "${!runs[@]}"; do
    IFS='|' read -r vfm method rho seed extra <<< "${runs[$i]}"
    # Eval (llava/model/builder.py) loads a checkpoint as LLaVA only if its name contains "llava".
    name="llava-${vfm}-${method}"
    [[ "${rho}" != none ]] && name+="-rho${rho}"
    [[ -n "${extra}" ]] && name+="$(echo "${extra}" | sed -E 's/--sam_[a-z]+_([a-z_]+) /-\1/g; s/ //g')"
    name+="-seed${seed}"
    command=(
        "${DEEPSPEED}" --include "${INCLUDE}" --master_port "${MASTER_PORT}" llava/train/train_mem.py
        --config configs/base.yaml
        --config "configs/vfm/${vfm}.yaml"
        --config "configs/sam/${method}.yaml"
        --deepspeed "${DS_CONFIG}"
        --per_device_train_batch_size "${PER_DEVICE_BATCH}"
        --gradient_accumulation_steps "${GRAD_ACCUM}"
        --output_dir "${OUTPUT_ROOT}/${WANDB_GROUP}/${name}"
        --seed "${seed}"
        --sam_include "${SAM_INCLUDE}"
        --wandb_project "${WANDB_PROJECT}"
        --wandb_group "${WANDB_GROUP}"
    )
    [[ "${rho}" != none ]] && command+=(--sam_rho "${rho}")
    [[ -n "${extra}" ]] && read -ra extra_args <<< "${extra}" && command+=("${extra_args[@]}")
    command+=("$@")

    echo "=== [$((i + 1))/${#runs[@]}] ${name}"
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
