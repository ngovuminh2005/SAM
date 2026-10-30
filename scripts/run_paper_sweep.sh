#!/usr/bin/env bash
set -euo pipefail
export SAM_CONSOLE_EVAL_ONLY=1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"

SECTION="image"
OPTIMIZERS=(CondSAM)
ADAM_OPTIMIZERS=(Adam Adam_SAM Adam_FriendlySAM CondSAM)
GLUE_OPTIMIZERS=(SGD SAM CondSAM)
RHO_LIST=(0.01 0.045 0.05 0.1 0.15 0.5)
REPEATS=3
EPOCHS=200
EVAL_EVERY=20
GPU_ID="0"
NPROC_PER_NODE=1
CONDSAM_BETA=0.95
CONDSAM_EPS=1e-8
CONDSAM_POWER_ITERS=${CONDSAM_POWER_ITERS:-1}
WANDB_PROJECT="CondSAM"
WANDB_MODE="online"
LOG_PREFIX="paper_sweep"
DATASET_ROOT="${PROJECT_ROOT}/datasets"
PYTHON="${PROJECT_ROOT}/env/bin/python"
DRY_RUN=false
TASKS=(cola sst2 mrpc qqp mnli qnli rte wnli stsb)
GLUE_RHOS=(0.002 0.005 0.005 0.005 0.005 0.005 0.004 0.004 0.004)
GLUE_EPOCHS=3
GLUE_BATCH_SIZE=32
GLUE_LR=2e-5
GLUE_CLIP_NORM=1
HESSIAN_BATCHES=20
HESSIAN_BATCH_SIZE=128
HESSIAN_ITERS=100
HESSIAN_K=5
NOISE_RATES=(0.2 0.4 0.8)
HESSIAN_RUN_DIRS=()
IMAGE_CONFIGS=(
 configs/CIFAR10_ResNet50.ini configs/CIFAR10_WRN16_4.ini
 configs/CIFAR10_PyramidNet110.ini configs/CIFAR10_WRN28_10.ini
 configs/CIFAR10_DenseNet121.ini configs/CIFAR10_ViT_Ti.ini
 configs/CIFAR100_ResNet50.ini configs/CIFAR100_WRN16_4.ini
 configs/CIFAR100_PyramidNet110.ini configs/CIFAR100_WRN28_10.ini
 configs/CIFAR100_DenseNet121.ini configs/CIFAR100_ViT_Ti.ini
 configs/TinyImageNet_ResNet50.ini configs/TinyImageNet_ResNet101.ini
)

export CUDA_VISIBLE_DEVICES="${GPU_ID}" SAM_DATASET_ROOT="${DATASET_ROOT}"
export WANDB_PROJECT WANDB_MODE
RUN_TAG="${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
export WANDB_RUN_GROUP="${RUN_TAG}_${SECTION}"
passed=0
failed=()
launch() {
 local name="$1"; shift
 export WANDB_NAME="${name}"
 local cmd=("${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" "$@")
 printf '[RUN] %s ' "${name}"; printf '%q ' "${cmd[@]}"; printf '\n'
 if ${DRY_RUN}; then return; fi
 if run_training "${cmd[@]}"; then ((passed+=1)); else failed+=("${name}:$?"); fi
}
image_run() {
 local config="$1" method="$2" rho="$3" repeat="$4"
 local key
 case "${method}" in
 CondSAM) key=condsamRho;; SAM) key=samRho;; ASAM) key=asamRho;;
 ESAM) key=esamRho;; GSAM) key=gsamRho;; FisherSAM) key=fishersamRho;;
 MSAM) key=msamRho;; FriendlySAM) key=friendlysamRho;; MongeSAM) key=mongesamRho;;
 Adam_SAM) key=adamSamRho;; Adam_FriendlySAM) key=adamwFriendlysamRho;;
 SGD|Adam) key=rho;; *) echo "Unsupported method ${method}" >&2; exit 2;;
 esac
 local name="${config##*/}_${method}_rho${rho}_noise${LABEL_NOISE_RATE:-0}_repeat${repeat}"
 launch "${name}" main.py --ifile "${config}" --optimizer "${method}" --epochs "${EPOCHS}" --evalEvery "${EVAL_EVERY}" \
   --rho "${rho}" --"${key}" "${rho}" --condsamBeta "${CONDSAM_BETA}" --condsamEps "${CONDSAM_EPS}" --condsamPowerIters "${CONDSAM_POWER_ITERS}" \
   --rndSeed False --logSubDir "${name}"
}
case "${SECTION}" in
 image)
 for config in "${IMAGE_CONFIGS[@]}"; do
  for method in "${OPTIMIZERS[@]}"; do
   radii=("${RHO_LIST[@]}")
   if [[ "${config}" == *ViT_Ti* ]]; then radii=(0.01); fi
   if [[ "${method}" == SGD || "${method}" == Adam ]]; then radii=(0); fi
   for rho in "${radii[@]}"; do
    for ((repeat=1;repeat<=REPEATS;repeat++)); do image_run "${config}" "${method}" "${rho}" "${repeat}"; done
   done
  done
 done;;
 adam)
 for config in configs/CIFAR10_ViT_Ti.ini configs/CIFAR100_ViT_Ti.ini; do
  for method in "${ADAM_OPTIMIZERS[@]}"; do
   for ((repeat=1;repeat<=REPEATS;repeat++)); do image_run "${config}" "${method}" 0.01 "${repeat}"; done
  done
 done;;
 glue)
 for task_index in "${!TASKS[@]}"; do
  task="${TASKS[task_index]}"
  export SAM_GLUE_RHO_OVERRIDE="${GLUE_RHOS[task_index]}"
  for method in "${GLUE_OPTIMIZERS[@]}"; do
   for ((repeat=1;repeat<=REPEATS;repeat++)); do
    launch "${task}_${method}_repeat${repeat}" nlp/run_glue_bert.py --ifile configs/NLP_GLUE_BERT.ini \
     --optimizer "${method}" --glue_task "${task}" --glue_output_root "logs/glue/${RUN_TAG}/repeat${repeat}" --epochs "${GLUE_EPOCHS}" --evalEvery "${EVAL_EVERY}" --batchSize "${GLUE_BATCH_SIZE}" --learningRate "${GLUE_LR}" --grad_clip_norm "${GLUE_CLIP_NORM}" --rndSeed False --condsamBeta "${CONDSAM_BETA}" --condsamEps "${CONDSAM_EPS}" --condsamPowerIters "${CONDSAM_POWER_ITERS}"
   done
  done
 done;;
 noise)
 for rate in "${NOISE_RATES[@]}"; do
  export LABEL_NOISE_RATE="${rate}" LABEL_NOISE_SEED=12345
  for config in configs/CIFAR10_WRN16_4.ini configs/CIFAR100_WRN16_4.ini; do
   for method in "${OPTIMIZERS[@]}"; do
    radii=("${RHO_LIST[@]}")
    if [[ "${method}" == SGD || "${method}" == Adam ]]; then radii=(0); fi
    for rho in "${radii[@]}"; do
     for ((repeat=1;repeat<=REPEATS;repeat++)); do image_run "${config}" "${method}" "${rho}" "${repeat}"; done
    done
   done
  done
 done;;
 hessian)
 if (( ${#HESSIAN_RUN_DIRS[@]} == 0 )); then echo 'Fill HESSIAN_RUN_DIRS before selecting hessian.' >&2; exit 2; fi
 for directory in "${HESSIAN_RUN_DIRS[@]}"; do
  launch "${directory##*/}_hessian" scripts/compute_hessian_topk.py --run_dir "${directory}" \
   --num_batches "${HESSIAN_BATCHES}" --batch_size "${HESSIAN_BATCH_SIZE}" --lanczos_iters "${HESSIAN_ITERS}" --k "${HESSIAN_K}" --disable_aug
 done;;
 *) echo "Unknown SECTION=${SECTION}" >&2; exit 2;;
esac
echo "[SUMMARY] passed=${passed} failed=${#failed[@]} dry_run=${DRY_RUN}"
if (( ${#failed[@]} )); then printf '%s\n' "${failed[@]}"; exit 1; fi
