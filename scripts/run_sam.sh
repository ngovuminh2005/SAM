#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="${RUN_NAME:-sam-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/models/${RUN_NAME}}"
command=(
    "${PYTHON:-python}" -m src.wise_ft
    --train-dataset ImageNet
    --model "${MODEL:-ViT-B/32}"
    --template openai_imagenet_template
    --data-location "${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
    --optimizer "${BASE_OPTIMIZER:-adamw}"
    --sam sam
    --sam-config "${SAM_CONFIG:-${PROJECT_ROOT}/optimizer/defaults.ini}"
    --lr "${LR:-0.00003}"
    --wd "${WEIGHT_DECAY:-0.1}"
    --epochs "${EPOCHS:-10}"
    --batch-size "${BATCH_SIZE:-512}"
    --num-workers "${NUM_WORKERS:-8}"
    --eval-datasets ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch
    --save "${SAVE_DIR}"
    --results-db "${SAVE_DIR}/results.jsonl"
    --exp_name "${RUN_NAME}"
    --wandb-project "${WANDB_PROJECT:-wise-ft}"
    "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
