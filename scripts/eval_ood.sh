#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/eval_ood.sh REFERENCE.pt FINETUNED.pt [extra CLI flags]'
    exit 0
fi
if (( $# < 2 )); then
    echo 'Usage: bash scripts/eval_ood.sh REFERENCE.pt FINETUNED.pt [extra CLI flags]' >&2
    exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reference="$1"
finetuned="$2"
shift 2
[[ "${reference}" == /* ]] || reference="${PWD}/${reference}"
[[ "${finetuned}" == /* ]] || finetuned="${PWD}/${finetuned}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -f "${reference}" ]] || { echo "Missing reference: ${reference}" >&2; exit 2; }
    [[ -f "${finetuned}" ]] || { echo "Missing fine-tuned checkpoint: ${finetuned}" >&2; exit 2; }
fi
cd "${PROJECT_ROOT}"
RUN_NAME="${RUN_NAME:-ood-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/results/${RUN_NAME}}"
command=(
    "${PYTHON:-python}" -m src.wise_ft
    --load "${reference},${finetuned}"
    --data-location "${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
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
