#!/usr/bin/env bash
# ImageNet + OOD (V2, R, A, Sketch) evaluation of finished runs; see eval_ood.py.
#
#   scripts/eval_ood.sh                     # every logs/*/ with final_weights.pt
#   scripts/eval_ood.sh logs/<uuid> ...     # chosen runs
#   DRY_RUN=1 scripts/eval_ood.sh           # print the command only
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
caller_dir="${PWD}"
cd "${PROJECT_ROOT}"

# ============================== SETTINGS ==============================
GPU_ID=0
DATA_ROOT="${PROJECT_ROOT}/../wise-ft/datasets"   # holds the four OOD folders
BATCH_SIZE=512
NUM_WORKERS=8
# ======================================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONNOUSERSITE=1
if [[ -x "${PROJECT_ROOT}/env/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/env/bin/python"
else
    PYTHON="$(command -v python)"
fi

runs=()
for run in "$@"; do
    [[ "${run}" == /* ]] || run="${caller_dir}/${run}"
    runs+=("${run}")
done
if (( ${#runs[@]} == 0 )); then
    for weights in logs/*/final_weights.pt; do
        [[ -f "${weights}" ]] && runs+=("$(dirname "${weights}")")
    done
fi
(( ${#runs[@]} )) || { echo "No finished runs (logs/*/final_weights.pt)" >&2; exit 2; }

command=("${PYTHON}" eval_ood.py "${runs[@]}" --data-root "${DATA_ROOT}"
         --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi
exec "${command[@]}"
