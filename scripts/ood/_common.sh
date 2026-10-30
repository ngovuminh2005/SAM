PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PROJECT_ROOT}/env/bin/python"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}" WANDB_PROJECT WANDB_MODE
export WANDB_RUN_GROUP="ood_$(basename "${SETTING}" .json)"
if (( ${#RUN_DIRS[@]} == 0 )); then
    echo 'Set RUN_DIRS in this task script, or pass trained run directories as positional arguments.' >&2
    exit 2
fi
for directory in "${RUN_DIRS[@]}"; do
    "${PYTHON}" scripts/evaluate_ood.py --setting "${SETTING}" --run-dir "${directory}" \
        --checkpoint "${CHECKPOINT}" --data-root "${DATA_ROOT}" --device "${DEVICE}" \
        --max-examples "${MAX_EXAMPLES}"
done
