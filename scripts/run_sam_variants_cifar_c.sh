#!/usr/bin/env bash
set -euo pipefail

# GPU settings: edit these values directly.
GPU_ID="0"             # GPU indices, e.g. "0" or "0,1".
NPROC_PER_NODE=1        # Number of training GPUs; must match GPU_ID.

# Paste your W&B API key between the quotes.
export WANDB_API_KEY=""

# Train each baseline, then evaluate its final checkpoint at severities 1..5.
# Usage: bash scripts/run_sam_variants_cifar_c.sh [config.ini ...]
# Defaults to five backbones on both CIFAR datasets. CondSAM excluded.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"

# Prefer the activated Conda environment; retain the local env as a fallback.
PYTHON="${PYTHON:-${CONDA_PREFIX:-${PROJECT_ROOT}/env}/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
DEVICE="${DEVICE:-auto}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}" # OOD only; 0 evaluates all examples.
export WANDB_PROJECT="variants-cifar-c"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SAM_DATASET_ROOT="${DATA_ROOT}"
export SAM_CONSOLE_EVAL_ONLY=1

STAMP="$(date -u +%Y%m%dT%H%M%S_%NZ)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/logs/variants_cifar_c_${STAMP}_$$}"
[[ "${RUN_ROOT}" == /* ]] || RUN_ROOT="${PROJECT_ROOT}/${RUN_ROOT}"
TRAIN_ROOT="${RUN_ROOT}/train"
RESULTS_ROOT="${RUN_ROOT}/cifar_c_results"
CONFIGS=("$@")
if (( ${#CONFIGS[@]} == 0 )); then
    CONFIGS=(
        configs/CIFAR10_ResNet50.ini
        configs/CIFAR10_WRN16_4.ini
        configs/CIFAR10_PyramidNet110.ini
        configs/CIFAR10_WRN28_10.ini
        configs/CIFAR10_DenseNet121.ini
        configs/CIFAR100_ResNet50.ini
        configs/CIFAR100_WRN16_4.ini
        configs/CIFAR100_PyramidNet110.ini
        configs/CIFAR100_WRN28_10.ini
        configs/CIFAR100_DenseNet121.ini
    )
fi
OPTIMIZERS=(SGD SAM ASAM MSAM ESAM FriendlySAM FisherSAM MongeSAM GSAM)

[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 2; }
# Do not reuse an old batch: evaluation must use this invocation's checkpoint.
[[ ! -e "${RUN_ROOT}" ]] || { echo "RUN_ROOT already exists: ${RUN_ROOT}; choose a new directory." >&2; exit 2; }

# Read the actual dataset field, not the config filename. Check everything before training.
DATASETS=()
for config in "${CONFIGS[@]}"; do
    dataset="$("${PYTHON}" - "${config}" <<'PY'
import configparser
import sys
config = configparser.ConfigParser()
if not config.read(sys.argv[1]):
    raise SystemExit(f'Missing config: {sys.argv[1]}')
dataset = config.get('general', 'dataset')
if dataset not in ('CIFAR10', 'CIFAR100'):
    raise SystemExit(f'Unsupported dataset: {dataset}; expected CIFAR10 or CIFAR100')
print(dataset)
PY
    )"
    DATASETS+=("${dataset}")
    if [[ "${dataset}" == CIFAR10 ]]; then
        corruption_dir="${DATA_ROOT}/CIFAR-10-C"
    else
        corruption_dir="${DATA_ROOT}/CIFAR-100-C"
    fi
    for file in gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur \
        zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression labels; do
        [[ -f "${corruption_dir}/${file}.npy" ]] || {
            echo "Missing ${corruption_dir}/${file}.npy; prepare with scripts/ood/prepare_datasets.sh" >&2
            exit 2
        }
    done
done

mkdir -p "${TRAIN_ROOT}" "${RESULTS_ROOT}"
failures=0
passed_train=0
passed_eval=0
for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[${index}]}"
    dataset="${DATASETS[${index}]}"
    setting_name="${dataset,,}_c"
    config_tag="$(basename "${config}" .ini)"
    for optimizer in "${OPTIMIZERS[@]}"; do
        case_name="${index}_${config_tag}_${optimizer}"
        case_root="${TRAIN_ROOT}/${case_name}"
        export WANDB_RUN_GROUP="train_${dataset,,}"
        export WANDB_NAME="${case_name}__seed42"
        cmd=("${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"
             main.py --ifile "${config}" --optimizer "${optimizer}"
             --logDir "${case_root}" --contin False --truncate False --rndSeed False)
        # Otherwise use epochs, evalEvery and all optimizer parameters from the config.
        if [[ -n "${EPOCHS:-}" ]]; then cmd+=(--epochs "${EPOCHS}"); fi
        if [[ -n "${EVAL_EVERY:-}" ]]; then cmd+=(--evalEvery "${EVAL_EVERY}"); fi
        echo "[TRAIN] ${case_name} seed=42 wandb=${WANDB_PROJECT}/${WANDB_RUN_GROUP}/${WANDB_NAME}"
        if run_training "${cmd[@]}"; then
            passed_train=$((passed_train + 1))
        else
            status=$?
            failures=$((failures + 1))
            echo "[FAIL] train ${case_name} exit=${status}; skipping its OOD evaluation." >&2
            continue
        fi

        # main.py generates logSubDir itself. The isolated parent identifies the exact new run.
        shopt -s nullglob
        checkpoints=("${case_root}"/*/final.model)
        if (( ${#checkpoints[@]} != 1 )); then
            failures=$((failures + 1))
            echo "[FAIL] Expected exactly one final.model under ${case_root}" >&2
            continue
        fi
        run_dir="${checkpoints[0]%/final.model}"
        for severity in 1 2 3 4 5; do
            export WANDB_RUN_GROUP="ood_${setting_name}_sev${severity}"
            export WANDB_NAME="${case_name}__${setting_name}__sev${severity}"
            output_dir="${RESULTS_ROOT}/${WANDB_NAME}"
            echo "[OOD] ${case_name} severity=${severity} wandb=${WANDB_PROJECT}/${WANDB_RUN_GROUP}/${WANDB_NAME}"
            if run_training "${PYTHON}" scripts/evaluate_ood.py \
                --setting "configs/ood/${setting_name}.json" \
                --run-dir "${run_dir}" --checkpoint final.model \
                --data-root "${DATA_ROOT}" --device "${DEVICE}" \
                --max-examples "${MAX_EXAMPLES}" --severities "${severity}" \
                --output-dir "${output_dir}"; then
                passed_eval=$((passed_eval + 1))
            else
                status=$?
                failures=$((failures + 1))
                echo "[FAIL] OOD ${case_name} severity=${severity} exit=${status}" >&2
            fi
        done
    done
done

"${PYTHON}" scripts/ood/summarize_cifar_c_results.py \
    --results-root "${RESULTS_ROOT}" \
    --markdown "${RUN_ROOT}/cifar_c_summary.md" --csv "${RUN_ROOT}/cifar_c_summary.csv"
echo "[SUMMARY] trained=${passed_train} evaluated=${passed_eval} failures=${failures}"
echo "[OUTPUT] ${RUN_ROOT}"
if (( failures )); then exit 1; fi
