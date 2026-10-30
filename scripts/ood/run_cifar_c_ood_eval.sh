#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-${PROJECT_ROOT}/env/bin/python}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/logs/ood_eval}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SEVERITIES="${SEVERITIES:-1 2 3 4 5}"
WANDB_PROJECT="${WANDB_PROJECT:-cifar-c}"
WANDB_MODE="${WANDB_MODE:-online}"
STAMP="$(date -u +%Y%m%dT%H%M%S_%NZ)"
RESULTS_ROOT="${RESULTS_ROOT:-${RUN_ROOT}/cifar_c_results_${STAMP}}"
SUMMARY_MD="${SUMMARY_MD:-${RUN_ROOT}/cifar_c_summary.md}"
SUMMARY_CSV="${SUMMARY_CSV:-${RUN_ROOT}/cifar_c_summary.csv}"

[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 2; }
[[ -d "${RUN_ROOT}" ]] || { echo "Missing checkpoint directory: ${RUN_ROOT}" >&2; exit 2; }
read -r -a severity_args <<< "${SEVERITIES}"
((${#severity_args[@]} > 0)) || { echo "SEVERITIES must contain values from 1 to 5" >&2; exit 2; }
declare -A seen_severities=()
for severity in "${severity_args[@]}"; do
    [[ "${severity}" =~ ^[1-5]$ ]] || { echo "Invalid severity: ${severity}; expected 1 through 5" >&2; exit 2; }
    [[ -z "${seen_severities[${severity}]:-}" ]] || { echo "Duplicate severity: ${severity}" >&2; exit 2; }
    seen_severities[${severity}]=1
done
check_cifar_c() {
    local directory="$1"
    local label="$2"
    local file
    for file in gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur \
        zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression labels; do
        [[ -f "${directory}/${file}.npy" ]] || {
            echo "Missing ${directory}/${file}.npy (${label}); prepare CIFAR-C with scripts/ood/prepare_datasets.sh" >&2
            return 1
        }
    done
}

check_cifar_c "${DATA_ROOT}/CIFAR-10-C" CIFAR-10-C
check_cifar_c "${DATA_ROOT}/CIFAR-100-C" CIFAR-100-C

shopt -s nullglob
declare -a all_runs=()
for dataset in CIFAR10 CIFAR100; do
    run_dirs=("${RUN_ROOT}/${dataset}_CondSAM"*)
    ((${#run_dirs[@]} > 0)) || { echo "No ${dataset} CondSAM run folders under ${RUN_ROOT}" >&2; exit 2; }
    for run_dir in "${run_dirs[@]}"; do
        [[ -d "${run_dir}" ]] || continue
        [[ -f "${run_dir}/params.json" && -f "${run_dir}/final.model" ]] || {
            echo "Expected params.json and final.model in ${run_dir}" >&2
            exit 2
        }
        all_runs+=("${run_dir}")
    done
done

mkdir -p "${RESULTS_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}" WANDB_PROJECT WANDB_MODE
failures=0

for run_dir in "${all_runs[@]}"; do
    run_name="$(basename "${run_dir}")"
    if [[ "${run_name}" == CIFAR10_* ]]; then
        setting="configs/ood/cifar10_c.json"
        setting_name="cifar10_c"
        group="ood_cifar10_c"
    else
        setting="configs/ood/cifar100_c.json"
        setting_name="cifar100_c"
        group="ood_cifar100_c"
    fi

    for severity in "${severity_args[@]}"; do
        export WANDB_RUN_GROUP="${group}_sev${severity}"
        export WANDB_NAME="${run_name}__${setting_name}__sev${severity}"
        output_dir="${RESULTS_ROOT}/${WANDB_NAME}"
        echo "[OOD] ${run_name} setting=${setting_name} severity=${severity} wandb=${WANDB_PROJECT}/${WANDB_RUN_GROUP}/${WANDB_NAME}"
        if "${PYTHON}" scripts/evaluate_ood.py \
            --setting "${setting}" --run-dir "${run_dir}" --checkpoint final.model \
            --data-root "${DATA_ROOT}" --device "${DEVICE}" \
            --max-examples "${MAX_EXAMPLES}" --severities "${severity}" \
            --output-dir "${output_dir}"; then
            echo "[OK] ${output_dir}/results.json"
        else
            status=$?
            failures=$((failures + 1))
            echo "[FAIL] ${run_name} setting=${setting_name} severity=${severity} exit=${status}" >&2
        fi
    done
done

"${PYTHON}" scripts/ood/summarize_cifar_c_results.py \
    --results-root "${RESULTS_ROOT}" --markdown "${SUMMARY_MD}" --csv "${SUMMARY_CSV}"
echo "[TABLE] ${SUMMARY_MD}"
echo "[CSV] ${SUMMARY_CSV}"
echo "[RESULTS] ${RESULTS_ROOT}"

if (( failures )); then
    echo "${failures} evaluation(s) failed; see logs above." >&2
    exit 1
fi
