#!/usr/bin/env bash
set -euo pipefail
export SAM_CONSOLE_EVAL_ONLY=1 # Print training metrics/time only on eval epochs.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/lib/sweep_process.sh"
export PATH="${PROJECT_ROOT}/env/bin:${PATH}"
export SAM_DATASET_ROOT="${SAM_DATASET_ROOT:-${PROJECT_ROOT}/datasets}"
export WANDB_PROJECT="CondSAM"
export WANDB_MODE="${WANDB_MODE:-online}"


# Usage:
#   bash scripts/run_model_sweep.sh [log_prefix] [epochs] [cifar_profile]
# Example:
#   bash scripts/run_model_sweep.sh model_sweep 5 cifar10

LOG_PREFIX=${1:-model_sweep}
EPOCHS=${2:-5}
CIFAR_PROFILE=${3:-cifar100}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_$RANDOM}

# Select GPU by index (0 or 1). Override with: GPU_ID=1 bash ...
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

case "${CIFAR_PROFILE}" in
    cifar10)
        CIFAR_WRN_CFG="configs/CIFAR10_WRN16_4.ini"
        CIFAR_RESNET_CFG="configs/CIFAR10_ResNet50.ini"
        CIFAR_VGG_CFG="configs/CIFAR10_VGG16.ini"
        CIFAR_PYRAMID_CFG="configs/CIFAR10_PyramidNet110.ini"
        ;;
    cifar100)
        CIFAR_WRN_CFG="configs/CIFAR100_WRN16_4.ini"
        CIFAR_RESNET_CFG="configs/CIFAR100_ResNet50.ini"
        CIFAR_VGG_CFG="configs/CIFAR100_VGG16.ini"
        CIFAR_PYRAMID_CFG="configs/CIFAR100_PyramidNet110.ini"
        ;;
    *)
        echo "Unsupported cifar_profile='${CIFAR_PROFILE}'. Use 'cifar10' or 'cifar100'."
        exit 1
        ;;
esac

IMGNET_WRN_CFG="configs/ImageNet_WRN16_4.ini"
IMGNET_RESNET_CFG="configs/ImageNet_ResNet50.ini"
IMGNET_VGG_CFG="configs/ImageNet_VGG16.ini"
IMGNET_PYRAMID_CFG="configs/ImageNet_PyramidNet50.ini"
IMGNET_VIT_CFG="configs/ImageNet_ViT.ini"

run_case() {
    local name="$1"
    local config_path="$2"
    shift 2

    local run_name="${LOG_PREFIX}_${name}"
    if [[ -n "${RUN_SUFFIX}" ]]; then
        run_name="${run_name}_${RUN_SUFFIX}"
    fi

    echo "[RUN] ${name}"
    run_training python -m torch.distributed.run main.py \
        --ifile "${config_path}" \
        --epochs "${EPOCHS}" \
        --logSubDir "${run_name}" \
        "$@"
}

# CIFAR models
run_case wrn_sgd "${CIFAR_WRN_CFG}" --model WRN --optimizer SGD
run_case resnet_sgd "${CIFAR_RESNET_CFG}" --model resnet --optimizer SGD --depth 50
run_case vgg16_sgd "${CIFAR_VGG_CFG}" --model VGG --optimizer SGD --depth 16 --BN True
run_case pyramidnet_sgd "${CIFAR_PYRAMID_CFG}" --model PyramidNet --optimizer SGD --depth 110 --pyramidAlpha 270 --pyramidBottleneck True

# ImageNet models
run_case wrn16_4_imagenet_sgd "${IMGNET_WRN_CFG}" --model WRN --optimizer SGD --depth 16 --widthFactor 4
run_case resnet50_imagenet_sgd "${IMGNET_RESNET_CFG}" --model resnet --optimizer SGD --depth 50
run_case vgg16_imagenet_sgd "${IMGNET_VGG_CFG}" --model VGG --optimizer SGD --depth 16 --BN True
run_case pyramidnet50_imagenet_sgd "${IMGNET_PYRAMID_CFG}" --model PyramidNet --optimizer SGD --depth 50 --pyramidAlpha 270 --pyramidBottleneck True
run_case vit_s_imagenet_adamw "${IMGNET_VIT_CFG}" --model ViT --optimizer AdamW --ViTSize S --ViTPatchsize 32
