# ImageNet

## Environment

```bash
conda create -n condsam python=3.12 -y
conda activate condsam
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

Requires NVIDIA GPUs. Launchers use the active Conda environment or `env/bin/python`.

## Datasets

Existing download commands for CIFAR-10/100, TinyImageNet, CIFAR-C, and SVHN:

```bash
python scripts/prepare_datasets.py --dataset all
bash scripts/ood/prepare_datasets.sh
```

These scripts do **not** download ImageNet-1K. Prepare ImageNet separately, with validation images sorted into class folders:

```text
datasets/ImageNet/
├── train/<class_id>/*.JPEG
└── val/<class_id>/*.JPEG
```

The loader requires 1,281,167 training images and 50,000 validation images. Set `DATASET_ROOT` in the launcher to the directory containing `ImageNet/`, or directly containing `train/` and `val/`.

## Configure and run

Edit the settings at the top of [scripts/run_imagenet.sh](scripts/run_imagenet.sh):

```bash
GPU_ID="0"
NPROC_PER_NODE=1
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="ImageNet"
export WANDB_RUN_GROUP="imagenet"
export WANDB_MODE="online"

CONFIG_PATH="configs/ImageNet_ResNet50.ini"
OPTIMIZER="CondSAM"
EPOCHS=100
EVAL_EVERY=20
TOP_K=(1 5)

BATCH_SIZE=""
LEARNING_RATE=""
WEIGHT_DECAY=""
MOMENTUM=""
RHO_LIST=(0.1 0.15 0.5 1 1.5)
CONDSAM_BETA_LIST=(0.99 0.995)
CONDSAM_EPS=1e-8
CONDSAM_POWER_ITERS=""
```

Empty values use the `.ini` configuration. CondSAM runs all 10 rho–beta combinations. `NPROC_PER_NODE` must match the number of GPUs in `GPU_ID`. Use an environment variable for the API key or an existing W&B login.

```bash
bash scripts/run_imagenet.sh
```
