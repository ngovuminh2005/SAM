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

Download and prepare ImageNet-1K:

```bash
bash scripts/download_datasets.sh
```

This downloads train/validation, verifies checksums, and prepares `datasets/ImageNet/{train,val}/<class_id>/*.JPEG`. Allow 300 GiB free during preparation. Rerun the command to resume an interrupted download. If ImageNet requires authentication, set `IMAGENET_COOKIES` to your exported cookies file.

For another disk, use `DATA_ROOT=/path/to/datasets bash scripts/download_datasets.sh`, then set `DATASET_ROOT` in the training launcher to that same directory.

Edit `DATASETS` in the download script or pass dataset names to select `imagenet`, `cifar10`, `cifar100`, `tiny-imagenet`, `cifar10_c`, `cifar100_c`, or `svhn`.

ImageNet downloader adapted from [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5/blob/af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce/data/scripts/get_imagenet.sh); see [attribution and license](third_party/ultralytics/NOTICE.md).

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
