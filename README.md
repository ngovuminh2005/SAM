# ImageNet Training with SAM Optimizers

Train ImageNet with configurable optimizers, GPU selection, hyperparameters, and multiple top-k accuracy metrics logged to W&B.

## Environment

```bash
conda create -n condsam python=3.12 -y
conda activate condsam
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

Training requires NVIDIA GPUs. The launcher uses the active Conda environment, otherwise `env/bin/python`. Set `PYTHON` to override the interpreter.

## ImageNet data

Prepare the full ImageNet-1K dataset in ImageFolder layout, with validation images sorted into class folders:

```text
datasets/ImageNet/
├── train/<class_id>/*.JPEG
└── val/<class_id>/*.JPEG
```

The loader expects 1,281,167 training images and 50,000 validation images. Set `DATASET_ROOT` in the launcher to the directory containing `ImageNet/`, or to a directory containing `train/` and `val/` directly.

## Configure and run

Edit [scripts/run_imagenet.sh](scripts/run_imagenet.sh). GPU and W&B settings are at the top:

```bash
GPU_ID="0,1"
NPROC_PER_NODE=2
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="ImageNet"
export WANDB_RUN_GROUP="imagenet"
export WANDB_MODE="online"
```

Provide your key through `WANDB_API_KEY` in the environment, or use an existing `wandb login`. Do not commit your actual key. `NPROC_PER_NODE` must equal the number of GPU indices. Batch size is global and divided across workers. The launcher rejects busy GPUs unless `ALLOW_SHARED_GPU=1`.

Choose the model, optimizer, evaluation interval, and top-k metrics in the same file:

```bash
CONFIG_PATH="configs/ImageNet_ResNet50.ini"
OPTIMIZER="CondSAM"
EPOCHS=100
EVAL_EVERY=20
TOP_K=(1 5 10)

BATCH_SIZE=256
LEARNING_RATE=0.1
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
RHO_LIST=(0.1 0.15 0.5 1 1.5)
CONDSAM_BETA_LIST=(0.99 0.995)
CONDSAM_EPS=1e-8
CONDSAM_POWER_ITERS=5
```

Edit the literal values and arrays directly in the `.sh` file. The default CondSAM sweep runs all **10 combinations** of the five `RHO_LIST` values and two `CONDSAM_BETA_LIST` values, sequentially. Each run has a separate W&B name containing its rho and beta.

Empty `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `MOMENTUM`, and `CONDSAM_POWER_ITERS` use the selected `.ini` values. `RHO_LIST=()` uses the configured radius for one sweep position; `CONDSAM_BETA_LIST=()` uses the configured beta. A single-element list runs just that value. Rho values override the selected optimizer's method-specific radius. CondSAM beta settings apply to `CondSAM`, `CondSam1`, `CondSam2`, and `CondSAM_adam`; other SAM optimizers sweep rho only. SGD, Adam, and AdamW run once and ignore both lists.

Available ImageNet configurations:

| Model | Config |
| --- | --- |
| ResNet-50 | `configs/ImageNet_ResNet50.ini` |
| WRN-16-4 | `configs/ImageNet_WRN16_4.ini` |
| WRN-28-10 | `configs/ImageNet_WRN28_10.ini` |
| PyramidNet-50 | `configs/ImageNet_PyramidNet50.ini` |
| VGG-16 | `configs/ImageNet_VGG16.ini` |
| ViT | `configs/ImageNet_ViT.ini` |

`OPTIMIZER` accepts `SGD`, `Adam`, `AdamW`, `SAM`, `ASAM`, `MSAM`, `ESAM`, `lookSAM`, `FriendlySAM`, `FisherSAM`, `MongeSAM`, `GSAM`, `BayesianSAM`, `CondSAM`, `CondSam1`, `CondSam2`, `CondSAM_adam`, `AdamW_SAM`, `Adam_SAM`, `AdamW_MSAM`, `AdamW_FriendlySAM`, and `Adam_FriendlySAM`. Names are case-sensitive. Configure other optimizer-specific options in the `.ini` file or `EXTRA_ARGS` in the launcher.

Preview the command, then start training:

```bash
DRY_RUN=1 bash scripts/run_imagenet.sh
bash scripts/run_imagenet.sh
```

Additional `main.py` arguments can be passed directly:

```bash
bash scripts/run_imagenet.sh --learningRate 0.05 --batchSize 128 --topK 1 5 20
```

Settings resolve in this order: `.ini` configuration, launcher options, launcher `EXTRA_ARGS`, then command-line arguments. The launcher always selects `OPTIMIZER`, `EPOCHS`, `EVAL_EVERY`, and `TOP_K`; edit those in the `.sh` file to change the defaults. Evaluation runs every `EVAL_EVERY` epochs and at the final epoch.

## Top-k metrics and outputs

Each k must be a positive integer no larger than the number of classes (1,000 for ImageNet). Duplicate k values are logged once. `TOP_K=()` disables additional top-k metrics; the original `accuracy` metric remains top-1. You can also set `topK = 1 5 10` in `.ini` files when invoking `main.py` directly.

For `TOP_K=(1 5 10)`, W&B receives:

- `train/accuracy_top1`, `train/accuracy_top5`, `train/accuracy_top10` each training epoch.
- `test/accuracy_top1`, `test/accuracy_top5`, `test/accuracy_top10` each evaluation epoch, using the validation set.

Accuracy values on W&B and in HDF5 are fractions from 0 to 1; terminal output uses percentages. The original `train/accuracy`, `test/accuracy`, and `best_test_accuracy` remain available. Missing W&B login triggers offline logging.

Artifacts are saved under `logs/<dataset>_<optimizer>_<model>_rho<rho>_lr<lr>/`, with a numbered suffix for repeated runs. The exact directory is printed as `[RUN_DIR]`. Each run includes `params.json`, `data.hdf5`, `terminal.log`, `training_time.txt`, `final.model`, and local W&B files. `--logDir` changes the parent output directory.

## CondSAM parameter sweeps

For CondSAM sweeps that also vary power iterations or repeats, edit [scripts/run_condsam_rho.sh](scripts/run_condsam_rho.sh):

```bash
CONFIG_PATH="configs/ImageNet_ResNet50.ini"
RHO_LIST=(0.1 0.15 0.5 1 1.5)
CONDSAM_BETA_LIST=(0.99 0.995)
CONDSAM_POWER_ITERS_LIST=(1 5)
TOP_K=(1 5)
REPEATS=1
```

Run the sweep with:

```bash
bash scripts/run_condsam_rho.sh configs/ImageNet_ResNet50.ini imagenet_condsam
```

This sweep always uses CondSAM. Use `run_imagenet.sh` to select another optimizer. Both launchers expose GPU and W&B settings at the top.
