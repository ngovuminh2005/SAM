# ImageNet

## Cài môi trường

```bash
conda create -n condsam python=3.12 -y
conda activate condsam
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

Cần GPU NVIDIA. Script dùng Python của môi trường Conda đang kích hoạt, hoặc `env/bin/python`.

## Dataset

Chuẩn bị ImageNet-1K theo cấu trúc, ảnh validation phải được chia vào thư mục lớp:

```text
datasets/ImageNet/
├── train/<class_id>/*.JPEG
└── val/<class_id>/*.JPEG
```

Cần đủ 1.281.167 ảnh train và 50.000 ảnh validation. Chỉnh `DATASET_ROOT` trong script nếu dữ liệu nằm ở nơi khác: trỏ tới thư mục chứa `ImageNet/`, hoặc chứa trực tiếp `train/` và `val/`.

## Chỉnh tham số và chạy

Chỉnh các giá trị ở đầu [scripts/run_imagenet.sh](scripts/run_imagenet.sh):

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

Giá trị rỗng dùng cấu hình trong `.ini`. CondSAM chạy đủ 10 tổ hợp rho–beta; `NPROC_PER_NODE` phải khớp số GPU trong `GPU_ID`. API key lấy từ biến môi trường hoặc W&B login hiện có.

```bash
bash scripts/run_imagenet.sh
```
