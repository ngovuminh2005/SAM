# ImageNet (FFCV) with SAM-style optimizers

ResNet ImageNet training built on [ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet), extended with SAM-style
perturbation policies (`sam`, `asam`, `condsam`, `friendlysam`, `fishersam`, `mongesam`) and W&B logging.

## Environment

```bash
conda create -y -p ./env -c conda-forge --override-channels \
    python=3.10 pip pkg-config libjpeg-turbo "opencv=4.10" "libpng<1.6.45"
conda env config vars set -p ./env PYTHONNOUSERSITE=1 PKG_CONFIG_PATH=$PWD/env/lib/pkgconfig
conda activate ./env
python -m pip install -r requirements.txt     # torch 2.4.1 (CUDA 12.1), ffcv 1.0.2, wandb, ...
wandb login                                   # or export WANDB_API_KEY
```

Requires NVIDIA GPUs. The environment must live at `./env`: the launcher runs `env/bin/python`.

- `ffcv` is compiled against OpenCV 4 and libjpeg-turbo from conda; its build does not find OpenCV 5.
- `libpng<1.6.45`: torchvision 0.19 bundles an older libpng, and newer OpenCV builds then fail with
  `undefined symbol: png_set_cICP`.
- `PYTHONNOUSERSITE=1` keeps packages in `~/.local` out of the environment.
- CuPy is not needed; images are normalized on the GPU with torch.

## Datasets

**1. Download ImageNet-1K (ILSVRC2012)**

```bash
bash prepare_imagenet.sh            # both splits; --train or --val for one
```

This downloads `ILSVRC2012_img_train.tar` (in 4 parallel parts) and `ILSVRC2012_img_val.tar`, verifies the official
MD5 checksums, extracts the per-class train archives, sorts validation images into class folders, and produces
`imagenet/{train,val}/<wnid>/*.JPEG` (1000 classes; 1,281,167 train and 50,000 val images; ~146 GiB).
Allow 300 GiB free during preparation. Rerun the command to resume an interrupted download; complete splits are
skipped.

- Archives come from a Hugging Face mirror of the original files. To use image-net.org instead:
  `IMAGENET_SOURCE=official IMAGENET_COOKIES=/path/to/cookies.txt bash prepare_imagenet.sh`.
- For another disk: `DATA_ROOT=/path/to/datasets bash prepare_imagenet.sh` (writes `/path/to/datasets/imagenet`).

**2. Convert to FFCV**

```bash
mkdir -p data
IMAGENET_DIR=$PWD/imagenet WRITE_DIR=$PWD/data ./write_imagenet.sh 500 1.0 90
```

Arguments are `<max side length> <JPEG fraction> <JPEG quality>`. This writes `data/train_500_1.0_90.ffcv`
(68.5 GB) and `data/val_500_1.0_90.ffcv` (3.1 GB) in ~7 minutes on 20 CPUs. Images are resized to a longer side of
at most 500 px and all are stored as quality-90 JPEG.

| Setting | Train file | Notes |
|---|---|---|
| `500 1.0 90` | 68.5 GB | used here; fits in RAM for `in_memory=1` |
| `500 0.50 90` | ~326 GB | original ResNet-50 setting (half stored as raw pixels) |
| `400 0.10 90` | ~388 GB | original ResNet-18 setting |

With 18 loader workers, `1.0` and `0.50` load equally fast (~10.5k img/s for the ResNet-18 pipeline on 20 CPUs).
The raw `imagenet/` folder is not used for training and can be deleted after conversion.

## Configure and run

Edit the settings at the top of [scripts/train_rn50.sh](scripts/train_rn50.sh):

```bash
GPUS_PER_NODE=1              # GPU processes on this node; > 1 enables DDP
GPU_IDS=(0)                  # which GPUs, e.g. (0 1 2 3); one id per process

EPOCHS=(88)                  # any; 16 24 32 40 56 88 reproduce rn50_configs/ exactly
METHODS=(sam)                # sam asam condsam friendlysam fishersam mongesam no_sam
RHOS=(0.01 0.05)             # perturbation radius (not used by no_sam)

# Only used by the matching method:
FRIENDLYSAM_SIGMAS=(1.0)
FRIENDLYSAM_LAMBDAS=(0.9)
FISHERSAM_ETAS=(1.0)
CONDSAM_BETAS=(0.95)
CONDSAM_POWER_ITERS=(1)

SEEDS=(42)
```

Every combination runs in turn; the example runs 2 jobs (`sam` with two rho values). `GPU_IDS` must list
`GPUS_PER_NODE` ids (DDP is single-node). Fixed settings (per-GPU batch 512, data paths,
`WANDB_PROJECT=ffcv-imagenet`) are in the `FIXED` block below. The learning rate follows the original 8-GPU recipe
(1.7 at a global batch of 4096) scaled linearly with the global batch: 0.2125 on one GPU.

```bash
DRY_RUN=1 bash scripts/train_rn50.sh     # print the commands
bash scripts/train_rn50.sh               # run
```

Arguments after the script name are passed to every run, e.g. `bash scripts/train_rn50.sh --wandb.mode=disabled`.
On one H100, ResNet-50 + SAM trains at ~3.2k img/s at 160 px (~7 min per epoch).

### Outputs

Each run writes `logs/<uuid>/` with `params.json`, a JSON-lines `log`, and `final_weights.pt`. W&B runs of one sweep
share a group and log:

- `train/loss`, `train/lr` every 50 steps;
- `val/top_1`, `val/top_5`, `val/loss` and `epoch/*` (train loss, time, throughput, resolution) after every epoch;
- `final/*` after training.

Evaluation runs after every epoch on the full 50k validation set at 256 px with flip test-time augmentation.

## Training recipe

From ffcv-imagenet ([rn50_configs/](rn50_configs/)): SGD (momentum 0.9, weight decay 1e-4 except on parameters
whose names contain `bn`), label smoothing 0.1, BlurPool, triangular LR (peak at epoch 2, zero at the last epoch), and
progressive resizing 160 → 192 px late in training. The LR and resolution schedules stretch with `EPOCHS`, so every
epoch count is a complete schedule; there is no early stopping.

A SAM step does a forward/backward pass, perturbs the weights by the policy's δ, does a second forward/backward
pass at `w + δ` (BatchNorm running statistics frozen), restores `w`, and steps SGD with that gradient. δ is
normalized with one L2 norm over all perturbed parameters (all trainable parameters by default; filter with
`--sam.include` / `--sam.exclude` regexes). CondSAM additionally conditions each `Conv2d`/`Linear` weight with
Kronecker factors collected on the first pass. SAM policies run in bf16 autocast; the plain baseline uses fp16.

A single run without the launcher stacks a policy config on a base config:

```bash
python train_imagenet.py -C rn50_configs/rn50_88_epochs.yaml -C sam_configs/sam.yaml \
    --training.distributed=0 --dist.world_size=1 --lr.lr=0.2125 \
    --data.train_dataset=data/train_500_1.0_90.ffcv --data.val_dataset=data/val_500_1.0_90.ffcv \
    --data.num_workers=18 --data.in_memory=1 --logging.folder=logs --logging.log_level=1 \
    --wandb.mode=online --wandb.project=ffcv-imagenet
```

## Credits

Training script and configurations from [ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet)
(Apache-2.0, see [LICENSE](LICENSE)). If you use this setup, cite:

```
@misc{leclerc2022ffcv,
    author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
    title = {ffcv},
    year = {2022},
    howpublished = {\url{https://github.com/libffcv/ffcv/}}
}
```

Perturbation policies are ported from the wise-ft project. The validation folder mapping is
[`valprep.sh`](https://github.com/soumith/imagenetloader.torch) from soumith/imagenetloader.torch.
