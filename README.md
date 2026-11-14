# ImageNet (FFCV) with SAM-style optimizers

ResNet ImageNet training built on [ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet), extended with SAM-style
perturbation policies (`sam`, `asam`, `condsam`, `condsam1`, `condsam2`, `condsam3`, `friendlysam`, `fishersam`, `mongesam`) and W&B logging.

## Environment

```bash
conda create -y -p ./env -c conda-forge --override-channels \
    python=3.10 pip pkg-config libjpeg-turbo "opencv=4.10" "libpng<1.6.45"
conda env config vars set -p ./env PYTHONNOUSERSITE=1 PKG_CONFIG_PATH=$PWD/env/lib/pkgconfig
conda activate ./env
python -m pip install -r requirements.txt     # torch 2.4.1 (CUDA 12.1), ffcv 1.0.2, wandb, ...
wandb login                                   # or export WANDB_API_KEY
```

Requires NVIDIA GPUs with a driver that supports CUDA 12.1 (>= 525). The launcher uses `./env/bin/python` if it
exists, otherwise `python` from the active environment.

- `ffcv` is compiled against OpenCV 4 and libjpeg-turbo from conda; its build does not find OpenCV 5.
- `libpng<1.6.45`: torchvision 0.19 bundles an older libpng, and newer OpenCV builds then fail with
  `undefined symbol: png_set_cICP`.
- `PYTHONNOUSERSITE=1` keeps packages in `~/.local` out of the environment.
- CuPy is not needed; images are normalized on the GPU with torch.

### W&B

Runs log to W&B online, so a new machine needs your API key (https://wandb.ai/authorize):

```bash
wandb login                          # once per machine; stores the key in ~/.netrc
export WANDB_API_KEY=<your-key>      # or, per shell (clusters, containers)
```

Without a key, set `--wandb.mode=offline` (sync later with `wandb sync logs/<uuid>/wandb/offline-run-*`) or
`--wandb.mode=disabled`.

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
IMAGENET_DIR=$PWD/imagenet WRITE_DIR=$PWD/data ./write_imagenet.sh 500 0.50 90
```

Arguments are `<max side length> <JPEG fraction> <JPEG quality>`. Images are resized to a longer side of at most
500 px; each is stored as a quality-90 JPEG with probability `<JPEG fraction>` and as raw pixels otherwise (~550 KB
each, about 10x a JPEG). This is the original ffcv-imagenet ResNet-50 setting and writes
`data/{train,val}_500_0.50_90.ffcv`.

| Setting | Train file | Notes |
|---|---|---|
| `500 0.50 90` | ~360–390 GB (estimated) | default; half the images are not re-compressed |
| `500 1.0 90` | 68.5 GB (measured) | optional; every image re-compressed as JPEG; ~7 min to write on 20 CPUs |

The launcher defaults to `DATA_TAG=500_0.50_90`. To use `1.0`, write it with `./write_imagenet.sh 500 1.0 90` and
set `DATA_TAG=500_1.0_90` in [scripts/train_rn50.sh](scripts/train_rn50.sh). With `IN_MEMORY=auto` (default) the
train file is cached in RAM when it fits in 80% of the machine's memory and read from disk otherwise; set `0` or `1`
to force either.

Converting needs the raw `imagenet/` folder (~146 GiB) and the output on disk at the same time: about 520 GB free
for `0.50`. The raw folder is not used for training and can be deleted afterwards. With 18 loader workers and the data
in RAM, `1.0` and `0.50` load equally fast (~10.5k img/s for the ResNet-18 pipeline on 20 CPUs); reading `0.50` from
disk has not been benchmarked.

## Configure and run

Edit the settings at the top of [scripts/train_rn50.sh](scripts/train_rn50.sh):

```bash
GPUS_PER_NODE=1              # GPU processes on this node; > 1 enables DDP
GPU_IDS=(0)                  # which GPUs, e.g. (0 1 2 3); one id per process

EPOCHS=(88)                  # any; 16 24 32 40 56 88 reproduce rn50_configs/ exactly
METHODS=(condsam)            # sam asam condsam friendlysam fishersam mongesam no_sam
RHOS=(0.01 0.05 0.1 0.5 1)   # perturbation radius (not used by no_sam)

# Only used by the matching method:
FRIENDLYSAM_SIGMAS=(1.0)
FRIENDLYSAM_LAMBDAS=(0.9)
FISHERSAM_ETAS=(1.0)
CONDSAM_BETAS=(0.99)
CONDSAM_POWER_ITERS=(5)
CONDSAM_STATS_EVERY=(1)      # refresh factors every N steps; >1 is faster, 1 is exact

SEEDS=(42)
```

Every combination runs in turn, methods innermost: the example runs 5 CondSAM jobs, one per rho (beta 0.99, 5 power
iterations); with several methods, e.g. `METHODS=(sam condsam)`, every method runs at each rho before the next.
`no_sam` runs once, first. `GPU_IDS` must list
`GPUS_PER_NODE` ids (DDP is single-node). Fixed settings (per-GPU batch 512, data paths,
`WANDB_PROJECT=ffcv-imagenet`) are in the `FIXED` block below; loader workers default to all CPUs but two,
split across the GPUs (`NUM_WORKERS=auto`). The learning rate follows the original 8-GPU recipe
(1.7 at a global batch of 4096) scaled linearly with the global batch: 0.2125 on one GPU.

```bash
DRY_RUN=1 bash scripts/train_rn50.sh     # print the commands
bash scripts/train_rn50.sh               # run
```

Arguments after the script name are passed to every run, e.g. `bash scripts/train_rn50.sh --wandb.mode=disabled`.
On one H100 at 160 px with `1.0` data in RAM, ResNet-50 trains at ~3.2k img/s with SAM (~7 min per epoch) and
~2.35k img/s with CondSAM (5 power iterations, exact per-step factors). CondSAM's factor sums run in a fused Triton
kernel on a second CUDA stream, and its update and perturbation as CUDA graphs. `CONDSAM_STATS_EVERY=N > 1` refreshes
the factors only every N steps, as K-FAC does; this changes the algorithm.

### Multi-GPU (DDP)

> **Note:** all training so far used one GPU. Multi-GPU runs have only been checked with CPU unit tests, not on real
> GPUs, so DDP behaviour, speed and accuracy are unverified.

With `GPUS_PER_NODE > 1`, each GPU computes the SAM perturbation from its own batch, and only the descent gradient is
averaged, as in the SAM paper (m-sharpness with `m` = per-GPU batch, the same as on one GPU). CondSAM then keeps its
Kronecker statistics per GPU. `--sam.local_perturbation=0` instead perturbs every GPU with the all-reduced gradient
(`m` = global batch). DDP is single-node only.

```bash
python -m unittest tests.test_ddp_sam tests.test_condsam    # CPU checks, ~1 min
CONDSAM_GPU_TESTS=1 python -m unittest tests.test_condsam   # also the CUDA paths
```

### Outputs

Each run writes `logs/<uuid>/` with `params.json`, a JSON-lines `log`, and `final_weights.pt`. W&B runs of one sweep
share a group, are named like `rn50-condsam-rho0.01-s42`, and log:

- `train/loss`, `train/lr` every 50 steps;
- `val/top_1`, `val/top_5`, `val/loss` and `epoch/*` (train loss, time, throughput, resolution) after every epoch;
- `final/*` after training.

Evaluation runs after every epoch on the full 50k validation set at 256 px with flip test-time augmentation.

## Baseline without SAM

Set `METHODS=(no_sam)` in [scripts/train_rn50.sh](scripts/train_rn50.sh); `no_sam` runs once per epoch count and
seed and ignores `RHOS` and the method-specific values. It uses the original ffcv-imagenet recipe in bf16, the same
precision as the SAM runs. For the original fp16, pass the flag on the command line:

```bash
bash scripts/train_rn50.sh                               # bf16, comparable to the SAM runs
bash scripts/train_rn50.sh --training.precision=fp16     # original precision
```

Extra flags apply to every run in the sweep, and SAM policies require bf16, so use `--training.precision=fp16` only
when `METHODS=(no_sam)`. To include the baseline in a SAM sweep instead, add it to the list, e.g.
`METHODS=(no_sam sam condsam)`.

## Additional run, 5/11/2026

After running the settings above, run [scripts/run_baseline.sh](scripts/run_baseline.sh) in one go: it runs every
SAM baseline this repo implements except CondSAM (`sam`, `asam`, `friendlysam`, `fishersam`, `mongesam`), back to
back, each at one fixed rho (and method-specific setting):

```bash
scripts/run_baseline.sh                 # run all 5 baselines
DRY_RUN=1 scripts/run_baseline.sh       # print the commands only
```

## OOD evaluation

After training, [scripts/eval_ood.sh](scripts/eval_ood.sh) evaluates every finished run (`logs/*/final_weights.pt`) on
the test sets of WiSE-FT: ImageNet (the run's validation `.ffcv` file), ImageNetV2, ImageNet-R, ImageNet-A and
ImageNet-Sketch. ImageNet-R and ImageNet-A are restricted to their 200 classes.

```bash
scripts/eval_ood.sh                     # every finished run
scripts/eval_ood.sh logs/<uuid> ...     # chosen runs
```

Set `GPU_ID` and `DATA_ROOT` at the top of the script. `DATA_ROOT` must contain `imagenetv2-matched-frequency-format-val`
(10,000 images), `imagenet-r` (30,000), `imagenet-a` (7,500) and `sketch` (50,889). The script reports top-1 and top-5
and writes `logs/<uuid>/ood_metrics.json`. All sets use the run's validation setting (center crop 224/256, 256 px, flip
test-time augmentation). The OOD images are read from their original files, while ImageNet uses the re-encoded `.ffcv`
file.

## Additional run, 7/11/2026

Run [scripts/run_condsam1.sh](scripts/run_condsam1.sh): a CondSAM1 sweep at rho 0.05, 0.1 and 0.5 (3 runs, beta 0.99,
5 power iterations, otherwise the same settings as `scripts/train_rn50.sh`). CondSAM1 (`condsam1`) keeps CondSAM's
direction inside each layer but rescales each layer to its gradient norm, so rho is split across layers as in SAM and
the total perturbation norm equals rho. The original `condsam` is unchanged.

```bash
scripts/run_condsam1.sh                 # run the 3 CondSAM1 jobs
DRY_RUN=1 scripts/run_condsam1.sh       # print the commands only
```

## Additional run, 11/11/2026

Run [scripts/run_condsam3.sh](scripts/run_condsam3.sh): it tests a CondSAM variant (`condsam3`, rho 0.05, 0.1, 0.15,
0.5; 88 epochs, otherwise the same as `scripts/train_rn50.sh`). Branch:
https://github.com/ngovuminh2005/SAM/tree/ffcv-imagenet

```bash
scripts/run_condsam3.sh                 # 4 CondSAM3 jobs
DRY_RUN=1 scripts/run_condsam3.sh       # print the commands only
```

## Additional run, 12/11/2026

Run [scripts/run_condsam.sh](scripts/run_condsam.sh): it reruns the original `condsam` (rho 0.1, 0.15, 0.5, 1) with a
different LR schedule ([rn50_configs/rn50_88_epochs_condsam.yaml](rn50_configs/rn50_88_epochs_condsam.yaml)),
otherwise the same settings as `scripts/run_condsam3.sh`. Branch:
https://github.com/ngovuminh2005/SAM/tree/ffcv-imagenet

```bash
scripts/run_condsam.sh                  # 4 CondSAM jobs
DRY_RUN=1 scripts/run_condsam.sh        # print the commands only
```

The script now defaults to the 13/11/2026 run below; to repeat this one, set `LR_SCHEDULE=cosine` and
`RHOS=(0.1 0.15 0.5 1)` at the top of it.

## Additional run, 13/11/2026

Run [scripts/run_condsam.sh](scripts/run_condsam.sh) again: the original `condsam` at rho 0.05, 0.1 and 0.15, each
with two variants of the LR schedule over the last epochs (6 runs), otherwise the same settings as the 12/11/2026 run.

```bash
scripts/run_condsam.sh                  # 6 CondSAM jobs
DRY_RUN=1 scripts/run_condsam.sh        # print the commands only
```

## Additional run, 14/11/2026

ResNet-18, 88 epochs (`rn18_configs/rn18_88_epochs.yaml`: batch 1024, lr 0.5 on one GPU), logged to the W&B project
`rn18-ffcv-imagenet`. Run both scripts:
[scripts/run_baseline_rn18.sh](scripts/run_baseline_rn18.sh) runs no SAM and the SAM baselines (`sam`, `asam`,
`friendlysam`, `fishersam`, `mongesam`) with the standard linear LR decay, and
[scripts/run_condsam_rn18.sh](scripts/run_condsam_rn18.sh) runs the original `condsam` at rho 0.05, 0.1, 0.15 and 0.5
with a cosine LR decay ([rn18_configs/rn18_88_epochs_condsam.yaml](rn18_configs/rn18_88_epochs_condsam.yaml)).

```bash
scripts/run_baseline_rn18.sh            # 6 jobs
scripts/run_condsam_rn18.sh             # 4 CondSAM jobs
DRY_RUN=1 scripts/run_condsam_rn18.sh   # print the commands only (same for run_baseline_rn18.sh)
```
