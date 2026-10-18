# OrthoSAM: Orthogonal Perturbations for Sharpness-Aware Minimization


## Requirements

The main training entrypoint uses CUDA/NCCL.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

## Data

Prepare CIFAR-10, CIFAR-100, and Tiny-ImageNet under the project `datasets/`
directory using the Hugging Face mirrors. Install the project requirements first,
then run:

```bash
python scripts/prepare_datasets.py --dataset all
```

The command downloads and prepares CIFAR-10 and CIFAR-100 as `train.npz` and
`test.npz`, and exports Tiny-ImageNet as `datasets/tiny-imagenet/{train,val}`.
To prepare only one dataset, use `--dataset cifar10`, `cifar100`, or
`tiny-imagenet`. Prepared data is local and is not included in this repository.

Training scripts set `SAM_DATASET_ROOT` to the project `datasets/` directory.
For data stored elsewhere, edit `DATASET_ROOT` in the script or export
`SAM_DATASET_ROOT=/path/to/datasets`. ImageNet must be supplied separately in
ImageFolder layout; point the config's `dataDir` or `SAM_DATASET_ROOT` at a
directory containing `train/` and `val/`.

## W&B setup

The CondSAM sweep logs to the W&B project `CondSAM` in online mode. Create an
API key in your W&B account and export it before starting a run:

```bash
export WANDB_API_KEY="your_wandb_api_key"
```

Keep the key in your shell environment; do not put it in this repository. If no
login is available, tracking falls back to offline mode and stores data in the
run directory.

## CondSAM rho sweep

The script defaults to `configs/CIFAR10_PyramidNet110.ini`. To run with another
config and prefix:

```bash
scripts/run_condsam_rho.sh configs/CIFAR10_ResNet50.ini cifar10_resnet50
```

You can also edit the values near the top of
[`scripts/run_condsam_rho.sh`](scripts/run_condsam_rho.sh):

- `CONFIG_PATH`: config file passed to `main.py`.
- `RHO_LIST`: CondSAM radii to sweep.
- `CONDSAM_BETA`, `CONDSAM_EPS`, `CONDSAM_POWER_ITERS`: CondSAM factor decay,
  numerical stability epsilon, and power iteration count.
- `EPOCHS`, `EVAL_EVERY`, `REPEATS`: training length, evaluation interval, and
  number of runs per radius.
- `GPU_ID`, `NPROC_PER_NODE`: visible GPU and distributed worker count.
- `DATASET_ROOT`, `PYTHON`: dataset directory and Python executable.
- `WANDB_PROJECT`, `WANDB_MODE`: W&B project and online/offline tracking mode.

The first two optional command-line arguments override `CONFIG_PATH` and
`LOG_PREFIX`; further arguments are forwarded to `main.py`. The run name
includes the config filename, radius, and repeat number. The script defaults to
one repeat and six radii.

Training output is written under `logs/<run_name>/`: `params.json`, `data.hdf5`,
`terminal.log`, and `training_time.txt`. A final model checkpoint is saved as
`final.model`; W&B run data is stored inside the run directory when offline.
The script uses `.venv/bin/python` by default; edit `PYTHON` if your environment
uses a different location.

### CondSAM parameters

`--optimizer CondSAM` enables CondSAM. Its main parameters are:

| Argument | Purpose | Default |
| --- | --- | --- |
| `--condsamRho` | Perturbation radius; `-1` uses the config's `rho` | `-1` |
| `--condsamBeta` | Exponential moving average decay for activation and gradient factors | `0.95` |
| `--condsamEps` | Numerical stability constant | `1e-8` |
| `--condsamPowerIters` | Power iterations per factor; must be positive | `1` |

The rho sweep sets `--condsamRho` from `RHO_LIST` and passes the other three
parameters from `CONDSAM_BETA`, `CONDSAM_EPS`, and `CONDSAM_POWER_ITERS` in the
script. Command-line values take precedence over values in the config.

## Run

Paper image-classification settings: 200 epochs; SGD-style momentum `0.9`;
cosine LR after warmup; CIFAR uses random crop, horizontal flip, Cutout
probability `0.5`, and label smoothing `0.1`; Tiny-ImageNet uses RandAugment
magnitude `10` and label smoothing `0.1`.

The paper method OrthoSAM is `--optimizer OrthoSAM`. The paper reports
mean/variance over three independent runs; add `--rndSeed True` when repeating
runs for statistics. Commands below pass `--epochs 200` explicitly, so they
override any local debug value in the config file.

### CIFAR-10 / ResNet-50

Config: [configs/CIFAR10_ResNet50.ini](configs/CIFAR10_ResNet50.ini)

Setting: `dataset=CIFAR10`, `model=resnet`, `depth=50`, `batchSize=256`,
`learningRate=0.1`, `weightDecay=0.001`, `momentum=0.9`, `samRho=0.05`.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/CIFAR10_ResNet50.ini \
  --optimizer OrthoSAM \
  --samRho 0.05 \
  --epochs 200
```

Run all paper image baselines plus OrthoSAM on this config:

```bash
GPU_ID=0 bash scripts/run_sam_variants.sh \
  configs/CIFAR10_ResNet50.ini \
  cifar10_resnet50 \
  --epochs 200
```

### CIFAR-100 / ResNet-50

Config: [configs/CIFAR100_ResNet50.ini](configs/CIFAR100_ResNet50.ini)

Setting: `dataset=CIFAR100`, `model=resnet`, `depth=50`, `batchSize=256`,
`learningRate=0.1`, `weightDecay=0.001`, `momentum=0.9`, `samRho=0.1`.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/CIFAR100_ResNet50.ini \
  --optimizer OrthoSAM \
  --samRho 0.1 \
  --epochs 200
```

Run all paper image baselines plus OrthoSAM on this config:

```bash
GPU_ID=0 bash scripts/run_sam_variants.sh \
  configs/CIFAR100_ResNet50.ini \
  cifar100_resnet50 \
  --epochs 200
```

### Tiny-ImageNet / ResNet-50

Tiny-ImageNet was prepared with the command above. To prepare only this dataset:

```bash
python scripts/prepare_datasets.py --dataset tiny-imagenet
```

Config: [configs/TinyImageNet_ResNet50.ini](configs/TinyImageNet_ResNet50.ini)

Setting: `dataset=TinyImageNet`, `model=resnet`, `depth=50`, `imageSize=64`,
`batchSize=125`, `learningRate=0.05`, `weightDecay=0.001`, `momentum=0.9`,
`randAugment=True`, `randAugment_magnitude=10`, `samRho=0.05`.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/TinyImageNet_ResNet50.ini \
  --optimizer OrthoSAM \
  --samRho 0.05 \
  --epochs 200
```

Run all paper image baselines plus OrthoSAM on this config:

```bash
GPU_ID=0 bash scripts/run_sam_variants.sh \
  configs/TinyImageNet_ResNet50.ini \
  tinyimagenet_resnet50 \
  --epochs 200
```

Tiny-ImageNet / ResNet-101 uses
[configs/TinyImageNet_ResNet101.ini](configs/TinyImageNet_ResNet101.ini):

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/TinyImageNet_ResNet101.ini \
  --optimizer OrthoSAM \
  --samRho 0.05 \
  --epochs 200
```

Logs are written under `logs/`. Relative `logDir = logs` entries in
[configs](configs) are resolved from the repository root.

## Other Presets

Config files are in [configs](configs). Supported datasets are `CIFAR10`,
`CIFAR100`, `TinyImageNet`

Supported model names are `WRN`, `resnet`, `resnetCIFAR`, `PyramidNet`,
`DenseNet`, and `ViT`.

## CondSAM

`CondSAM` uses empirical activation/backpropagation second moments to construct
perturbations, then applies the same SGD/momentum/weight-decay/Nesterov update as
SAM. It supports vision and GLUE classification/regression. It runs in eager mode.
Vision CNNs use `channels_last`. Conv patches are materialized from strided
views over image chunks, avoiding per-image CUDA `im2col` launches; full factors
still update every step. See [timing measurements](docs/CONDSAM_PERFORMANCE.md).
The radius sweep refuses to start on a busy GPU and holds a lock against duplicate
sweeps; edit `ALLOW_SHARED_GPU` only for intentional sharing.
Training shell scripts set `SAM_CONSOLE_EVAL_ONLY=1`: epoch metrics/timing print
only at evaluation epochs (including the final epoch), while W&B and HDF5 still
record every training epoch. Startup messages and errors remain visible.
Training sweeps handle Ctrl+C, SIGTERM and terminal hangup by terminating the
active torchrun launcher and exiting the sweep, rather than moving to the next
radius after an interruption. This handling applies to newly started scripts.

For a loss averaged over a local batch of `N` examples, multiply output-gradient
hooks by `N` before collecting statistics. Flatten Linear batch/token dimensions
into rows; extract Conv2d patches with the layer's stride, padding and dilation.
With activation rows `X` and rescaled derivative rows `D`, compute in FP32:

$$\widehat A=X^\top X/M,\qquad \widehat B=D^\top D/M.$$

Here `M=N` for ordinary Linear, `M=N*T` for token Linear (all supplied positions),
and `M=N*H_out*W_out` for Conv2d. These are uncentered second moments, not outer
products of batch-averaged vectors. Token count does not replace `N` when undoing
sample-mean loss reduction. With training BatchNorm, derivatives include coupling
between examples and should not be interpreted as independent per-example scores.
DDP sums outer products and row counts across ranks before dividing.

Initialize factors to identity, unit vectors randomly (synchronized across ranks),
and update once per step, using only the first forward/backward:

$$A_t=\beta A_{t-1}+(1-\beta)\widehat A_t,\qquad
B_t=\beta B_{t-1}+(1-\beta)\widehat B_t.$$

Run `condsamPowerIters` warm-started power iterations per factor (default: one)
to obtain unit vectors `x,y` and Rayleigh
estimates $\widehat\sigma_A=x^\top Ax$, $\widehat\sigma_B=y^\top By$; these are
not exact largest eigenvalues. For each supported weight matrix gradient `G`, let
$p=\operatorname{numel}(G)$, $s=y^\top Gx$ and
$\bar s=s+\epsilon\operatorname{sign}_+(s)$, with $\operatorname{sign}_+(0)=1$:

$$E=\frac{(\widehat\sigma_A\widehat\sigma_B)^2}{2\bar s}yx^\top
-\frac{\|A\|_F^2\|B\|_F^2}{2p(\|G\|_F^2+\epsilon)}G.$$

Use `E` for Linear (including output heads) and Conv2d with `groups=1`.
Biases, normalization parameters, embeddings and grouped convolutions use their
ordinary gradient; zero `E` also falls back to the gradient. Concatenate all
these directions conceptually and normalize **once over the entire model**:

$$R=\sqrt{\sum_q\|d_q\|_F^2},\qquad \Delta\theta_q=\rho d_q/R.$$

When `R=0`, perturbations are zero. Non-finite statistics/directions raise errors.
The second pass uses the same batch, with factor collection and BatchNorm running
statistics disabled. Restore original weights before clipping and the SGD update.
Hooks and perturbations are cleaned up even on exceptions. Factors, power vectors
and momentum are included in optimizer checkpoints.

This is a pooled Kronecker-statistics method, **not the full K-FAC optimizer**:
it uses actual training targets/loss, no inverse Fisher update, no convolutional
cross-position covariance and no extra spatial-count multiplier. Full factors cost
`O(d_in^2 + d_out^2)` memory per layer; chunked accumulation reduces temporary
memory but does not remove this cost. Neither decreased condition number nor a
loss-ascent direction is guaranteed by the proposed formula.

Defaults: `condsamRho=-1` (use `rho`), `condsamBeta=0.95`, `condsamEps=1e-8`,
`condsamPowerIters=1` (positive integer). Set `CONDSAM_POWER_ITERS` in either
CondSAM/paper sweep script, or pass it as an environment variable when launching
the shell script. Direct Python launches accept `--condsamPowerIters`.
The iteration count is saved in optimizer parameter groups; older optimizer
checkpoints without it retain the configured count when loaded.
The rho scheduler uses the resolved CondSAM radius. GLUE applies its existing
per-task rho override to CondSAM as well. The automatic benchmark list is unchanged.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --ifile configs/CIFAR10_ResNet50.ini --optimizer CondSAM \
  --condsamRho 0.05 --condsamBeta 0.95 --condsamEps 1e-8 --condsamPowerIters 1 --epochs 200
```

Focused checks (CPU/Gloo requires local socket access):

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_condsam.py' -v
```

## W&B and paper experiment coverage

All vision optimizers, GLUE and Hessian analysis log to W&B project `CondSAM`
on rank 0. Config, actual random seed, loss/accuracy or GLUE metrics, LR/rho,
epoch timing and Hessian eigenvalues are recorded as applicable. Log in once:

```bash
wandb login
```

Without credentials the tracker saves offline runs under each run directory;
`wandb sync /path/to/offline-run-*` can upload them later.
`WANDB_MODE=disabled` disables tracking for direct Python entrypoints.
See [W&B offline documentation](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues).

Edit values directly in `scripts/run_paper_sweep.sh`, including `SECTION`
(`image`, `adam`, `glue`, `noise`, `hessian`), optimizer/radius lists, repeats,
GPU, epochs, and `WANDB_MODE`. `DRY_RUN=true` prints commands without training.
Hessian mode requires checkpoint directories in `HESSIAN_RUN_DIRS`.
Then run:

```bash
bash scripts/run_paper_sweep.sh
```

The [PDF settings audit](docs/PAPER_SETTINGS_AUDIT.md) details coverage and
remaining reproduction differences. The PDF contains label-noise robustness,
not a dedicated OOD evaluation. The sweep matches explicit training settings;
it does not establish that existing algorithms exactly reproduce the PDF.

## Evaluation interval and OOD

Training now evaluates every **20 epochs**, plus the final epoch. Edit `EVAL_EVERY`
in the sweep files or pass `--evalEvery`; train logging and checkpoint schedules
remain independent. A three-epoch GLUE run evaluates at epoch 3.

OOD has a shared abstract task class and six settings: CIFAR-10-C, CIFAR-100-C,
CIFAR-10 → CIFAR-100, CIFAR-100 → CIFAR-10, and each CIFAR dataset → SVHN.
Task JSON files are under `configs/ood/`; individual launchers are grouped under
`scripts/ood/`. Edit each launcher's `RUN_DIRS` to point to trained vision runs.

See [OOD setup, metrics and scripts](docs/OOD.md). CIFAR-C/SVHN data preparation is
provided separately in `scripts/ood/prepare_datasets.sh`; these additional datasets
are not downloaded by the evaluator. OOD is an extension beyond the supplied PDF.
