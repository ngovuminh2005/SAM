# SAM Variants: Training and CIFAR-C Evaluation

## Environment

Training requires CUDA-compatible NVIDIA GPUs. Create and activate the Conda environment:

```bash
conda create -n condsam python=3.12 -y
conda activate condsam
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

The launcher uses the active Conda environment's Python interpreter.

## Datasets

Download clean CIFAR datasets and the corresponding corruption benchmarks:

```bash
python scripts/prepare_datasets.py --dataset cifar10
python scripts/prepare_datasets.py --dataset cifar100
python scripts/prepare_ood_datasets.py \
  --datasets cifar10_c cifar100_c --data-root datasets
```

Data is stored under `datasets/{cifar10,cifar100,CIFAR-10-C,CIFAR-100-C}/`.

## Configuration and Execution

Edit `GPU_ID` (GPU indices), `NPROC_PER_NODE` (training GPU count), and W&B credentials directly at the top of
[scripts/run_sam_variants_cifar_c.sh](scripts/run_sam_variants_cifar_c.sh):

```bash
GPU_ID="0"
NPROC_PER_NODE=1
export WANDB_API_KEY="your_wandb_api_key"
```

For two-GPU DDP training, set `GPU_ID="0,1"` and `NPROC_PER_NODE=2`.
Save the `.sh` file, then execute the launcher command below.
`NPROC_PER_NODE` must match the number of listed GPUs. The configured batch size
is global and is divided across training workers. OOD evaluation uses the first
listed GPU.

Training hyperparameters, including epochs, batch size, learning rate and
optimizer-specific perturbation radii, are defined in the corresponding
`configs/CIFAR{10,100}_<backbone>.ini` files. The default `CONFIGS` list in the
launcher includes all five backbones for each dataset:

| Backbone | Config suffix |
| --- | --- |
| ResNet-50 | `ResNet50.ini` |
| WRN-16-4 | `WRN16_4.ini` |
| PyramidNet-110 | `PyramidNet110.ini` |
| WRN-28-10 | `WRN28_10.ini` |
| DenseNet-121 | `DenseNet121.ini` |

```bash
conda activate condsam
bash scripts/run_sam_variants_cifar_c.sh
```

The launcher processes CIFAR-10 followed by CIFAR-100, with backbones in the
table order. Each backbone trains SGD, SAM, ASAM, MSAM, ESAM, FriendlySAM,
FisherSAM, MongeSAM and GSAM sequentially, once per optimizer.
Hyperparameters are fixed by each configuration; the training seed is `42 + rank`.
After each training run, `final.model` is evaluated on the corresponding CIFAR-C
benchmark in five separate severity runs (1–5), each covering 15 corruption types.
The default execution produces 90 training runs and 450 evaluation runs.
Pass config paths as positional arguments to run a subset instead of the full list.

## Tracking and Outputs

Training and evaluation log to W&B project **`variants-cifar-c`**. Training groups
are `train_cifar10` and `train_cifar100`; evaluation groups follow
`ood_<dataset>_c_sev<N>`. Evaluation logs per-corruption accuracy and
`ood/avg_accuracy`. Missing W&B credentials trigger offline storage.

Local artifacts are written to `logs/variants_cifar_c_<timestamp>_<pid>/`:

| Path | Artifacts |
| --- | --- |
| `train/<case>/<run>/` | `params.json`, `data.hdf5`, `terminal.log`, `training_time.txt`, `final.model`; intermediate checkpoints when enabled |
| `cifar_c_results/<case>__<setting>__sev<N>/` | `partial_results.json`, `results.json`, `results.csv` |
| Batch root | `cifar_c_summary.md`, `cifar_c_summary.csv` |
| Each run's `wandb/` directory | Local W&B tracking files when enabled |
