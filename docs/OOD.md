# OOD evaluation

This is an extension to the supplied PDF. It implements post-training evaluation
without adaptation or access to OOD data during training. It supports image
checkpoints trained on CIFAR-10 or CIFAR-100 with any existing vision optimizer.

## Training evaluation schedule

All training INI files now default to `evalEvery = 20`. Vision and GLUE evaluate
at epochs 20, 40, ... and at the final epoch even when it is not divisible by 20.
Thus a 200-epoch run has 10 scheduled evaluations; a 3-epoch GLUE run evaluates
once at epoch 3. Training metrics remain logged every epoch. Checkpoint saving
is independent of this schedule. MSAM's final BatchNorm recalibration is followed
by the final evaluation, without an earlier duplicate final evaluation.

`EVAL_EVERY=20` is editable directly in the two CondSAM/paper sweep scripts.
Direct Python usage supports `--evalEvery 20`. Legacy checkpoint parameters
without the key use 20 on resume. New HDF5 logs contain explicit `train/epoch`
and `test/epoch` arrays so sparse test records are not treated as consecutive epochs.

## Task architecture

- `ood/base.py`: `AbstractOODTask`, the `OODCase` contract, deterministic case
  iteration, shared DataLoader/subset handling and model-mode restoration.
- `ood/settings.py`: typed `OODSetting`, JSON config validation and defaults.
- `ood/tasks.py`: `CorruptionTask` and `DetectionTask` implementations.
- `ood/data.py`: prepared CIFAR, torchvision SVHN test split, memory-mapped
  corruption arrays, and deterministic preprocessing from the training checkpoint.
- `ood/metrics.py`: MSP/energy scores and detection metrics with explicit orientation.
- `ood/checkpoint.py`: strict architecture/class-count restoration from `params.json`
  and `modelState`; handles DDP and torch.compile prefixes.
- `scripts/evaluate_ood.py`: common CLI; JSON, CSV, incremental results and W&B.

Extend `AbstractOODTask` by supplying `cases()` and `evaluate_case()`, optionally
`aggregate()`. Register the class in `ood.create_task`; extend setting/data
validation for any new dataset. Task instances receive the evaluation transform.

## Task settings and launchers

Every task has one JSON under `configs/ood/` and one shell script under `scripts/ood/`:

| Setting / script stem | ID training dataset | Evaluation |
|---|---|---|
| `cifar10_c` | CIFAR-10 | CIFAR-10-C |
| `cifar100_c` | CIFAR-100 | CIFAR-100-C |
| `cifar10_vs_cifar100` | CIFAR-10 | CIFAR-100 detection |
| `cifar100_vs_cifar10` | CIFAR-100 | CIFAR-10 detection |
| `cifar10_vs_svhn` | CIFAR-10 | SVHN detection |
| `cifar100_vs_svhn` | CIFAR-100 | SVHN detection |

Edit `RUN_DIRS`, `CHECKPOINT`, GPU/device, data root, W&B mode and sample limit
directly in each `run_<stem>.sh`. Each RUN_DIR must contain a matching trained
`params.json` and `final.model`; mismatched ID datasets or model weights fail.
Alternatively pass trained run directories as positional arguments:

```bash
bash scripts/ood/run_cifar10_vs_cifar100.sh logs/YOUR_CIFAR10_TRAINED_RUN
bash scripts/ood/run_cifar10_c.sh logs/YOUR_CIFAR10_TRAINED_RUN
```

These launchers evaluate existing checkpoints. They do not launch training or
silently choose a checkpoint. Multiple run directories evaluate the same task
on multiple models/optimizers. The OOD runner supports CPU or one GPU and evaluates
all selected examples; it does not use DistributedSampler/padded test samples.

## Data

Clean CIFAR-10/100 are already prepared in the project's NPZ format. The added
OOD datasets must be downloaded before their tasks can run:

```bash
bash scripts/ood/prepare_datasets.sh
```

Edit `DATASETS` in this script to choose only the desired downloads. It downloads
only the SVHN **test** split and official CIFAR-C archives (each about 2.9 GB),
verifies archive MD5, and extracts them under `datasets/`. No downloads are
triggered automatically by OOD evaluation itself.

Expected paths:

```text
datasets/cifar10/{test.npz,metadata.json}
datasets/cifar100/{test.npz,metadata.json}
datasets/CIFAR-10-C/{labels.npy,gaussian_noise.npy,...}
datasets/CIFAR-100-C/{labels.npy,gaussian_noise.npy,...}
datasets/SVHN/test_32x32.mat
```

Sources: [official corruption repository](https://github.com/hendrycks/robustness),
[CIFAR-10-C](https://zenodo.org/records/2535967),
[CIFAR-100-C](https://zenodo.org/records/3555552),
[SVHN](http://ufldl.stanford.edu/housenumbers/).
The CIFAR-C loader slices 10,000 images per severity from each 50,000-image NPY.
It supports labels stored once (10,000) or repeated (50,000), and memory-maps images.

## Metrics and protocol

Corruption settings default to the standard 15 corruption types × severities 1–5,
plus a clean-ID reference case: 76 cases per checkpoint. They report per-case
accuracy/error/cross-entropy, per-severity/per-corruption accuracy, mean accuracy,
mean raw error and clean-to-corrupt accuracy drop. Raw mean error is not
AlexNet-normalized mCE. No random crop, flip, Cutout or RandAugment is applied.
All input datasets use the ID checkpoint's resize and normalization.

Detection uses full ID test data against the full selected OOD test data, with
MSP (`1 - max softmax`) and energy (`-T * logsumexp(logits/T)`, default T=1).
Both scores increase toward OOD. Outputs are AUROC, AUPR-OUT, AUPR-IN and FPR95.
For FPR95, ID is the positive class: report the fraction of OOD samples falsely
accepted at the first attainable threshold retaining at least 95% of ID samples.
AUPR-OUT uses OOD positives; AUPR-IN uses ID positives. Natural split sizes are
retained, so AUPR depends on the ID/OOD prevalence. Metrics are fractions [0,1].
No threshold or temperature is tuned using the OOD test data.
Score reference: [official Energy OOD implementation](https://github.com/weitliu/energy_ood).

`MAX_EXAMPLES=0` means full evaluation. Any positive value selects the first N
examples per case/split for diagnostics, and marks the report `diagnostic_subset`;
these subset numbers must not be presented as full benchmark results.

## Outputs

Each run writes `results.json`, `results.csv`, and `partial_results.json` below
`<training_run>/ood/<task>/<timestamp>/`. Reports include checkpoint provenance,
settings, counts, device, metric definitions and whether a diagnostic subset was used.
W&B project `CondSAM` receives per-case metrics and aggregate summaries in a separate
`job_type=ood` run linked through the source checkpoint/config. Existing training
run histories are not overwritten.

Tests cover schedule/final epoch, severity slicing, label layout, score orientation,
abstract lifecycle, ignored OOD labels, strict checkpoint loading, and an end-to-end
CPU CLI run with a synthetic diagnostic dataset and real ResNet checkpoint.
Full corruption/detection benchmarks require the actual OOD datasets and trained models.
