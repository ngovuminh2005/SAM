# Data and runs

The project environment is `/media/volume/tucnv/SDPO_sigmoid/wise-ft/env`.
From the repository root, update that exact prefix, not another environment:

```bash
conda env update --prefix ./env --file environment.yml
conda activate /media/volume/tucnv/SDPO_sigmoid/wise-ft/env
python -m pip check
```

`environment.yml` uses Python 3.10 and installs `requirements.txt`, including
matched PyTorch 2.6.0/torchvision 0.21.0, W&B 0.27.2, click, PyYAML, CLIP
tokenizer dependencies, dataset loaders and plotting packages. It replaces the
old Python 3.6/CUDA 11 build-specific export and its unrelated packages. The
W&B version is the one used for the project's offline integration checks.

To repair Python packages in the existing prefix without invoking Conda:

```bash
./env/bin/python -m pip install -r requirements.txt
./env/bin/python -m pip check
```

Use normal dependency resolution for these files. `--no-deps` skips W&B's
required packages such as click, pydantic and requests; the top-level
requirements file is not a complete transitive lockfile.

## Datasets

Both download scripts use this repository's `datasets/` by default. `DATA_ROOT`
or `--data-root` on the test script can choose another destination.

```bash
bash prepare_imagenet_wiseft.sh
bash prepare_imagenet_tests.sh
```

Run downloads in the `wise-ft` tmux session. The existing `datasets` window
contains the test download/extraction job; output is also in
`prepare_imagenet_tests.log`. Attach with `tmux attach -t wise-ft`.

The test script supports resume and skips already prepared datasets. It checks
the expected numbers of images/classes before publishing each destination;
V2 and Sketch additionally use pinned Hugging Face revisions and SHA256 hashes.
The source and checksum manifest is `datasets/.imagenet-tests-manifest.json`.

| Evaluation dataset | Folder | Images |
| --- | --- | ---: |
| ImageNet | `imagenet/val` | 50,000 |
| ImageNetV2 | `imagenetv2-matched-frequency-format-val` | 10,000 |
| ImageNetR | `imagenet-r` | 30,000 |
| ImageNetA | `imagenet-a` | 7,500 |
| ImageNetSketch | `sketch` | 50,889 |

## Training and W&B

Use module entry points from the repository root:

```bash
python -m src.wise_ft \
  --train-dataset ImageNet --model ViT-B/32 \
  --template openai_imagenet_template \
  --optimizer adamw --sam sam --lr 3e-5 --epochs 10 \
  --batch-size 512 --save models/sam-adamw \
  --exp_name sam-adamw --wandb-project wise-ft
```

Data defaults to this repository's `datasets/`. `--sam` and the base optimizer
remain independent; see [optimizer/README.md](optimizer/README.md) for INI
defaults and parameter selection.

W&B records effective arguments, training loss, learning rate, rho, batch time,
epoch metrics, alpha curves and a final results table. One W&B run covers both
training and all final evaluations. It finishes successfully only after the
evaluation suite completes; exceptions finish it with error status.

Without `WANDB_API_KEY`, the default mode is offline so runs do not need an
interactive login. Set `--wandb-mode online` after `wandb login` (or provide
`WANDB_API_KEY`) to upload live. `WANDB_MODE` can set the default mode.
`--wandb-mode disabled` disables tracking. Offline logs live below the run's
`wandb/` directory and can later be uploaded with `wandb sync`.
`--wandb-entity`, `--wandb-group`, `--wandb-dir` and `--log-every` customize logging.

The current model is OpenAI CLIP `ViT-B/32` at 224-pixel resolution, matching
the upstream README example. The loader also accepts official `ViT-L/14`
(224 pixels) and `ViT-L/14@336px` (336 pixels) checkpoints. The latter is the
paper's default model. Weights are downloaded on first use and checked against
the SHA256 embedded in OpenAI's model URL; model dimensions and preprocessing
resolution are inferred from the checkpoint.

For a sweep, edit `parameters.model.value` in its YAML. For a single run, use
`MODEL='ViT-L/14@336px' bash scripts/run_sam.sh --sam condsam`. Sweep YAML flags
take precedence over the script's `MODEL` environment variable. Batch 512 is
the B/32 README recipe; choose a batch fitting your hardware when changing models.

## Automatic final evaluation

Both `python -m src.wise_ft` and standalone `python -m src.models.finetune`
automatically evaluate at the end of training. The standalone command uses its
`--load` checkpoint as the reference; WiSE-FT uses the zero-shot checkpoint.

Defaults are all five datasets above and 11 alphas from 0 to 1 in steps of 0.1:
55 dataset/alpha combinations, including the reference (`alpha=0`) and final
fine-tuned weights (`alpha=1`). `--eval-datasets` and `--alpha` customize the
suite; both endpoints are always included. This evaluates the current trained
run, rather than starting training runs for other SAM variants.

The suite writes `evaluation_summary.json` below `--save`. Partial reports have
`complete: false`; only a fully successful sweep has `complete: true`. Optional
`--results-db results.jsonl` also records each evaluated alpha as a JSONL row.
Weights are restored to the fine-tuned state after evaluation, even on failure.
Interpolated model files are written only with `--save-interpolated`.

To evaluate existing checkpoints without training:

```bash
python -m src.wise_ft --load models/zeroshot.pt,models/finetuned.pt \
  --save results/existing --wandb-mode offline
```

## SAM example scripts and sweep

Run from the repository root using a Python environment with the requirements
installed. Scripts use `python` by default; set `PYTHON` to an interpreter path
to choose an environment explicitly. For example, the existing `../dft_env`
has the runtime dependencies used for this project's integration checks.

SAM with the bundled defaults (`optimizer/defaults.ini`, `samRho = 0.01`):

```bash
SAVE_DIR=models/sam-defaults bash scripts/run_sam.sh
```

SAM with its standalone INI:

```bash
SAM_CONFIG=configs/sam.ini SAVE_DIR=models/sam-config bash scripts/run_sam.sh
```

The example scripts follow the original README's end-to-end AdamW recipe: learning rate
`3e-5`, weight decay `0.1`, betas `(0.9, 0.999)` and epsilon `1e-8`. They use ViT-B/32,
batch size `512`, 10 epochs, and the default 11-alpha/five-dataset final
suite. `SAVE_DIR`, `RUN_NAME`, `MODEL`, `DATA_ROOT`, `LR`, `WEIGHT_DECAY`,
`BATCH_SIZE`, `EPOCHS`, `NUM_WORKERS` and `BASE_OPTIMIZER` override shell defaults.
Extra CLI flags are appended last, so they override corresponding arguments.
Use `SAVE_DIR` to keep checkpoint and JSONL output in the same directory.
Without `SAVE_DIR`, a timestamp/PID-based directory prevents run collisions.

W&B grid sweep (`configs/sam_sweep.yaml`): the current 4 SAM radii
`[0.05, 0.01, 0.02, 0.005]` give 4 training runs. AdamW and the optimizer settings above remain fixed.
Learning rate and batch size now follow the fine-tuning recipe rather than the
upstream parser defaults (`0.001` and `128`). The learning rate scheduler is
unchanged from upstream: 500 linear warm-up steps followed by cosine decay,
updated once per optimizer step, including when SAM uses two backward passes.
Each run automatically evaluates OOD and its alpha sweep.
The sweep objective is `epoch_eval/ImageNet/top1` (in-distribution accuracy).
Sweep jobs use online mode and need W&B login/API credentials.

```bash
wandb login
bash scripts/sweep_sam.sh --project wise-ft
# Resume an existing sweep, with at most three runs on this agent:
bash scripts/sweep_sam.sh --sweep-id ENTITY/PROJECT/SWEEP_ID --count 3
```

`--entity`, `--project`, `--config` and `--count` configure the agent. Its Python
interpreter is passed through to the training subprocesses. Start the script
in the `wise-ft` tmux session for long jobs.

`--count N` limits this agent to at most N runs; it does not define the grid.
It is optional. With no count, the agent has no local run limit and a finite
grid finishes when W&B has no remaining combinations. Multiple agents share
the same sweep, so an individual agent may execute fewer runs.

Each supported baseline has its own sweep file. Rho and method-specific
parameters vary; AdamW, learning rate, weight decay, model, epochs, batch size
and evaluation remain fixed:

| Config | Perturbation | Grid |
| --- | --- | --- |
| `configs/sam_sweep.yaml` | SAM | 4 radii, fixed AdamW |
| `configs/asam_sweep.yaml` | ASAM | 3 radii, fixed AdamW |
| `configs/condsam_sweep.yaml` | CondSAM | 4 radii × beta 0.99 × 5 power iterations = 4 |
| `configs/friendlysam_sweep.yaml` | FriendlySAM | 3 radii × 3 sigma × 2 lambda = 18 |
| `configs/fishersam_sweep.yaml` | FisherSAM | 3 radii × 3 eta = 9 |
| `configs/mongesam_sweep.yaml` | MongeSAM | 3 radii, fixed AdamW |
| `configs/no_sam_sweep.yaml` | No perturbation | One fixed AdamW run; no rho parameter |

```bash
bash scripts/sweep_condsam.sh
bash scripts/sweep_no_sam.sh
```

Each baseline has a matching `scripts/sweep_<baseline>.sh` with its YAML path
already specified: `sam`, `asam`, `condsam`, `friendlysam`, `fishersam`,
`mongesam`, and `no_sam`. The default W&B project is `wise-ft`, declared in
`scripts/sweep_sam.sh`; change `WANDB_PROJECT` there or set the environment
variable. `--project` can override it for a single invocation.

ASAM, CondSAM, FriendlySAM, FisherSAM and MongeSAM read their method-specific
defaults from `optimizer/defaults.ini`. The sampled `--sam-rho` overrides the
corresponding INI radius for each run. Removed methods have no sweep configs.

The current CondSAM grid is `rho=[0.05, 0.5, 1.5, 1]`, `beta=[0.99]`,
`power_iters=[5]`; FriendlySAM uses `sigma=[0.5, 1.0, 2.0]`,
`lmbda=[0.9, 0.95]`; FisherSAM `eta=[0.1, 1.0, 10.0]`.
CondSAM's numerical `eps=1e-8` stays at its INI default.

W&B run names include effective values after CLI/INI resolution, for example
`sam-adamw-rho0p5-lr0p00003`. Epoch count, weight decay and batch size are omitted
from names but remain in W&B config. Decimal points become `p`,
including values originally written in scientific notation. CondSAM names
contain the three method settings `rho`, `beta`, `poweriters`; numerical `eps`
remains in config. FriendlySAM and FisherSAM also include their method settings. A user
`--exp_name`/script `RUN_NAME` is appended as a suffix, preserving the script's
timestamp/PID for uniqueness.

Evaluate the trained checkpoints again on ID + all four OOD datasets and the
default 11 alphas:

```bash
SAVE_DIR=results/sam-ood bash scripts/eval_ood.sh \
  models/sam-defaults/zeroshot.pt \
  models/sam-defaults/finetuned/checkpoint_10.pt
```

Preview commands and sweep configuration without training or contacting W&B:

```bash
DRY_RUN=1 bash scripts/run_sam.sh
bash scripts/sweep_sam.sh --dry-run
DRY_RUN=1 bash scripts/eval_ood.sh reference.pt finetuned.pt
```
