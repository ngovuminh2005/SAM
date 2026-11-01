# SAM perturbation policies

`optimizer/` contains perturbation mathematics and statistics, not descent
optimizers. Policies do not run a model, choose a loss, call backward, clip
gradients, change parameter values, or call an optimizer step. There is no
dependency on the source project's `utility.Args` or `smooth_crossentropy`.

Responsibilities:

| Component | Owns |
| --- | --- |
| `src/models/training.py` | Base optimizer construction, named parameter selection, forward/backward passes, temporary perturbation, exact restoration, BatchNorm handling, gradient clipping, descent step |
| `optimizer/*.py` | Perturbation directions, normalization and method-specific statistics |
| PyTorch/custom base optimizer | Learning rate, weight decay, update momentum/Adam moments and actual parameter updates |
| `src/models/finetune.py` | Batches, scheduler, logging, evaluation and checkpoints |

## Configuration

Existing commands retain AdamW and one forward/backward pass by default:
`--optimizer adamw --sam none`.

[`defaults.ini`](defaults.ini) is loaded automatically, independently of the
working directory. `[SAM]` contains a separate radius for each supported policy:
`samRho`, `asamRho`, `condsamRho`, `friendlysamRho`, `fishersamRho`, and
`mongesamRho`, all initially `0.01`. Each is declared exactly once, exclusively
in `[SAM]`; in particular, `[CondSAM]` has no second rho declaration.
Other defaults are FriendlySAM `sigma = 1.0` / `lmbda = 0.9`, FisherSAM
`eta = 1.0`, and CondSAM `beta = 0.95` / `eps = 1e-8` / `power_iters = 1`.
There are no settings for removed baselines.

Use `--sam-config path/to/experiment.ini` to select a different INI. Precedence
is explicit CLI flags, then the selected INI, then built-in code defaults for
omitted keys. A custom INI replaces the bundled file; it is not merged with it.
Missing files, unknown sections/keys, and invalid values are errors.
This config does not select the SAM method or the base optimizer.

Choose the descent optimizer independently of the perturbation policy:

```bash
--optimizer adamw --sam sam --sam-rho 0.05
--optimizer sgd --momentum 0.9 --sam condsam --condsam-beta 0.95
```

Supported policies: `sam`, `asam`, `condsam`, `friendlysam`, `fishersam`,
`mongesam`. Their formulas follow the imported implementations. Fine-tuning uses
the selected policy's INI radius (`0.01` by default); `--sam-rho` overrides it for a run.
Direct policy construction retains its Python constructor defaults.

`--sam-include` and `--sam-exclude` are regexes over the unwrapped model's
parameter names. The selected trainable parameters alone contribute to the
perturbation direction and its normalization. Excluded parameters still get
descent gradients and optimizer updates. An empty selection is an error.
For example, `--sam-exclude '(^|\.)bias$'` excludes biases from perturbation.
`--freeze-encoder` still selects head-only fine-tuning before these filters.

`--grad-clip-norm` defaults to `1.0`, preserving the original descent clipping.
Clipping happens after the perturbed backward pass and after weights have been
restored. The ascent gradient is not clipped. `--sam-rho 0` uses one pass.

## Policy interface

Construct `policy = SAM(selected_params, rho=...)`, or another policy, using
parameters selected by the caller. `policy.observe(model, batch_size)` is a
context manager around the first forward/backward pass. CondSAM uses hooks
there to estimate factors; the other policies need no hooks. After backward,
`policy.perturbations()` returns `{parameter: delta}`. The training loop applies
and restores these deltas, then calls the independently constructed optimizer.
`train_batch` also accepts a custom optimizer instance directly.

CondSAM keeps the imported empirical Kronecker-factor formula, EMA, power
iteration and plain-SAM fallback for unsupported weights. Factors are collected
only for selected `Linear` and ungrouped `Conv2d` weights. Functional operations
that bypass a module's forward hook, including PyTorch attention projection
weights, use the fallback unless that weight has other observed module calls.
Factor statistics are pooled across DataParallel replicas/DDP workers.
Observation assumes the caller's loss is averaged over the batch, as in WiSE-FT.

Optimizer checkpoints remain `optim_<epoch>.pt` with the original PyTorch state
format. Policies additionally save `sam_<epoch>.pt`, containing ordered parameter
names and policy state. Match those names before loading the policy state into
a new selection. As before, `--load` loads a model, not a full training resume.

## Removed code

Removed `CondSam1`, `CondSam2`, coupled `Adam*_SAM`/`Adam*_FriendlySAM`, custom
copies of SGD/AdamW, and their old factory. Compose a policy with a real base
optimizer instead. Legacy checkpoints from the other project's optimizer
classes are not compatible with these policy checkpoints.

ESAM, GSAM, LookSAM, MSAM and BayesianSAM were also removed: their imported
implementations own sample selection, gradient modification, step scheduling,
or the descent update. They are not represented as equivalent perturbation-only
policies. Reintroducing them requires an explicit training-loop protocol.

Run the focused CPU checks with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest discover -s tests -v
```
