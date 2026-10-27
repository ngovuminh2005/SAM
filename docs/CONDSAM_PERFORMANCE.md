# CondSAM performance correction

The previous implementation collected Conv2d statistics one image at a time and
allocated a new dense Gram matrix for each image. ResNet-50 has 53 convolutions;
at batch 256 this meant 13,568 Python iterations per optimizer step, each launching
multiple small CUDA operations. Factor/gradient checks and norm branches also
called `.item()` repeatedly, forcing CPU/GPU synchronization. Single-rank DDP
unnecessarily ran factor collectives.

Changes in `optimizer/condsam.py`:

- Accumulate covariance over batches targeting 256 MiB of patch/derivative data
  per chunk (reshape copies can use additional memory); use larger Gram GEMMs and accumulate directly into existing sum buffers.
- Build patches with `Tensor.unfold` strided views, permute and materialize once
  per chunk. The earlier batched `F.unfold` still launched CUDA im2col once per
  image: a profile found 4,352 im2col kernels per ResNet-50 step. No patch or image
  is sampled or omitted. Padding, dilation and stride retain their definitions.
- Keep eager CNN weights/inputs in `channels_last`, avoiding repeated cuDNN layout
  conversions. This also makes many 1×1 activation/derivative row views contiguous.
- Use larger covariance GEMMs (row chunks up to 262,144) and foreach perturbation
  and restore operations. Global norms remain FP64. Ordinary scales apply in FP32
  without converting entire directions to FP64; scales outside the normal FP32
  range retain the FP64 path, including extreme tiny/large gradient tests.
- Reuse factor-sum buffers between steps. They are transient optimizer attributes,
  excluded from checkpoints and cleared on state loading.
- Batch finite checks, keep zero-direction/power-vector selection on GPU, check
  the global perturbation norm once, and skip collectives for single-rank DDP.

Full A/B factors, all batch images/positions, EMA frequency, power iteration,
perturbation formula and global rho normalization are retained. Floating-point
summation order changes; bitwise identical training trajectories are not promised.

## Measurements

### Current comparison with the training baseline

H100 80 GB, ResNet-50/CIFAR-10, batch 256, FP32, one-rank NCCL DDP/SyncBN.
SAM uses `torch.compile` as in `main.py`; CondSAM remains eager. The actual
CIFAR loader/augmentation and host-to-device transfers are included. Learning
rate is zero; no checkpoint or W&B run is created. Two other CondSAM training
jobs shared the GPU throughout these measurements and were left running.

| Implementation | SAM compiled, s/step | CondSAM, s/step | CondSAM / SAM |
|---|---:|---:|---:|
| Earlier batched F.unfold, NCHW | 0.2127 | 0.6122 | 2.88× |
| Strided patch views, NCHW | 0.2151 | 0.5260 | 2.44× |
| Strided patch views, channels_last | 0.2144 | 0.4350 | 2.03× |
| Current: larger chunks and foreach operations | 0.2116 | 0.3934 | **1.86×** |

The final comparison alternates optimizer order across two rounds, each with
five warm-up steps and twenty timed steps. Individual round means were
SAM 0.21172/0.21153 s and CondSAM 0.39807/0.38871 s. The table uses the median
of the round means. Earlier stages used three warm-up and twelve timed steps
per round. Compilation, warm-up, logging, evaluation and checkpoint saving are
excluded. This meets the requested 2× bound in this short comparison; it does not
guarantee an epoch time or a bound across other models, GPUs or workloads.

Raw final measurements: [JSON](benchmarks/condsam_resnet50_cifar10_h100.json).
Reproduce with a free GPU, or account for other workloads:

```bash
CUDA_VISIBLE_DEVICES=0 SAM_DATASET_ROOT="$PWD/datasets" OMP_NUM_THREADS=1 \
env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/benchmark_condsam.py --ifile configs/CIFAR10_ResNet50.ini \
  --steps 20 --warmup 5 --rounds 2 --output /tmp/condsam_timing.json
```

The radius sweep now holds a per-GPU lock and checks for existing compute PIDs
before launching any training. `ALLOW_SHARED_GPU=0` is the default; an intentional
sharing override is editable directly in the shell script. The shell passed syntax
checks; its full execution was not used to test the guard because automatic review
rejected launching a command that could start a long sweep.

### Earlier synthetic diagnostics (before the current changes)

Measured briefly on the available H100 80 GB, FP32 tensors with the project's
`torch.set_float32_matmul_precision('high')` setting. Covariance measurements use
three repetitions and compare with the previous per-image accumulation; numerical
agreement was checked. Results are wall-clock timings, not full training runs.

| Conv, batch 256 | Old covariance | Batched covariance | Speedup |
|---|---:|---:|---:|
| C_in=C_out=64, 3×3, H=W=32 | 29.47 ms | 7.68 ms | 3.84× |
| C_in=C_out=512, 3×3, H=W=4 | 103.14 ms | 5.57 ms | 18.50× |

Complete ResNet-50/CIFAR-size optimizer steps, synthetic batch 256, same process,
learning rate zero, two warm-up steps and three timed steps:

| Optimizer | Seconds/step |
|---|---:|
| SAM | 0.1849 |
| Optimized CondSAM | 0.2574 |

Here CondSAM cost about 1.39× SAM. These timings exclude data loading, logging,
DDP wrapper overhead and evaluation; the GPU was shared with existing work.
They do not establish a replacement full-epoch time. Earlier actual logs showed
roughly 335–344 seconds per epoch with the old implementation.

Full covariance is still substantial: this model has approximately 583 MiB of
A/B factor storage, with equally sized reusable sum buffers, in addition to model
activations and optimizer state. The optimization removes implementation overhead;
it does not eliminate the O(M*d_in² + M*d_out²) covariance computation.

Running Python processes retain their imported code. This change takes effect
on a new process; existing training has not been stopped or restarted.
