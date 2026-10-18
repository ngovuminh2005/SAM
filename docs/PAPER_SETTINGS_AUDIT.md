# Coverage of the supplied OrthoSAM PDF

Source: `26550_OrthoSAM_Orthogonal_Pert (1).pdf`, 27 pages.
Reviewed the experiment section (pp. 6–9), Appendix B (pp. 18–19),
Algorithm 1 (p. 19), and searched the remaining pages for OOD protocols.

## Experiments and launch coverage

| PDF setting | Count | Project support |
|---|---:|---|
| CIFAR-10 × ResNet-50, WRN-16-4, PyramidNet-110, WRN-28-10, DenseNet-121, Tiny-ViT | 6 | Six INI presets; `SECTION=image` |
| CIFAR-100 × same backbones | 6 | Six INI presets; `SECTION=image` |
| Tiny-ImageNet × ResNet-50, ResNet-101 | 2 | Two INI presets; `SECTION=image` |
| Tiny-ViT Adam comparisons on CIFAR-10/100 | 2 dataset/backbone pairs | `SECTION=adam`; Adam, Adam_SAM, Adam_FriendlySAM plus OrthoSAM/CondSAM |
| BERT-base GLUE: CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, WNLI, STS-B | 9 tasks | `NLP_GLUE_BERT.ini`; `SECTION=glue` |
| Label noise: CIFAR-10/100 × WRN-16-4 × 20%, 40%, 80% | 6 | `SECTION=noise`; fixed corruption seed, clean test set |
| Hessian: CIFAR-10 × DenseNet-121, WRN-16-4 | 2 backbone settings | `SECTION=hessian`; trained checkpoints required |
| Per-epoch training cost | Existing image settings | Logged by vision/GLUE entrypoints and W&B |

Counts are dataset/backbone/task settings, not numbers of optimizer/radius/repeat runs.
CondSAM is an extension to the comparison; it is not a method reported in this PDF.
No experiments have been launched by the integration work.

## OOD clarification

There is no dedicated OOD experiment/protocol/table in this PDF: no CIFAR-C,
CIFAR-P, ImageNet-C/A/R, cross-domain test dataset, or specified corruption test
suite. Domain/data shifts appear in the introduction as motivation. The measured
robustness experiment is **training-label noise**, evaluated on clean test data
(Section 5.5, p. 8; Table 6b, p. 9). This is distinct from OOD evaluation.

## Recipe alignment

Table 7, p. 18: all 14 image settings use 200 epochs, momentum 0.9,
no Nesterov. CIFAR convolutional recipes use one-epoch warmup then cosine LR,
crop/flip, Cutout probability 0.5, normalization and smoothing 0.1.
Tiny-ViT uses LR 0.001, batch 256, weight decay 0.0005, ten-epoch warmup,
no Cutout/smoothing, and rho 0.01. Tiny-ImageNet ResNets use batch 125,
LR 0.05, weight decay 0.001, RandAugment magnitude 10 and smoothing 0.1.

Fixed the five debug epoch defaults (CIFAR-10/100 ResNet-50, CIFAR-10 WRN-28-10,
Tiny-ImageNet ResNet-50/101), and CIFAR-10 PyramidNet batch size 64 -> 256.
Other explicit Table 7 batch/LR/momentum/weight-decay values matched those presets.
The image sweep uses the PDF radius grid: 0.01, 0.045, 0.05, 0.1, 0.15, 0.5;
Tiny-ViT overrides this grid with 0.01. Default repeat count is three.
A grid produces candidates; it does not choose or aggregate the best radius automatically.
Some individual legacy method-specific rho defaults are outside this grid;
the paper sweep explicitly overrides the method radius and scheduler base radius.

Added missing GLUE INI: three epochs, batch 32, LR 2e-5, gradient clip norm 1,
linear schedule in the entrypoint. Existing task rho table matches p. 18:
CoLA 0.002; RTE/STS-B/WNLI 0.004; remaining tasks 0.005.
The PDF does not specify GLUE momentum/weight decay fully; this INI chooses
existing SGD-family momentum 0.9 and weight decay 0, smoothing 0.
The GLUE code evaluates labeled validation splits; official test submissions are
not implemented even though Table 5 is captioned as test results.

Hessian defaults match the appendix: training split, 20 batches of 128,
100 Lanczos steps, full reorthogonalization, five leading eigenvalues, no augmentation.
Results are an estimate from these 20 batches, not a full-dataset exact spectrum.

Noise generation now samples uniformly among incorrect classes, with a fixed seed
and an exact selected fraction. Previous remapping to the next class biased its
probability. The PDF does not explicitly say whether a replacement may retain the
original class; this implementation retains the project's intent of changed labels.

## Remaining reproduction limits

- Existing OrthoSAM ascent multiplies the NS projection by rho without the explicit
  per-matrix Frobenius renormalization in Algorithm 1, line 8. Matching training
  settings does not resolve this algorithmic difference.
- OrthoSAM's name filter does not recognize every classifier layout, including
  `linear_head.1.weight` in SimpleViT; that head can enter the matrix branch contrary
  to the stated classifier fallback. This audit does not change OrthoSAM's algorithm.
- Tiny-ViT is the project's SimpleViT implementation. The PDF does not provide
  enough architectural detail to prove it is the identical Tiny-ViT implementation.
- Tiny-ImageNet images are exported through the HF mirror and JPEG re-encoding.
- GLUE/model downloads need network/cache access; vision datasets are prepared.
- ImageNet/VGG/Tiny-ImageNet ViT-B presets are extra project presets, outside
  Table 7. Full ImageNet data is not prepared and is not required by Table 7.

## W&B

All vision optimizers, GLUE and Hessian analysis log to project `CondSAM` via
`utility/tracking.py`. Only rank 0 starts a run. Vision logs registered train/test
metrics, LR, rho and epoch time; GLUE logs aggregated training loss and official
validation metrics; Hessian logs eigenvalues. Config includes actual rank-0 seed,
label-noise rate/seed, and optimizer parameters.

Shell sweep values are editable in `run_condsam_rho.sh` and `run_paper_sweep.sh`.
W&B is online after login, otherwise it saves offline runs. No key is stored in
source. Use `env/bin/wandb login` before an online sweep; use `wandb sync` for
offline runs later. The integration check successfully initialized, logged and finished an online
run in project CondSAM. Twelve unit checks passed, including two-rank Gloo and
W&B lifecycle/metric logging. All five sweep sections passed command dry-runs;
full training experiments have not been launched.

## Added OOD extension and eval interval

The project now includes six post-training OOD settings and separate launchers
under `scripts/ood/`; see `docs/OOD.md`. These extend the project and are not
experiments claimed in the source PDF. Training now evaluates every 20 epochs
and at the final epoch; checkpoint saving remains independent.
