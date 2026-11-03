# CoVFT with SAM-style optimizers

LLaVA-1.5 Stage 2 (instruction tuning) built on [CoVFT](https://github.com/weeknan/CoVFT) (CVPR 2026,
[paper](https://arxiv.org/abs/2603.21077)), extended with SAM-style perturbation policies (`sam`, `asam`, `condsam`,
`friendlysam`, `fishersam`, `mongesam`), YAML/JSON/INI configs and W&B logging. Vision-encoder tuning types:
`context_moe_layernorm` (CoVFT), `fullft`, `lora`, `bias`, `vpt`.

All commands run from the repo root.

## Environment

```bash
conda create -y -p ./env python=3.10
conda activate ./env
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# DeepSpeed needs nvcc/CUDA_HOME at import and libcurand to build its CPU Adam
conda install -y -p ./env -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev libcurand-dev
mkdir -p ./env/etc/conda/activate.d && echo 'export CUDA_HOME="$CONDA_PREFIX"' > ./env/etc/conda/activate.d/cuda_home.sh
conda deactivate && conda activate ./env
pip install -r requirements.txt                          # transformers 4.47.1, deepspeed 0.15.4, wandb, pyyaml, ...
pip install flash-attn==2.7.4.post1 --no-build-isolation  # prebuilt wheel
wandb login                                              # or export WANDB_API_KEY; or --wandb_mode offline
```

Requires NVIDIA GPUs with a driver for CUDA 12.4 (>= 550). Do not upgrade `transformers` past 4.47.x: the code uses
`Trainer(tokenizer=...)` and `cache_position` handling that newer releases changed.

## Models and data

```bash
hf download lmsys/vicuna-7b-v1.5
hf download openai/clip-vit-large-patch14-336
hf download bert-base-uncased
mkdir -p checkpoints   # Stage 1 projector from the CoVFT authors -> checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin
gdown 1dnkJrKqQYjURejeXuq4S9YrlWdelKeup -O checkpoints/proj.zip && unzip -n checkpoints/proj.zip -d checkpoints/ && rm checkpoints/proj.zip

# Training data: LLaVA v1.5 Mix 665K (~60 GB of zips)
D=./data/llava_finetune
mkdir -p $D/coco $D/gqa $D/textvqa $D/vg $D/ocr_vqa
hf download liuhaotian/LLaVA-Instruct-150K llava_v1_5_mix665k.json --repo-type dataset --local-dir $D
wget -c http://images.cocodataset.org/zips/train2017.zip -P $D/coco && unzip -q $D/coco/train2017.zip -d $D/coco
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -P $D/gqa && unzip -q $D/gqa/images.zip -d $D/gqa
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P $D/textvqa && unzip -q $D/textvqa/train_val_images.zip -d $D/textvqa
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip  -P $D/vg && unzip -q $D/vg/images.zip  -d $D/vg
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P $D/vg && unzip -q $D/vg/images2.zip -d $D/vg
```

OCR-VQA has no archive: get `dataset.json` and `loadDataset.py` from [ocr-vqa.github.io](https://ocr-vqa.github.io/) and
save the images as `.jpg` in `$D/ocr_vqa/images/`.

<details><summary>Evaluation data</summary>

```bash
E=./playground/data/eval
gdown 1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy -O $E/llava_eval.zip && unzip -n $E/llava_eval.zip -d $E && rm $E/llava_eval.zip
mkdir -p $E/gqa/data $E/scienceqa/images $E/mmbench $E/mmbench_cn
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -P $E/gqa/data && unzip -q $E/gqa/data/questions1.2.zip -d $E/gqa/data
ln -sfn "$(realpath $D/gqa/images)" $E/gqa/data/images
curl -L https://gist.githubusercontent.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187/raw -o $E/gqa/data/eval.py
wget -c https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json   -P $E/scienceqa
wget -c https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/pid_splits.json -P $E/scienceqa
wget -c https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip -P $E/scienceqa/images && unzip -q $E/scienceqa/images/test.zip -d $E/scienceqa/images
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P $E/textvqa
ln -sfn "$(realpath $D/textvqa/train_images)" $E/textvqa/train_images
wget -c https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv    -P $E/mmbench
wget -c https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv -P $E/mmbench_cn
```

MME images are available on request from the [authors](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
(unzip to `$E/MME/MME_Benchmark_release_version/`). MMVP, RealWorldQA, AI2D, CV-Bench, ADE and COCO load from the HF Hub.
</details>

## Configure and run

Edit the settings at the top of [scripts/v1_5/train_sweep.sh](scripts/v1_5/train_sweep.sh):

```bash
GPUS_PER_NODE=1              # processes on this node; > 1 runs data parallel (DeepSpeed)
GPU_IDS=(0)                  # which GPUs, one id per process

VFM_TYPES=(covft)            # covft fullft lora bias vpt
METHODS=(no_sam sam)         # no_sam sam asam condsam friendlysam fishersam mongesam
RHOS=(0.01 0.05)             # perturbation radius (not used by no_sam)

# Only used by the matching method:
FRIENDLYSAM_SIGMAS=(1.0)
FRIENDLYSAM_LAMBDAS=(0.9)
FISHERSAM_ETAS=(1.0)
CONDSAM_BETAS=(0.95)
CONDSAM_POWER_ITERS=(1)

SEEDS=(42)
```

Every combination runs in turn, methods innermost; `no_sam` runs once per VFM type and seed. The global batch is 128
(the paper's recipe) and `--gradient_accumulation_steps` is derived from `PER_DEVICE_BATCH` (4) and the GPU count.
Several GPUs use ZeRO-3 ([scripts/zero3.json](scripts/zero3.json)); one GPU also offloads the optimizer state to CPU RAM
([scripts/zero3_offload_optimizer.json](scripts/zero3_offload_optimizer.json)), since a 7B full fine-tune needs ~110 GB.

```bash
DRY_RUN=1 bash scripts/v1_5/train_sweep.sh     # print the commands
bash scripts/v1_5/train_sweep.sh               # run
```

Arguments after the script name go to every run, e.g. `bash scripts/v1_5/train_sweep.sh --wandb_mode offline`. A single
run without the launcher:

```bash
deepspeed --include localhost:0,1 llava/train/train_mem.py \
    --config configs/base.yaml --config configs/vfm/covft.yaml --config configs/sam/sam.yaml \
    --deepspeed scripts/zero3.json --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
    --output_dir checkpoints/llava-covft-sam --sam_rho 0.05 --seed 42
```

The original scripts in `scripts/v1_5/` (`covft_exp4_last12L_layernorm.sh`, `finetune_*_vfm.sh`) still work for 2 GPUs.

### Configs

| File | Holds |
|---|---|
| [configs/base.yaml](configs/base.yaml) | LLaVA-1.5 Stage 2 recipe shared by every run (lr 2e-5, 1 epoch, cosine, bf16, ...) |
| [configs/vfm/*.yaml](configs/vfm) | vision-encoder tuning type (`covft` = CoMoE from layer 11, 4 dense experts) |
| [configs/sam/*.yaml](configs/sam) | perturbation policy |
| [llava/optimizer/defaults.ini](llava/optimizer/defaults.ini) | one radius per policy (0.01) and policy options; `--sam_config` selects another INI |

Precedence: command-line flags, then `--config` files (later files win), then the INI, then code defaults. YAML/JSON keys are
argument names; a section is an optional prefix (`sam: {rho: 0.05}` sets `--sam_rho`). Unknown keys are errors. Each run
writes the resolved arguments to `<output_dir>/params.json`.

### SAM

Each micro-batch takes two passes. The first computes the gradient of the parameters matching `--sam_include` (default
`vision_tower`: CoMoE experts, gates, context cross-attention and LayerNorms for CoVFT) with `torch.autograd.grad`, so
DeepSpeed's accumulated gradients are untouched; the policy turns it into a perturbation. The second pass is the usual
training step at the perturbed weights, so every trainable parameter (LLM and projector included) descends with the
gradient taken there; then the exact weights are restored. Perturbing the 7B LLM as well (`--sam_include ''`) does not fit
in 80 GB. A step costs about twice a `no_sam` step.

Under ZeRO-3 the perturbation is applied through `deepspeed.zero.GatheredParameters`, and every rank uses the all-reduced
first-pass gradient (`m` = global batch); CondSAM then pools its Kronecker factors over ranks.
`--sam_local_perturbation True` gives each rank its own perturbation (SAM paper m-sharpness) and requires ZeRO-2
(`--deepspeed scripts/zero2.json`).

### Multi-GPU

> **Note:** SAM has been checked on one H100 only (ZeRO-3, ZeRO-3 with CPU offload, ZeRO-2; tiny LLM with the real CLIP
> and BERT). Multi-GPU runs, speed and accuracy are unverified.

```bash
python -m unittest tests.test_config tests.test_sam_trainer   # configs; SAM step vs. a hand-written reference (GPU)
```

### Outputs

Each run writes `checkpoints/sweeps/<group>/llava-<vfm>-<method>-...-seed<seed>/` with the model, `params.json` and the W&B
files. W&B runs of one sweep share a group (`WANDB_PROJECT=covft`) and are named after what varies
(`context_moe_layernorm-sam-rho0.05-lr2e-05-seed42`); besides HF's `train/*`, SAM runs log `sam/clean_loss` (loss at the
unperturbed weights), `sam/grad_norm` and `sam/delta_norm`. HF reports `train/loss` summed over the accumulated
micro-batches with DeepSpeed; gradients are averaged correctly.

## Evaluation

```bash
CKPT_DIR=./checkpoints/sweeps/<group>/<run>   # the name must contain "llava"
for s in sqa gqa textvqa mmbench mmbench_cn mme; do bash scripts/v1_5/eval/$s.sh --ckpt $CKPT_DIR; done
for b in mmvp realworldqa ai2d omni ade coco; do bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark $b; done
```

MMBench writes files for submission to OpenCompass instead of a score. `run_eval.sh` runs everything for one checkpoint;
its MMMU step fails because the repo has no `mmmu/` folder.

## Citation

```bibtex
@inproceedings{zhou2026covft,
  title={CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models},
  author={Zhou, Nan and Wang, Huiqun and Zheng, Yaoyan and Huang, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
