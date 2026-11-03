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
# CUDA toolkit inside the env (DeepSpeed needs it)
conda install -y -p ./env -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev libcurand-dev
mkdir -p ./env/etc/conda/activate.d && echo 'export CUDA_HOME="$CONDA_PREFIX"' > ./env/etc/conda/activate.d/cuda_home.sh
conda deactivate && conda activate ./env
pip install -r requirements.txt                          # transformers 4.47.1, deepspeed 0.15.4, wandb, pyyaml, ...
pip install flash-attn==2.7.4.post1 --no-build-isolation  # prebuilt wheel
wandb login                                              # or export WANDB_API_KEY; or --wandb_mode offline
```

Requires an NVIDIA driver for CUDA 12.4 (>= 550). Keep `transformers` at 4.47.x.

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
# OCR-VQA (80K images, rerun to resume)
gdown --folder https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_ -O $D/ocr_vqa
python scripts/download_ocrvqa.py --dataset $D/ocr_vqa/dataset.json --llava $D/llava_v1_5_mix665k.json --out $D/ocr_vqa/images
```

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
GPUS_PER_NODE=2              # processes on this node; > 1 runs data parallel (DeepSpeed)
GPU_IDS=(0 1)                # which GPUs, one id per process

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

Every combination runs in turn; `no_sam` runs once per VFM type and seed. The global batch stays 128 (paper recipe).
Two GPUs use ZeRO-3; one GPU also offloads the optimizer to CPU RAM.

```bash
DRY_RUN=1 bash scripts/v1_5/train_sweep.sh     # print the commands
bash scripts/v1_5/train_sweep.sh               # run
```

Extra arguments go to every run, e.g. `--wandb_mode offline`. A single run:

```bash
deepspeed --include localhost:0,1 llava/train/train_mem.py \
    --config configs/base.yaml --config configs/vfm/covft.yaml --config configs/sam/sam.yaml \
    --deepspeed scripts/zero3.json --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
    --output_dir checkpoints/llava-covft-sam --sam_rho 0.05 --seed 42
```

### Configs

| File | Holds |
|---|---|
| [configs/base.yaml](configs/base.yaml) | LLaVA-1.5 Stage 2 recipe (lr 2e-5, 1 epoch, bf16, ...) |
| [configs/vfm/*.yaml](configs/vfm) | vision-encoder tuning type |
| [configs/sam/*.yaml](configs/sam) | SAM method |
| [llava/optimizer/defaults.ini](llava/optimizer/defaults.ini) | default rho (0.01) and method options |

Precedence: command line > `--config` files (later wins) > INI. Each run saves its arguments to `<output_dir>/params.json`.

### SAM

Each step takes two passes: the gradient of the parameters matching `--sam_include` (default `vision_tower`) gives the
perturbation, then the usual step runs at the perturbed weights and the weights are restored. A step costs about twice
a `no_sam` step. With ZeRO-3 every GPU uses the same (all-reduced) perturbation; `--sam_local_perturbation True` (one per
GPU, as in the SAM paper) needs ZeRO-2.

> **Note:** tested on one H100 only; multi-GPU runs are unverified.

### Outputs

Each run writes `checkpoints/sweeps/<group>/llava-<vfm>-<method>-...-seed<seed>/` (model, `params.json`, W&B files).
Runs of one sweep share a W&B group in project `covft`; SAM runs also log `sam/clean_loss`, `sam/grad_norm`,
`sam/delta_norm`.

## Evaluation

```bash
CKPT_DIR=./checkpoints/sweeps/<group>/<run>   # the name must contain "llava"
for s in sqa gqa textvqa mmbench mmbench_cn mme; do bash scripts/v1_5/eval/$s.sh --ckpt $CKPT_DIR; done
for b in mmvp realworldqa ai2d omni ade coco; do bash scripts/v1_5/eval/run_benckmark_cambrain.sh --ckpt $CKPT_DIR --benchmark $b; done
```

MMBench writes a file to submit to OpenCompass instead of a score.

## Citation

```bibtex
@inproceedings{zhou2026covft,
  title={CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models},
  author={Zhou, Nan and Wang, Huiqun and Zheng, Yaoyan and Huang, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
