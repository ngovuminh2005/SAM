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
# flash-attn: install the prebuilt wheel directly (torch 2.5, cxx11abi=FALSE, cp310)
pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -c "import torch, flash_attn; print(torch.cuda.is_available(), flash_attn.__version__)"
# W&B key, kept in the env (env/ is gitignored); train_sweep.sh loads it too. Or `wandb login`; or --wandb_mode offline
echo '[ -n "${WANDB_API_KEY:-}" ] || export WANDB_API_KEY=<your key>' > ./env/etc/conda/activate.d/wandb.sh
hf auth login                                            # or export HF_TOKEN; anonymous downloads get rate limited (429)
```

Requires an NVIDIA driver for CUDA 12.4 (>= 550). Keep `transformers` at 4.47.x.

`pip install flash-attn==2.7.4.post1 --no-build-isolation` fails with `[Errno 18] Invalid cross-device link` when
`/tmp` and `~/.cache/pip` are on different filesystems (e.g. tmpfs + NFS): flash-attn's `setup.py` downloads the wheel
then `os.rename`s it into the pip cache. Use the direct wheel URL above (check the ABI with
`python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`), or keep pip's temp dir on the same filesystem:
`TMPDIR=~/.cache/tmp pip install flash-attn==2.7.4.post1 --no-build-isolation` (after `mkdir -p ~/.cache/tmp`).

## Models and data

```bash
hf download lmsys/vicuna-7b-v1.5
hf download openai/clip-vit-large-patch14-336
hf download bert-base-uncased
mkdir -p checkpoints   # Stage 1 projector from the CoVFT authors -> checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin
gdown 1dnkJrKqQYjURejeXuq4S9YrlWdelKeup -O checkpoints/proj.zip && unzip -n checkpoints/proj.zip -d checkpoints/ && rm checkpoints/proj.zip

# Training data: LLaVA v1.5 Mix 665K (~75 GB of zips). Images come from HF mirrors of the official zips (same
# files and layout): Stanford's servers (GQA, VG) time out from our cluster and cocodataset.org is slow.
D=./data/llava_finetune
mkdir -p $D/coco $D/gqa $D/textvqa $D/vg $D/ocr_vqa
hf download liuhaotian/LLaVA-Instruct-150K llava_v1_5_mix665k.json --repo-type dataset --local-dir $D
hf download pcuenq/coco-2017-mirror train2017.zip --repo-type dataset --local-dir $D/coco && unzip -q -n $D/coco/train2017.zip -d $D/coco
hf download BoyangZ/GQA_llava images.zip --repo-type dataset --local-dir $D/gqa && unzip -q -n $D/gqa/images.zip -d $D/gqa
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P $D/textvqa && unzip -q -n $D/textvqa/train_val_images.zip -d $D/textvqa
hf download BoyangZ/VisualGenome_VG_100K_1_and_2 images.zip images2.zip --repo-type dataset --local-dir $D/vg
unzip -q -n $D/vg/images.zip -d $D/vg && unzip -q -n $D/vg/images2.zip -d $D/vg
# OCR-VQA: the 80K images LLaVA uses, already as .jpg
hf download BoyangZ/OCR_VQA ocr_vqa_images_llava_v15.zip --repo-type dataset --local-dir $D/ocr_vqa
unzip -q -n $D/ocr_vqa/ocr_vqa_images_llava_v15.zip -d $D/ocr_vqa
rm -rf $D/.cache $D/*/.cache                             # hf download bookkeeping

# Check: 0 missing images, 0 fully masked samples (loads the 1 GB json: run on a compute node, not the login node)
python scripts/check_data.py --samples 200
rm $D/*/*.zip                                            # ~70 GB, once the check passes
```

Expected counts: coco/train2017 118287, gqa/images 148854, ocr_vqa/images 208473, textvqa/train_images 25119,
vg/VG_100K 64346, vg/VG_100K_2 43903; `check_data.py` reports 665298 samples and `missing images: 0`.

Layout: `$D/{coco/train2017, gqa/images, ocr_vqa/images, textvqa/train_images, vg/VG_100K, vg/VG_100K_2}`.
Official sources, if reachable: `http://images.cocodataset.org/zips/train2017.zip`,
`https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip`, `https://cs.stanford.edu/people/rak248/VG_100K_2/images{,2}.zip`;
OCR-VQA from Amazon URLs with `gdown --folder https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_ -O $D/ocr_vqa`
then `python scripts/download_ocrvqa.py --dataset $D/ocr_vqa/dataset.json --llava $D/llava_v1_5_mix665k.json --out $D/ocr_vqa/images`
(rerun to resume; dead links are listed in `failed_downloads.txt`).

<details><summary>Evaluation data</summary>

```bash
E=./playground/data/eval
gdown 1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy -O $E/llava_eval.zip && unzip -n $E/llava_eval.zip -d $E && rm $E/llava_eval.zip
mkdir -p $E/gqa/data $E/scienceqa/images $E/mmbench $E/mmbench_cn
# GQA testdev_balanced_questions.json (from Stanford's questions1.2.zip), rebuilt from the lmms-lab mirror
hf download lmms-lab-encoder/GQA testdev_balanced_instructions/testdev-00000-of-00001.parquet --repo-type dataset --local-dir $E/gqa/hf
python scripts/gqa_questions_from_hf.py --parquet $E/gqa/hf/testdev_balanced_instructions/testdev-00000-of-00001.parquet \
    --out $E/gqa/data/testdev_balanced_questions.json && rm -rf $E/gqa/hf
ln -sfn "$(realpath $D/gqa/images)" $E/gqa/data/images
curl -L https://gist.githubusercontent.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187/raw -o $E/gqa/data/eval.py
wget -c https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json   -P $E/scienceqa
wget -c https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/pid_splits.json -P $E/scienceqa
wget -c https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip -P $E/scienceqa/images && unzip -q $E/scienceqa/images/test.zip -d $E/scienceqa/images
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P $E/textvqa
ln -sfn "$(realpath $D/textvqa/train_images)" $E/textvqa/train_images
wget -c https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv    -P $E/mmbench
wget -c https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv -P $E/mmbench_cn
# Benchmarks read from the HF Hub at eval time; cache them now
for r in MMVP/MMVP lmms-lab/RealWorldQA lmms-lab/ai2d nyu-visionx/CV-Bench \
         SaiCharithaAkula21/benchmark_ade_manual SaiCharithaAkula21/benchmark_coco_filtered; do
    hf download $r --repo-type dataset > /dev/null; done
```

The ScienceQA S3 bucket times out from our cluster. The same 2017 test images are in the
[derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) test parquet, whose rows follow
`pid_splits.json["test"]` in order: write each row's `image.bytes` to `$E/scienceqa/images/test/<pid>/image.png`.

MME images are available on request from the [authors](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
(unzip to `$E/MME/MME_Benchmark_release_version/`).
</details>

## Configure and run

Edit the settings at the top of [scripts/v1_5/train_sweep.sh](scripts/v1_5/train_sweep.sh):

```bash
GPUS_PER_NODE=${GPUS_PER_NODE:-2}   # processes on this node; > 1 runs data parallel (DeepSpeed)
GPU_IDS=...                         # from $GPU_IDS, default 0 .. GPUS_PER_NODE-1

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
All runs use ZeRO-3. With 1-2 GPUs the optimizer state (~84 GB fp32) is offloaded to CPU RAM, since on two 80 GB GPUs
it alone takes 42 GB each and SAM-style runs go out of memory; from 4 GPUs it stays on the GPUs (override with
`DS_CONFIG=...`). On a one-GPU allocation run
`GPUS_PER_NODE=1 bash scripts/v1_5/train_sweep.sh` (or `GPU_IDS="2"` to pick the GPU).

One run is ~5200 optimizer steps (665K samples / 128). Every 500 steps the run saves `checkpoint-<step>/` (model +
DeepSpeed optimizer state, ~110 GB; only the latest is kept) and at the end the final model (~14 GB). To resume after a
crash, rerun the sweep with its group name (the folder under `checkpoints/sweeps/`): each run continues from its
latest checkpoint. Keep the same GPU count and batch settings; delete `checkpoint-*/` once a run has finished.

```bash
WANDB_GROUP=sweep-20261103-090615 GPUS_PER_NODE=1 bash scripts/v1_5/train_sweep.sh
```

```bash
DRY_RUN=1 bash scripts/v1_5/train_sweep.sh     # print the commands
bash scripts/v1_5/train_sweep.sh               # run
```

Extra arguments go to every run, e.g. `--wandb_mode offline`. A single run:

```bash
deepspeed --include localhost:0,1 llava/train/train_mem.py \
    --config configs/base.yaml --config configs/vfm/covft.yaml --config configs/sam/sam.yaml \
    --deepspeed scripts/zero3_offload_optimizer.json --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
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
