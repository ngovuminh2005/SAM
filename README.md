# Logits-SAM DPO + CondSAM

DPO với logits-SAM (ICLR 2026, *Sharpness-Aware Minimization in Logit Space Efficiently Enhances
Direct Preference Optimization*), dựa trên [code chính thức](https://github.com/RitianLuo/logits-sam-dpo)
và [alignment-handbook](https://github.com/huggingface/alignment-handbook). Thêm baseline **CondSAM**:
cùng chỉ nhiễu `lm_head` như logits-SAM, nhưng dùng hướng nhiễu CondSAM
([src/alignment/condsam_trainer.py](src/alignment/condsam_trainer.py)). Thí nghiệm: Mistral-7B
(`zephyr-7b-sft-full`) trên UltraFeedback, Table 2 của paper.

1. **Clone repo**

   ```bash
   git clone --branch sam-dpo --single-branch https://github.com/ngovuminh2005/SAM.git logits-sam-dpo
   cd logits-sam-dpo
   ```

2. **Tạo môi trường trong `./env`**

   ```bash
   conda create -y -p ./env python=3.11
   conda activate "$PWD/env"
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -e .
   pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   wandb login
   ```

   Môi trường eval (vLLM) riêng trong `./env_vllm`, không dùng chung với `./env`:

   ```bash
   conda create -y -p ./env_vllm python=3.12
   conda activate "$PWD/env_vllm"
   pip install uv && uv pip install vllm==0.32.0 --torch-backend=cu129
   ```

3. **Model và dataset**

   Tự tải lần chạy đầu. Tải trước:

   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('alignment-handbook/zephyr-7b-sft-full')"
   python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/ultrafeedback_binarized')"
   ```

4. **Chỉnh config**

   - [recipes/zephyr-7b-beta/dpo/config_full_paper_2gpu.yaml](recipes/zephyr-7b-beta/dpo/config_full_paper_2gpu.yaml):
     setting của paper: `lr=5e-7`, `beta=0.01`, batch tổng `128`, 1 epoch, `seed: 42`.
   - [scripts/launch_dpo.sh](scripts/launch_dpo.sh): biến môi trường `SEED`, `RHO`, `NUM_GPUS` (mặc định 2),
     `PER_DEVICE` (mặc định 4). Grad accumulation tự tính để batch tổng luôn là 128.
   - CondSAM: `CONDSAM_ARGS="--condsam_beta 0.95 --condsam_power_iters 1"` (mặc định như CondSAM gốc).
   - [scripts/sweep_condsam.sh](scripts/sweep_condsam.sh): lưới `RHOS` (mặc định `1e-5 1e-4 1e-3 1e-2 1e-1`)
     và `SEEDS`.

5. **Chạy** (trên node có GPU)

   ```bash
   bash scripts/run_mistral_paper.sh   # DPO thường, rồi DPO + logits-SAM (rho=1e-5, Table 7)
   bash scripts/sweep_condsam.sh       # sweep rho cho CondSAM
   ```

   Ví dụ: `RHOS="1e-2 1e-1" SEEDS="42 43" bash scripts/sweep_condsam.sh`.
   W&B project: `sam-dpo`. Log trong `logs/`, model trong `data/<run>/`. Chạy lại cùng lệnh để tiếp tục:
   run đã xong được bỏ qua, run dở tiếp từ checkpoint.
