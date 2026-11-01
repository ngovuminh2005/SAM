# WiSE-FT + CondSAM

1. **Clone repo**

   ```bash
   git clone --branch wise-ft --single-branch https://github.com/ngovuminh2005/SAM.git wise-ft
   cd wise-ft
   ```

2. **Tạo và activate môi trường trong `./env`**

   ```bash
   conda env create --prefix ./env --file environment.yml
   conda activate "$PWD/env"
   python -m pip check
   ```

   Nếu env đã tồn tại: `conda env update --prefix ./env --file environment.yml`.

3. **Chuẩn bị dataset**

   Máy hiện tại đã có đủ ImageNet train/val và OOD: ImageNet-V2, R, A, Sketch
   trong `./datasets`. Trên máy mới, chạy trong tmux `wise-ft`:

   ```bash
   bash prepare_imagenet_wiseft.sh
   bash prepare_imagenet_tests.sh
   ```

4. **Chỉnh config CondSAM**

   - [optimizer/defaults.ini](optimizer/defaults.ini): `condsamRho` trong `[SAM]`;
     `beta`, `eps`, `power_iters` trong `[CondSAM]`.
   - [configs/condsam_sweep.yaml](configs/condsam_sweep.yaml): các giá trị quét
     `sam-rho`, `condsam-beta`, `condsam-power-iters`; cùng `lr`, `wd`, `epochs`,
     `batch-size`. Giá trị trong sweep ghi đè INI.
   - [scripts/sweep_condsam.sh](scripts/sweep_condsam.sh) đã trỏ đúng config.
     Optimizer cập nhật trọng số mặc định là AdamW.

5. **Chạy CondSAM**

   ```bash
   wandb login
   bash scripts/sweep_condsam.sh
   ```

   W&B project: `wise-ft`. Mỗi run tự eval 5 dataset × 11 alpha khi train xong;
   kết quả nằm trong `models/<run>/evaluation_summary.json` và W&B.
