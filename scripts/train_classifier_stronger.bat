@echo off
REM Stronger classifier preset for VAIPE single-pill classification

echo ========== VAIPE CLASSIFIER - STRONGER PRESET ==========
echo.
echo This preset targets better minority-class performance:
echo   - focal loss
echo   - stronger class-balanced sampling
echo   - more epochs / patience
echo   - separate output directory
echo.

call .venv\Scripts\activate.bat

python train.py ^
  --data-root "archive (1)/public_train/pill" ^
  --output-dir checkpoints\retrain_cgimif_focal_s42_v2 ^
  --cache-dir checkpoints\retrain_cgimif_s42_det8\crop_cache_160 ^
  --image-size 160 ^
  --batch-size 64 ^
  --epochs 80 ^
  --patience 10 ^
  --lr 2e-4 ^
  --weight-decay 1e-4 ^
  --model-variant cg_imif_color_fusion ^
  --color-bins 8 ^
  --sampler-power 0.75 ^
  --loss-type focal ^
  --class-weight-power 0.75 ^
  --focal-gamma 1.5 ^
  --seed 42 ^
  --deterministic

echo.
echo ========== TRAINING COMPLETE ==========
echo Results saved in: checkpoints\retrain_cgimif_focal_s42_v2
echo.
pause
