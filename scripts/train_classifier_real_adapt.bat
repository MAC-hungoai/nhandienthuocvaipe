@echo off
REM Fine-tune classifier with extra real photos from the user

echo ========== VAIPE CLASSIFIER - REAL PHOTO ADAPTATION ==========
echo.
echo This preset keeps the best classifier and adapts it with:
echo   - original cropped pill data
echo   - your real photos under data\user_real_photos\pill
echo.

call .venv\Scripts\activate.bat

python train.py ^
  --data-root "archive (1)/public_train/pill" ^
  --extra-data-root "data\user_real_photos\pill" ^
  --output-dir checkpoints\retrain_cgimif_real_adapt_v1 ^
  --init-checkpoint checkpoints\retrain_cgimif_s42_det8\best_model.pth ^
  --model-variant cg_imif_color_fusion ^
  --epochs 24 ^
  --patience 6 ^
  --lr 8e-5 ^
  --weight-decay 1e-4 ^
  --loss-type focal ^
  --class-weight-power 0.75 ^
  --focal-gamma 1.5 ^
  --sampler-power 0.8 ^
  --num-workers 2 ^
  --seed 42 ^
  --no-deterministic

echo.
echo ========== TRAINING COMPLETE ==========
echo Results saved in: checkpoints\retrain_cgimif_real_adapt_v1
echo.
pause
