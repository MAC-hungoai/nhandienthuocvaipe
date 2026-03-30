@echo off
REM Fine-tune detector with extra real photos from the user

echo ========== VAIPE DETECTOR - REAL PHOTO ADAPTATION ==========
echo.
echo This preset keeps the best detector and adapts it with:
echo   - original VAIPE training data
echo   - your real photos under data\user_real_photos\pill
echo.

call .venv\Scripts\activate.bat

python detection_train.py ^
  --data-root "archive (1)/public_train/pill" ^
  --extra-data-root "data\user_real_photos\pill" ^
  --output-dir checkpoints\detection_mnv3_real_adapt_v1 ^
  --init-checkpoint checkpoints\detection_mnv3_hardmining_ft_v2\best_model.pth ^
  --model-name fasterrcnn_mobilenet_v3_large_fpn ^
  --resize-long-side 1024 ^
  --batch-size 2 ^
  --epochs 6 ^
  --patience 3 ^
  --lr 3e-5 ^
  --weight-decay 1e-4 ^
  --sampler-power 0.5 ^
  --hard-mining-topk 0.20 ^
  --hard-mining-boost 2.0 ^
  --hard-mining-warmup 0 ^
  --score-threshold 0.30 ^
  --iou-threshold 0.50 ^
  --num-workers 2 ^
  --seed 42 ^
  --no-deterministic

echo.
echo ========== TRAINING COMPLETE ==========
echo Results saved in: checkpoints\detection_mnv3_real_adapt_v1
echo.
pause
