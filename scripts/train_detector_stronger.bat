@echo off
REM Stronger detector preset for full multi-pill images

echo ========== VAIPE DETECTOR - STRONGER PRESET ==========
echo.
echo This preset fine-tunes the current best MobileNetV3 detector:
echo   - start from existing best checkpoint
echo   - lower LR for fine-tuning
echo   - more epochs
echo   - slightly stronger hard-example replay
echo.

call .venv\Scripts\activate.bat

python detection_train.py ^
  --data-root "archive (1)/public_train/pill" ^
  --output-dir checkpoints\detection_mnv3_hardmining_ft_v2 ^
  --init-checkpoint checkpoints\detection_mnv3_hardmining_ft_lr5e5_e3\best_model.pth ^
  --model-name fasterrcnn_mobilenet_v3_large_fpn ^
  --resize-long-side 1024 ^
  --batch-size 2 ^
  --epochs 8 ^
  --patience 3 ^
  --lr 5e-5 ^
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
echo Results saved in: checkpoints\detection_mnv3_hardmining_ft_v2
echo.
pause
