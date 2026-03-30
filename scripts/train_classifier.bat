@echo off
REM Script để train Single-Pill Classification model trên Windows
REM

echo ========== VAIPE SINGLE-PILL CLASSIFIER TRAINING ==========
echo.
echo Sap bat dau training voi default hyperparameters.
echo De tuning hyperparameters, xem: python train.py --help
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run training
python train.py ^
  --data-root "archive (1)/public_train/pill" ^
  --output-dir checkpoints ^
  --image-size 160 ^
  --batch-size 64 ^
  --epochs 50 ^
  --patience 6 ^
  --lr 3e-4 ^
  --model-variant cg_imif_color_fusion ^
  --color-bins 8 ^
  --sampler-power 0.5 ^
  --loss-type cross_entropy ^
  --deterministic

echo.
echo ========== TRAINING COMPLETE ==========
echo Results saved in: checkpoints/
echo.
echo Check:
echo   - best_model.pth - Best model based on validation loss
echo   - history.json - Training history
echo   - test_metrics.json - Test set evaluation
echo   - training_curves.png - Loss/accuracy curves
echo   - confusion_matrix.png - Per-class predictions
echo.

pause
