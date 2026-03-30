@echo off

echo ========== BOOTSTRAP REAL PHOTO LABELS ==========
echo.
echo This creates draft JSON labels from the current model.
echo You still need to review the generated labels before training.
echo.

call .venv\Scripts\activate.bat

python bootstrap_real_photo_labels.py --data-root "data\user_real_photos\pill"

echo.
pause
