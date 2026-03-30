@echo off

echo ========== VALIDATE REAL PHOTO DATASET ==========
echo.

call .venv\Scripts\activate.bat

python validate_real_photo_dataset.py --data-root "data\user_real_photos\pill"

echo.
pause
