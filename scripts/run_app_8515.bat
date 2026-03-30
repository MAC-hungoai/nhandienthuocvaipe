@echo off

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Khong tim thay .venv\Scripts\python.exe
  echo Hay tao moi truong ao truoc:
  echo   python -m venv .venv
  echo   .\.venv\Scripts\activate
  echo   pip install -r requirements.txt
  exit /b 1
)

.\.venv\Scripts\python.exe -m streamlit run app_streamlit_modern.py --server.port 8515
