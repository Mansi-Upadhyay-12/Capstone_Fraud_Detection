@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at venv\Scripts\python.exe
    echo Run: python -m venv venv ^&^& .\venv\Scripts\pip.exe install -r app\backend\requirements.txt
    exit /b 1
)

if not exist "%~dp0app\backend\artifacts\stage2_imputer_scaler_train_1_34.joblib" (
    echo [ERROR] Missing Stage-2 preprocessing artifact in app\backend\artifacts\
    exit /b 1
)

set "PYTHONPATH=%~dp0"
echo Starting FraudLens FastAPI Backend at http://127.0.0.1:8010 ...
"%~dp0venv\Scripts\python.exe" -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8010
