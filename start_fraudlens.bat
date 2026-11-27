@echo off
setlocal
cd /d "%~dp0"

echo ============================================================================
echo  FraudLens - Intelligent Transaction Risk Analysis
echo  Starting Backend (127.0.0.1:8010) and Frontend (127.0.0.1:5173)...
echo ============================================================================

start "FraudLens Backend (8010)" /D "%~dp0" cmd /k call start_backend.bat
timeout /t 3 /nobreak >nul
start "FraudLens Frontend (5173)" /D "%~dp0" cmd /k call start_frontend.bat

echo.
echo Backend API Docs : http://127.0.0.1:8010/docs
echo Backend Health   : http://127.0.0.1:8010/api/health
echo Frontend UI      : http://127.0.0.1:5173
echo.
echo Both services launched in separate windows. Close those windows to stop them.
