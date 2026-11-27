@echo off
setlocal
cd /d "%~dp0"

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js / npm not found on PATH. Install Node.js 18+ first.
    exit /b 1
)

if not exist "%~dp0app\frontend\node_modules" (
    echo Installing frontend dependencies in app\frontend...
    call npm --prefix app\frontend install
    if errorlevel 1 exit /b 1
)

echo Starting FraudLens Vite React Frontend at http://127.0.0.1:5173 ...
call npm --prefix app\frontend run dev
