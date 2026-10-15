@echo off
REM Start the AI Shopping Assistant (run setup_and_verify.bat once first).
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run setup_and_verify.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe main.py %*
pause
