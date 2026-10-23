@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 缺少 .venv，请按 README 安装开发环境。
  pause
  exit /b 1
)
set "YOLO_AUTOINSTALL=false"
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
