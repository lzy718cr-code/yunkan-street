@echo off
chcp 65001 >nul
REM ============================================================
REM  低显存版（本机 RTX 5060 Laptop 8GB 用）
REM  与 训练.cmd 的区别：batch 8->4，workers 8->4，并开启
REM  分批累积。精度略有影响，结论以 训练.cmd 为准。
REM ============================================================
setlocal
cd /d "%~dp0"

set PY=..\云瞰街巷_技术优化版\.venv\Scripts\yolo.exe
if not exist "%PY%" (
  echo [错误] 找不到 %PY%
  pause
  exit /b 1
)

"%PY%" detect train ^
  model=yolo12m.yaml ^
  data=VisDrone-4.yaml ^
  epochs=200 ^
  batch=4 ^
  imgsz=640 ^
  device=0 ^
  optimizer=SGD ^
  lr0=0.01 ^
  lrf=0.01 ^
  momentum=0.937 ^
  weight_decay=0.0005 ^
  warmup_epochs=3.0 ^
  close_mosaic=10 ^
  patience=100 ^
  amp=True ^
  seed=0 ^
  deterministic=True ^
  workers=4 ^
  cache=False ^
  hsv_s=0.7 ^
  hsv_v=0.4 ^
  translate=0.1 ^
  scale=0.5 ^
  fliplr=0.5 ^
  mosaic=1.0 ^
  copy_paste=0.0 ^
  nbs=64 ^
  box=7.5 ^
  cls=0.5 ^
  dfl=1.5 ^
  project=runs ^
  name=visdrone4_yolo12m_lowvram ^
  exist_ok=False

echo.
echo 训练结束。权重位于 runs\visdrone4_yolo12m_lowvram\weights\best.pt
pause
