@echo off
chcp 65001 >nul
REM ============================================================
REM  云瞰街巷 · YOLOv12m 四类机动车训练
REM  超参数与原作品 2026-04 训练记录一致（设计文档 4.2 + args.yaml）
REM ============================================================
setlocal
cd /d "%~dp0"

set PY=..\云瞰街巷_技术优化版\.venv\Scripts\yolo.exe
if not exist "%PY%" (
  echo [错误] 找不到 %PY%
  echo 请先确认 云瞰街巷_技术优化版\.venv 已安装依赖。
  pause
  exit /b 1
)

if not exist "VisDrone-4.yaml" (
  echo [错误] 缺少 VisDrone-4.yaml，请先运行 准备数据集.py
  pause
  exit /b 1
)

echo.
echo 开始训练 YOLOv12m on VisDrone-4 ...
echo 训练集 6471 张 / 验证集 548 张 / epochs 200 / batch 8 / imgsz 640
echo 显存不足请改用 低显存训练.cmd
echo.

"%PY%" detect train ^
  model=yolo12m.yaml ^
  data=VisDrone-4.yaml ^
  epochs=200 ^
  batch=8 ^
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
  workers=8 ^
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
  name=visdrone4_yolo12m ^
  exist_ok=False

echo.
echo 训练结束。权重位于 runs\visdrone4_yolo12m\weights\best.pt
echo 请先运行 验证.cmd 核对指标，再考虑替换程序中的 models\best.pt
pause
