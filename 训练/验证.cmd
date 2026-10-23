@echo off
chcp 65001 >nul
REM ============================================================
REM  独立验证：用验证集跑一次并输出报告
REM  验证集固定为 548 张 / 17040 实例，与原作品口径一致
REM  训练权重与验证集必须分开，不得用训练集成绩充当验证成绩
REM ============================================================
setlocal
cd /d "%~dp0"

set PY=..\云瞰街巷_技术优化版\.venv\Scripts\yolo.exe
set W=..\云瞰街巷_技术优化版\models\best.pt
if not exist "%PY%" (
  echo [错误] 找不到 %PY%
  pause
  exit /b 1
)

if not exist "%W%" (
  echo [错误] 找不到权重 %W%
  pause
  exit /b 1
)

if not exist "VisDrone-4.yaml" (
  echo [错误] 缺少 VisDrone-4.yaml，请先运行 准备数据集.py
  pause
  exit /b 1
)

"%PY%" detect val ^
  model="%W%" ^
  data=VisDrone-4.yaml ^
  imgsz=640 ^
  batch=8 ^
  device=0 ^
  conf=0.001 ^
  iou=0.7 ^
  plots=True ^
  project=runs ^
  name=val_current

echo.
echo 验证完成。报告位于 runs\val_current\
echo 关键指标：mAP50、mAP50-95、Precision、Recall，以及 car/van/truck/bus 分类别表现。
pause
