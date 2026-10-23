# 云瞰街巷 3.0 · 小空间人群使用画像

在使用户原有 YOLOv12m 四类车辆权重（car / van / truck / bus）及 ByteTrack 的基础上，新增“以人为主角”的人行观测模式与空间画像层：逐帧落盘、轨迹重建、停留与流向指标、一键生成热力图与回放。本版没有重新训练模型，也未验证检测精度提升。

## 启动成品

打开 `dist/云瞰街巷/云瞰街巷.exe`。请保留同目录的 `_internal`，分享时复制整个 `云瞰街巷` 文件夹，不能只发送 EXE。成品已带 Python 和 CUDA 运行库，不需要另装 Python 或 CUDA Toolkit；GPU 模式需要能支持所带 CUDA 12.8 运行库的 NVIDIA 驱动。没有可用 CUDA 设备时，“自动”模式使用 CPU。

默认模型在 `_internal/models/best.pt`；也可以在 EXE 旁新建 `models/best.pt` 覆盖默认选择，或在界面选择其他兼容的本地权重。默认使用 FP32，源码可通过 `EngineConfig(half=True)` 显式启用 GPU 半精度。程序不自动下载模型或安装依赖。

1. 选择图片、视频，或设置真实摄像头编号后使用摄像头。
2. 填写场地编号；如果掌握真实采集起点，可填写带时区的 ISO 日期时间（例如 `2026-10-19T09:00:00+08:00`）。留空时只保存源时间。
3. 可在预览中框选 ROI，调整水平统计线及连续驻留秒数，然后开始分析。
4. 置信度、IoU、推理尺寸和画面数量提示阈值修改后于下一帧生效；场地、来源、模型、设备、线和 ROI 在下一次会话设置。
5. 使用“停止”结束，数据会落盘；关闭窗口会先等待正在处理的帧结束，不强行销毁线程。

默认数据目录为 EXE 旁 `data`，无写权限时使用用户目录 `CloudlaneData`。可以用环境变量 `CLOUDLANE_DATA_DIR` 指定其他目录。每个会话独立保存 `session.json`、`frames.csv`、`detections.csv`、`events.csv` 和 `summary.json`；历史记录打开已保存结果，不重新分析原视频。CSV 使用 Excel 可读的 UTF-8 BOM。

“历史结果”默认显示中文摘要，可切换查看原始 JSON，也可直接打开该次分析的 CSV 数据目录。

2.1 版导出采用 schema 2：`frames.csv` 每帧记录实际生效的 `conf`、`iou`、`imgsz` 和 `density_threshold`，便于核对运行中调参。摘要和会话元数据保存应用版本、所用权重的 SHA-256，以及实际推理设备和半精度设置。`构建信息.json` 记录构建所用源码、EXE 和原权重的 SHA-256；打包验证报告会核对这些信息。

## 指标含义

| 显示 | 实际含义 |
| --- | --- |
| 当帧数量及数量曲线 | 该源时间的检测框数量，不是累计人流 |
| 轨迹数 | 本会话出现过的不同跟踪 ID 数；遮挡导致 ID 变化可能重复计数，不等于真实人数 |
| 上行 / 下行过线 | 跟踪目标底部中心跨越水平统计线的事件；同一 ID 每个方向最多计一次，使用滞回减少抖动 |
| 停留中 | 当前仍有目标位于框选区域且达到设定秒数；离开或漏失会重置 |
| 处理 FPS | 推理及统计处理速度，不是视频原始帧率或端到端吞吐 |

文件视频逐帧分析和记录，界面最多约 15 次/秒刷新并仅保留最近预览，曲线最多 300 点。视频、图片时间来自源时间，不把计算机处理快慢算成交通停留时间。切换来源或重新开始会创建全新模型/跟踪器和会话，避免 ID 串入。

原版 EXE 的实际跟踪为 ByteTrack；原数量曲线是当帧目标数量。此版的过线计数、连续 ROI 驻留、逐帧 CSV 与空间画像层均属本次新增功能，不能回写成旧版已经实现的成果。车辆模式只识别四类机动车；人行模式使用 COCO 通用权重识别行人与非机动车。两种模式的权重不同，指标不能混用。

## 人行模式与空间画像

1. “场景”默认即为“人行观测”，程序使用 COCO 通用权重 `models/yolo11s.pt`，只取行人、自行车、电动车（摩托车）三类，输入尺寸 1024。通用权重不依赖本项目数据，换点位不需要重新标注或训练。
2. 打开视频后可直接开始分析；框图（“框选驻留区域”）是可选的，框了之后同一块矩形会同时作为热力图的遮罩，颜色不会糊到立面、树冠和天空上。
3. 需要统计“一条断面上的穿越量”时，点「画断面线」，在画面里点两下画一条**地面断面线**。透视不会把地面上的直线变成曲线，所以画面里画的这条线在地面上就是一条真正的直线断面：人从一侧穿到另一侧才计数，斜着走、横着走都算。车辆观测里的“过线位置”是一条固定水平线，只适合正对断面；人行观测默认用画线方式。
4. 分析结束后程序自动重建空间画像，弹出热力图与指标面板；也可以随时点顶部“空间画像”按钮重算。会话目录 `space/` 下生成：
   - `tracks.csv` 轨迹表
   - `metrics.json` 全部指标
   - `heatmap.jpg` 存在强度（人·秒）
   - `heatmap_dwell.jpg` 仅停留段
   - `flow.jpg` 各网格主流向
   - `time_curve.png` 逐分钟节律
   - `playback.mp4` 轨迹回放片段
   - `summary.md` 可直接贴进汇报的文字摘要
4. 会话开始时自动保存一张 `poster.jpg` 底图，因此原始视频被移走或删除后，画像仍可重算。
6. 需要米制时再点“四点标定”：依次点地面矩形的左下、右下、右上、左上，再点一次矩形中心作校核，然后输入矩形实际宽高（米）。标定只对当前机位有效，结果写入会话目录的 `calibration.json`，所有距离类指标随之从“画面宽比例”升级为米。

命令行等价入口：

```powershell
python tools/space_pipeline.py data/sessions/<会话> --video 原视频.mp4 --playback 24
python tools/make_space_report.py --space data/sessions/<会话>/space --out 报告.docx
```

口径说明：驻留是会话内连续可见时长，不是真实停留意愿；未标定时所有距离按画面宽比例，不做米制换算；跟踪 ID 会断裂，轨迹条数不等于人数；交织是距离筛查值，不是冲突或安全评价。年龄与性别需可见人脸，俯视机位下样本偏差大，未纳入任何结论。

## 已做验证及边界

`validation/runtime_report.json` 是本机 RTX 5060 Laptop 的软件运行测试：2 次预热后，对 5 张原材料图片分别测 CPU / GPU，固定 imgsz=640、conf=0.5、IoU=0.45。CPU FP32 平均 87.11 ms，GPU FP32 平均 13.86 ms，GPU FP16 平均 16.32 ms；计时包含预处理、检测、后处理、ByteTrack 和统计，不包含加载、解码、导出及绘图。样本很少，不能当作稳定性能承诺，更不是旧 EXE 的实测对比或检测精度证明。

`validation/synthetic_photos_software_validation.avi` 是样本照片拼接的 12 帧软件验证片段，用来验证读视频、逐帧导出、会话重置和打包后运行。它不是真实连续交通视频，不用于评估流量、驻留或跟踪精度。没有基于独立标注集验证 mAP、漏检率、IDF1 或恶劣天气稳定性；“无错检漏检”“彻底解决 ID 切换”等表述没有此版证据支持。

## 源码运行与重建

建议使用 Windows 64 位 Python 3.12。`requirements.txt` 是主要运行依赖；`requirements-lock.txt` 是本次环境的完整版本快照。GPU / CPU 共用本构建的 CUDA 版 PyTorch。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe main.py
powershell -ExecutionPolicy Bypass -File build.ps1
```

双击 `启动源码.cmd` 可运行已安装的源码环境。构建使用 `Cloudlane.spec`，输出 onedir 目录；保留原 `models/best.pt`，不要用其他模型替代后仍宣称同一实验。测试覆盖计数口径、时间戳、ROI 中断、CSV/摘要一致性、热更新、线程退出、来源切换、旧信号隔离及 UI 生命周期。

本次回归测试为 48 项，结果见 `validation/tests_20261020.xml`。打包版 CPU / GPU 验证各处理同一合成片段的 12 帧；检查逐帧 CSV、源时间、事件及摘要一致性、参数快照、模型 SHA-256、应用版本和实际 FP32 设置。结果分别保存在 `validation/packaged_cpu_verification.json` 和 `validation/packaged_gpu_verification.json`，均属于软件运行验证，不代表真实交通或检测精度评估。

打包版可做无界面验证（示例在项目目录执行）：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
& 'dist/云瞰街巷/云瞰街巷.exe' --smoke-test 'validation/synthetic_photos_software_validation.avi' --device auto --max-frames 12 --data-root 'validation/packaged_data' --report 'validation/packaged_gpu.json' --screenshot 'validation/packaged_ui.png'
```

正常使用时不要设置 `QT_QPA_PLATFORM=offscreen`。来源及第三方信息见 `SOURCE_PROVENANCE.md`。
