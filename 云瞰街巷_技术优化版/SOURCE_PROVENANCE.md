# 源码与模型来源

本目录是依据用户提供的原程序功能重新实现的可维护 Python / PySide6 应用，不是从原 EXE 还原出的完整源码。原始压缩包和 EXE 未被修改，也未运行原 EXE。

`models/best.pt` 来自用户的“大数据应用_李吉坤 _总文件夹.zip”内原应用 `_internal/best.pt`，保留原有训练权重。SHA-256：

`975aefac6e72860b240d7c9ef6f4785899725373a45e4c7ab8bfa57a27c2a384`

权重含 car、van、truck、bus 四类，原权重记录为 YOLOv12m。此版未重新训练模型，未修改网络结构，不能写成团队从零研发 YOLOv12 算法或新增行人检测。

检测调用 [Ultralytics](https://github.com/ultralytics/ultralytics) 8.4.37，跟踪调用该版本的 ByteTrack；[YOLOv12 原始项目](https://github.com/sunsmarterjie/yolov12) 和 [ByteTrack 原始项目](https://github.com/ifzhang/ByteTrack) 应按实际使用情况署名。应用界面、线程隔离、会话统计、ROI 驻留、过线事件、结构化存储和测试属于本次工程改进。

第三方运行库包括 PyTorch / torchvision、Ultralytics、PySide6、OpenCV、NumPy、SciPy、lap 等；已打包库的包元数据和随库文件保留在 `_internal`。分发或商业使用时应沿用相应项目的许可条件。

源码入口 `main.py`；界面 `traffic_app/ui.py`；推理线程 `traffic_app/engine.py`；统计 `traffic_app/analytics.py`；存储 `traffic_app/storage.py`；测试 `tests/`；构建 `Cloudlane.spec` 和 `build.ps1`。
