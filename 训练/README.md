# 训练说明 · 云瞰街巷 YOLOv12m 四类机动车

> 本文回答「怎么训练模型」，并把可运行的脚本都放在本目录。
> 所有超参数来自原作品《云瞰车流 智绘通途》的设计与开发文档 4.2 节及随包附带的
> `args.yaml`，不是临时编的参数。

---

## 一、结论先说

原模型**可以完整复现**，因为它用的数据集、参数、成绩都有明确记录；但**训练代码和数据集本身不在这台机器上**，需要重新准备一次。

好消息：Ultralytics 8.4.37 内置了 VisDrone2019 的一键下载 + 标注转换脚本，
比原团队 2026 年 4 月手写转换代码省事得多。

三条路线，按目的选：

| 路线 | 目的 | 工作量 | 适合现在做吗 |
|---|---|---|---|
| **A. 复现四类基线** | 重训一份 car/van/truck/bus 权重，作为可核对的基线 | 低 | 建议做，作为后续一切对比的参照 |
| **B. 加类别** | 补上行人、自行车、电动车，解锁「人车交织」研究 | 中高 | 结合方案第三阶段，课题真正需要 |
| **C. 场地微调** | 用西号巷等自家场地视频微调，贴合本地场景 | 中 | 数据够就做，效果最直接 |

---

## 二、原模型是怎么训出来的

| 项目 | 实际设置 |
|---|---|
| 数据集 | VisDrone2019-DET（天津大学，6471 训练 / 548 验证 / 1610 test-dev），约 2.3GB |
| 数据特点 | 无人机航拍，960×540 或 1360×765，最小目标仅 15×15 像素，含白天/黄昏/夜间、晴/雨/雾 |
| 类别 | 4 类：car、van、truck、bus（原图为 10 类，过滤得到） |
| 模型 | YOLOv12m，从 `yolo12m.yaml` 预训练权重起步 |
| 关键超参 | `epochs=200, batch=8, imgsz=640, optimizer=SGD, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3, close_mosaic=10, patience=100, amp=True, seed=0, deterministic=True, workers=8` |
| 增强 | `mosaic=1.0, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, copy_paste=0.0` |
| 硬件 | RTX 4090 24GB，训练约 24 小时（表 4-2 累计 15149 秒） |
| 标注转换 | 原团队手写 VisDrone→YOLO 脚本，**保留严重遮挡、运动模糊等困难样本** |
| 权重路径 | `runs/BS/m/train18/weights/best.pt` |

### 原作品记录的成绩

YOLOv12m 训练 200 轮后的独立验证（548 张 / 17040 实例，car 14064、van 1975、truck 750、bus 251）：

| 指标 | 数值 |
|---|---|
| Precision | 0.68 |
| Recall | 0.52 |
| mAP50 | 0.55 |
| mAP50-95 | 0.39 |

分类别 mAP50：car 0.801 / van 0.489 / truck 0.390 / bus 0.603（YOLOv10m 同数据集对照值）。

**注意口径差异**：随包五组 YOLOv10 对比实验的 `results.csv` 记录了各规模「最优一轮」，
YOLOv10m 为 P .68772 / R .52038 / mAP50 .56834 / mAP50-95 .40081（epoch 135）。
YOLOv12m 的对比表值 .56834/.40081 与设计文档表 4-2 末轮的 .55/.39 **并不一致**——
这正是材料核对时标记的问题。**不要挑最高的那个数字当新系统成绩。**

---

## 三、本机现状（已具备，不用重装）

| 项 | 状态 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop，8GB 显存，驱动 582.05 |
| Python | 3.12.10（`.venv`） |
| PyTorch | 2.8.0+cu128，`torch.cuda.is_available() = True` |
| Ultralytics | 8.4.37（含 `cfg/models/12/yolo12.yaml`、`cfg/datasets/VisDrone.yaml`） |
| 现有权重 | `云瞰街巷_技术优化版/models/best.pt`，SHA-256 `975aefac...a27c2a384` |

训练本身不需要 PySide6，用 `.venv` 里的 `yolo.exe` 即可。

**显存提示**：4090 24GB 用的是 batch=8，8GB 卡上 batch=8 跑 640 可能吃紧。
先跑 `低显存训练.cmd`（batch=4）验证流程跑通，再决定是否上 batch=8。
200 轮在 5060 Laptop 上会明显比 24 小时长，建议先用 `--epochs 30` 试跑测速。

---

## 四、快速开始

```powershell
cd C:\Users\96901\Desktop\城垣杯加yolo\训练

# 1) 准备数据集（下载约 2.3GB，会自动转成四类）
..\云瞰街巷_技术优化版\.venv\Scripts\python.exe 准备数据集.py

# 2) 训练
低显存训练.cmd        # 8GB 显卡推荐先跑这个

# 3) 独立验证，核对指标
验证.cmd

# 4) 确认无误后接入程序（自动备份旧权重 + 更新 SHA-256）
..\云瞰街巷_技术优化版\.venv\Scripts\python.exe 接入新权重.py runs\visdrone4_yolo12m_lowvram\weights\best.pt
```

`准备数据集.py` 会同时保留 `datasets/VisDrone`（10 类原始标注）和
`datasets/VisDrone4`（四类，图片走硬链接不占额外磁盘），
**后面做路线 B 加类别时直接复用 10 类那份，不用重新下载。**

---

## 五、路线 B：加行人/自行车/电动车

这是结合方案里的「第三阶段」，也是真正打开「人车交织」研究的前提。
现有四类权重不识别行人和非机动车。

### B1. 从 VisDrone 10 类直接训多类模型

最省事。Ultralytics 内置的 `VisDrone.yaml` 已经是 10 类（pedestrian / people /
bicycle / car / van / truck / tricycle / awning-tricycle / bus / motor），
直接：

```powershell
..\云瞰街巷_技术优化版\.venv\Scripts\yolo.exe detect train ^
  model=yolo12m.yaml data=VisDrone.yaml epochs=200 batch=8 imgsz=640 ^
  device=0 optimizer=SGD lr0=0.01 close_mosaic=10 amp=True workers=8 ^
  project=runs name=visdrone10_yolo12m
```

代价：模型从零学 10 类，比四类难一些，但换来行人/自行车直接可用。
**注意**：这会改变 `best.pt` 的类别体系，程序里「车辆数量」的口径会从
「四类机动车」变成「10 类全目标」，界面分类计数和材料表述都要同步改。

### B2. 在现有四类模型上增量训练（迁移学习）

保留原权重的车辆知识，只新学行人类别：

```powershell
..\云瞰街巷_技术优化版\.venv\Scripts\yolo.exe detect train ^
  model=..\云瞰街巷_技术优化版\models\best.pt data=我的10类.yaml ^
  epochs=100 imgsz=640 batch=4 device=0 optimizer=AdamW lr0=0.001 ^
  freeze=10 project=runs name=finetune_add_ped
```

要点：
- `freeze=10` 冻结骨干前 10 层，先只训检测头，避免新类的随机梯度把旧知识冲掉。
- 新增类别的 head 是随机初始化的，需要比从零训练更多的 epoch。
- **必须用固定验证集复测四类的 P/R 和 mAP**，增量训练后原类别性能常常下降，
  不能只看新类涨了多少。

### B3. 用自家场地视频做微调（最贴合课题）

1. 采集：西号巷等临街空间，固定机位，覆盖工作日/周末、多个时段。
2. 抽帧：每 1–3 秒抽 1 帧。相邻帧高度重复，全量标注浪费。
3. 切图：用 `城垣杯\AI训练用图片处理` 这套现成工具——
   - 第一步「图片高清化」= Real-ESRGAN x4 放大（工具已在 `城垣杯\AI训练用图片处理\第一步图片高清化`）
   - 第二步「图片裁剪」= 切成训练块
   - 切图注意：**车辆跨块要处理**，否则一半车身会被当成新目标
4. 标注：X-AnyLabeling / labelme / CVAT 都行，类别至少
   `car / van / truck / bus / pedestrian / bicycle / motorcycle`
5. 划分：按**场地/时段**划分 train/val，而不是随机划分同一批视频的帧，
   否则验证集泄漏，成绩虚高。
6. 训练：用 B2 的命令，数据集换成自家 yaml。

---

## 六、验证纪律（这几条是红线）

材料核对时已经标出来的坑，训练时务必守住：

1. **训练集和验证集必须分开**，验证集要独立、且不参与任何调参选择。
2. **不要只挑最好的一轮**。`results.csv` 里取 max 是可以的，但报告里要写明
   「各轮次中的最优值」还是「最后一轮」，并保持全材料一致。
3. **必须报分类别结果**。truck 的 mAP50 只有 0.390，明显低于 car 的 0.801，
   只报总体 mAP 会掩盖短板。
4. **不写「无错检漏检」「彻底解决 ID 切换」「恶劣天气稳定」**。
   没有人工真值对照、没有 IDF1、没有分组天气测试，这些结论都不成立。
   替换写法见 `学姐问题回复与材料核对说明.md` 第 3 节。
5. **软件运行测试 ≠ 检测精度评测**。程序跑通、GPU 测速、48 项回归测试，
   都不能替代 mAP / 漏检率 / IDF1 的评测。
6. **换权重要同步改 SHA-256**。`接入新权重.py` 会自动更新 `SOURCE_PROVENANCE.md`；
   程序每次分析也会把权重 SHA-256 写进 `summary.json`，所以「沿用原权重」还是
   「重新训练」能被机器核对，不靠口头声明。
7. **更新后重跑回归测试**：`.venv\Scripts\python.exe -m pytest -q`（48 项）。

---

## 七、常见问题

**Q：`yolo12m.yaml` 还是 `yolov12m.yaml`？**
A：Ultralytics 8.4.37 里是 `yolo12m.yaml`（配置在 `cfg/models/12/yolo12.yaml`）。
`yolov12m.yaml` 会报 FileNotFoundError。原作品用的是 YOLOv12 官方仓库
（sunsmarterjie/yolov12）的配置，署名时按实际使用情况标注即可。

**Q：数据集下载慢或失败？**
A：内置脚本从 ultralytics 官方资产地址拉取，国内可能较慢。
可手动从 VisDrone 官方 GitHub 下载 `VisDrone2019-DET-train.zip` 和
`VisDrone2019-DET-val.zip`，解压到 `datasets/VisDrone/` 下的同名目录，再重跑脚本。

**Q：训练要跑多久？**
A：原作品 4090 上约 24 小时。5060 Laptop 8GB 性能明显更低，
batch=4、200 轮预计在 12–30 小时区间，建议先用 30 轮试跑测速再决定。

**Q：训练完还需要改程序吗？**
A：只要类别体系不变（还是 car/van/truck/bus 四类），**不需要改代码**——
程序从权重里读类别名。改的只是 `models\best.pt` 这一个文件。
若改成 10 类，则需核对界面分类计数的表述、材料里的「四类」措辞。

**Q：能不能在城垣杯的场地照片上训练？**
A：那些是空间尺度调查照片，不是交通视频，框不出机动车实例。
训练数据还得靠 VisDrone 或自家交通视频。
