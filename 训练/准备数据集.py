"""准备 VisDrone2019 四类机动车训练集（car / van / truck / bus）。

原作品（云瞰车流 智绘通途）使用的就是 VisDrone2019-DET 数据集 + VisDrone-4.yaml
四类配置。本脚本完成三件事：

1. 调用 Ultralytics 内置的 VisDrone 下载与格式转换脚本（约 2.3GB，10 类）。
2. 从 10 类标注中筛出 car/van/truck/bus 四类并重映射为 0-3，
   同时保留全部困难样本（截断、遮挡），与原报告的转换口径一致。
3. 用硬链接（几乎不占额外磁盘）建立 VisDrone4 数据集根目录，并生成 VisDrone-4.yaml。

用法（在本目录下）：
    ..\\云瞰街巷_技术优化版\\.venv\\Scripts\\python.exe 准备数据集.py

说明：数据集本体约 2.3GB，需要联网。若下载失败，可手动从
https://github.com/VisDrone/VisDrone-Dataset 下载 VisDrone2019-DET-train.zip 与
VisDrone2019-DET-val.zip，解压到 datasets/VisDrone/ 下同名目录后重跑本脚本。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
VISDRONE_ROOT = DATASETS / "VisDrone"
VISDRONE4_ROOT = DATASETS / "VisDrone4"

# 需要保留的四个机动车类别，其余（行人、骑行者、自行车、三轮车、摩托等）不参与本模型训练
KEEP_CLASSES = ("car", "van", "truck", "bus")


def _read_builtin_names() -> list[str]:
    """从 Ultralytics 内置 VisDrone.yaml 读取类别顺序，避免写死类别号。"""
    import yaml
    import ultralytics

    cfg = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "VisDrone.yaml"
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    return [str(n) for n in names]


def _download_and_convert() -> None:
    """触发 Ultralytics 内置的 VisDrone 下载 + 标注转换脚本。"""
    from ultralytics.data.utils import check_det_dataset

    print("[1/4] 下载并转换 VisDrone2019（首次约 2.3GB，请耐心等待）…")
    info = check_det_dataset("VisDrone.yaml", autodownload=True)
    print("      数据集位置：", info.get("path") or info.get("train"))


def _convert_10_to_4(names: list[str]) -> dict[str, int]:
    """建立 10 类 -> 4 类的重映射表。"""
    mapping: dict[str, int] = {}
    for wanted in KEEP_CLASSES:
        if wanted not in names:
            raise SystemExit(f"内置 VisDrone 类别里找不到 {wanted}，实际类别：{names}")
        mapping[names.index(wanted)] = len(mapping)
    return mapping


def _filter_labels(src_labels: Path, dst_labels: Path, remap: dict[int, int]) -> tuple[int, int]:
    """按类别过滤并重写标签。返回(保留目标数, 处理文件数)。"""
    kept_objects = 0
    files = 0
    dst_labels.mkdir(parents=True, exist_ok=True)
    for txt in sorted(src_labels.glob("*.txt")):
        out_lines: list[str] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls in remap:
                out_lines.append(f"{remap[cls]} " + " ".join(parts[1:5]))
        # 空标签文件保留：它代表画面中确实没有机动车的负样本
        (dst_labels / txt.name).write_text(
            ("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8"
        )
        kept_objects += len(out_lines)
        files += 1
    return kept_objects, files


def _link_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接，不支持时退回复制。"""
    try:
        os.link(src, dst)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


def _build_root(names: list[str]) -> None:
    remap = _convert_10_to_4(names)
    print("      类别重映射（10 类 -> 4 类）：", {names[k]: v for k, v in remap.items()})

    for split in ("train", "val"):
        src_images = VISDRONE_ROOT / "images" / split
        src_labels = VISDRONE_ROOT / "labels" / split
        if not src_images.is_dir() or not src_labels.is_dir():
            raise SystemExit(f"缺少 {src_images} 或 {src_labels}，请先完成下载与转换。")

        dst_images = VISDRONE4_ROOT / "images" / split
        dst_labels = VISDRONE4_ROOT / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)

        print(f"[2/4] 处理 {split}：建立硬链接图片…")
        for img in sorted(src_images.glob("*.jpg")):
            target = dst_images / img.name
            if not target.exists():
                _link_or_copy(img, target)

        print(f"[3/4] 处理 {split}：筛选四类标注…")
        kept, files = _filter_labels(src_labels, dst_labels, remap)
        print(f"      {files} 个标注文件，保留目标 {kept} 个")


def _write_yaml() -> None:
    yaml_path = HERE / "VisDrone-4.yaml"
    yaml_path.write_text(
        "# 四类机动车训练集，与原作品 VisDrone-4.yaml 口径一致\n"
        f"path: {VISDRONE4_ROOT.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: car\n"
        "  1: van\n"
        "  2: truck\n"
        "  3: bus\n",
        encoding="utf-8",
    )
    print("[4/4] 已生成", yaml_path)


def main() -> int:
    DATASETS.mkdir(parents=True, exist_ok=True)
    names = _read_builtin_names()
    if not (VISDRONE_ROOT / "labels" / "train").is_dir():
        _download_and_convert()
    _build_root(names)
    _write_yaml()
    print("\n完成。下一步运行 训练.cmd（或 低显存训练.cmd）。")
    print("注意：datasets/VisDrone 下的 10 类标注已保留，后续加行人/非机动车类别时直接复用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
