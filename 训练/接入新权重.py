"""把新训练出的权重接入云瞰街巷程序，并同步更新来源说明中的 SHA-256。

程序每次分析都会把所用权重的 SHA-256 写进 summary.json，所以换权重后材料里
"沿用原权重"还是"重新训练"可以被机器核对，不靠口头声明。

用法（在本目录下）：
    ..\\云瞰街巷_技术优化版\\.venv\\Scripts\\python.exe 接入新权重.py runs\\visdrone4_yolo12m\\weights\\best.pt

也可以用 --dry-run 只看会改哪些文件，不实际写入。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "云瞰街巷_技术优化版"
TARGET_WEIGHT = PROJECT / "models" / "best.pt"
PROVENANCE = PROJECT / "SOURCE_PROVENANCE.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="接入新训练权重")
    parser.add_argument("weight", type=Path, help="新权重的 best.pt 路径")
    parser.add_argument("--dry-run", action="store_true", help="只检查不写入")
    parser.add_argument("--note", default="本机重新训练", help="来源说明里新增的备注")
    args = parser.parse_args()

    source = args.weight.resolve()
    if not source.is_file():
        raise SystemExit(f"找不到权重文件：{source}")
    if source.suffix != ".pt":
        raise SystemExit("权重应为 .pt 文件")

    new_sha = sha256(source)
    old_sha = sha256(TARGET_WEIGHT) if TARGET_WEIGHT.is_file() else "(无)"

    print("新权重：", source)
    print("新 SHA-256：", new_sha)
    print("原权重：", TARGET_WEIGHT)
    print("原 SHA-256：", old_sha)
    if new_sha == old_sha:
        print("\n与当前权重完全相同，无需操作。")
        return 0

    if args.dry_run:
        print("\n[dry-run] 将执行：")
        print(f"  1. 备份 {TARGET_WEIGHT.name} -> models/best.pt.bak")
        print(f"  2. 复制新权重到 {TARGET_WEIGHT}")
        print(f"  3. 更新 {PROVENANCE.name} 中的 SHA-256 与说明")
        return 0

    if TARGET_WEIGHT.is_file():
        backup = TARGET_WEIGHT.with_suffix(".pt.bak")
        shutil.copy2(TARGET_WEIGHT, backup)
        print("已备份原权重：", backup)

    shutil.copy2(source, TARGET_WEIGHT)
    print("已写入：", TARGET_WEIGHT)

    if PROVENANCE.is_file():
        text = PROVENANCE.read_text(encoding="utf-8")
        if old_sha != "(无)" and old_sha in text:
            text = text.replace(old_sha, new_sha)
        stamp = f"\n## 权重更新记录\n\n- {args.note}：替换为 `{source.name}`，SHA-256 `{new_sha}`。\n"
        text = text.rstrip() + "\n" + stamp
        PROVENANCE.write_text(text, encoding="utf-8")
        print("已更新来源说明：", PROVENANCE)
    else:
        print("警告：找不到", PROVENANCE)

    print("\n下一步：")
    print("  1. 运行 验证.cmd，核对 mAP 与分类别表现")
    print("  2. 重跑项目自带的回归测试：.venv\\Scripts\\python.exe -m pytest -q")
    print("  3. 重新打包前核对 构建信息.json 中的权重 SHA-256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
