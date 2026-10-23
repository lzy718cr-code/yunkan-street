"""Command line entry point for offline space analysis of a recorded session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_app.calibration import Calibration, CalibrationError
from traffic_app.space import analyse_session


def main() -> int:
    parser = argparse.ArgumentParser(description="把一次会话的逐帧检测结果转成平面轨迹与空间使用指标")
    parser.add_argument("session", type=Path, help="包含 frames.csv / detections.csv 的会话目录")
    parser.add_argument("--out", type=Path, default=None, help="输出目录，默认 <session>/space")
    parser.add_argument("--calibration", type=Path, default=None, help="四点标定 JSON")
    parser.add_argument("--classes", default="person", help="参与分析的类别，逗号分隔；留空表示全部")
    parser.add_argument("--cell", type=float, default=0.5, help="占用网格边长（米或画面比例）")
    parser.add_argument("--gap", type=float, default=2.0, help="断轨拼接允许的最大时间间隔（秒）")
    parser.add_argument("--distance", type=float, default=None, help="断轨拼接允许的最大空间距离（默认 1 米；未标定时为画面宽度的 2%%）")
    parser.add_argument("--background", type=Path, default=None, help="可选底图")
    args = parser.parse_args()

    calibration = None
    if args.calibration:
        try:
            calibration = Calibration.load(args.calibration)
        except (OSError, KeyError, CalibrationError) as exc:
            print(f"标定文件不可用：{exc}", file=sys.stderr)
            return 2
    classes = [name.strip() for name in args.classes.split(",") if name.strip()] or None
    out_dir = args.out or (args.session / "space")
    try:
        metrics = analyse_session(
            args.session, out_dir, calibration,
            class_names=classes, cell_size=args.cell,
            gap_seconds=args.gap, stitch_distance=args.distance,
            background=args.background,
        )
    except (OSError, ValueError) as exc:
        print(f"分析失败：{exc}", file=sys.stderr)
        return 1
    brief = {key: metrics[key] for key in (
        "coordinate_unit", "tracks_raw", "tracks_after_stitch", "merged_fragments",
        "person_seconds", "mean_dwell_seconds", "stationary_ratio",
        "occupied_cells", "calibration_reprojection_rmse",
    )}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"输出目录：{out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
