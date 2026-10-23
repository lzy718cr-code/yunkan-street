"""One-command post-processing for a recorded session.

Rebuilds the space-use portrait from ``detections.csv``: trajectories, presence
heat, time curve, indicators and a short Markdown summary. Calibration is taken
from the session's own ``calibration.json`` when present, so the same command
produces metres or frame fractions without extra arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_app.calibration import Calibration
from traffic_app.portrait import build_portrait, read_image, write_image


def main() -> int:
    parser = argparse.ArgumentParser(description="会话后处理：轨迹、热力图、指标与摘要")
    parser.add_argument("session", type=Path)
    parser.add_argument("--video", type=Path, default=None,
                        help="会话没有 poster.jpg 时，从原视频取一帧作为底图")
    parser.add_argument("--frame", type=int, default=None, help="取第几帧作为底图")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--classes", default="person,bicycle,motorcycle")
    parser.add_argument("--cell", type=float, default=None)
    parser.add_argument("--mask", type=Path, default=None,
                        help="地面多边形 JSON：[[x,y], ...]，画面比例坐标")
    parser.add_argument("--playback", type=float, default=0.0,
                        help="额外生成多少秒的轨迹回放视频，0 表示不生成")
    args = parser.parse_args()

    session = args.session.resolve()
    output = (args.out or (session / "space")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    calibration = None
    calibration_path = session / "calibration.json"
    if calibration_path.is_file():
        calibration = Calibration.load(calibration_path)
        print(f"使用标定：{calibration_path}")

    background = None
    if not (session / "poster.jpg").is_file() and args.video is not None:
        import cv2
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            print(f"无法打开视频：{args.video}", file=sys.stderr)
            return 1
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = args.frame if args.frame is not None else max(1, total // 2)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            print("无法读取该帧", file=sys.stderr)
            return 1
        background = output / "poster.jpg"
        write_image(background, frame, 90)
        print(f"已从视频第 {index} 帧生成底图：{background}")

    mask = None
    if args.mask is not None:
        mask = json.loads(args.mask.read_text(encoding="utf-8"))

    classes = [name.strip() for name in args.classes.split(",") if name.strip()] or None
    portrait = build_portrait(
        session, output, calibration=calibration, classes=classes,
        background=background, mask_polygon=mask, cell_size=args.cell,
        playback_seconds=args.playback)

    for warning in portrait.warnings:
        print(f"提示：{warning}")
    keys = ("coordinate_unit", "tracks_raw", "tracks_after_stitch", "person_seconds",
            "mean_dwell_seconds", "stationary_ratio", "occupied_cells")
    print(json.dumps({key: portrait.metrics[key] for key in keys}, ensure_ascii=False, indent=2))
    print(json.dumps(portrait.metrics["near_miss"], ensure_ascii=False, indent=2))
    print(f"输出目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
