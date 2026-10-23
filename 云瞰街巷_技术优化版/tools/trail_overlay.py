"""Draw observed person trajectories back onto a frame of the source video.

A plan-view scatter is hard to read for a non-specialist; the same tracks drawn
on the original frame look like a long exposure and are easier to explain in a
report or a slide. Colours run from blue (early) to red (late); dots mark where
a track starts and ends.

Only used with sessions recorded without a ground calibration, because the
stored coordinates are frame fractions. Calibrated sessions should use the
plan view instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_app.space import load_tracks


def track_colour(fraction: float) -> tuple[int, int, int]:
    hue = int(round(120 * (1.0 - min(max(fraction, 0.0), 1.0))))
    patch = np.uint8([[[hue, 235, 255]]])
    blue, green, red = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(blue), int(green), int(red)


def main() -> int:
    parser = argparse.ArgumentParser(description="把轨迹叠加回原视频帧")
    parser.add_argument("session", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=None, help="默认取视频中间帧")
    parser.add_argument("--window", type=float, default=12.0,
                        help="只画该帧之前多少秒内仍在出现的轨迹；0 表示全部")
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--min-duration", type=float, default=0.0,
                        help="只画持续时间不短于该值的轨迹（秒）")
    parser.add_argument("--smooth", type=int, default=0,
                        help="滑动平均窗口，0 表示不平滑")
    parser.add_argument("--thickness", type=int, default=2)
    args = parser.parse_args()

    tracks, unit, _stats = load_tracks(args.session, None, ["person"])
    if unit != "frame_fraction":
        print("该会话带标定，请使用平面图输出", file=sys.stderr)
        return 2
    if not tracks:
        print("没有可用轨迹", file=sys.stderr)
        return 1

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

    height, width = frame.shape[:2]
    frame_time = None
    frames_csv = args.session / "frames.csv"
    if frames_csv.is_file():
        import csv
        with frames_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["frame_index"]) == index:
                    frame_time = float(row["source_time_seconds"])
                    break
    if frame_time is None:
        frame_time = index / 25.0

    starts = [track.start for track in tracks]
    t0, t1 = min(starts), max(starts)
    overlay = frame.copy()
    drawn = 0
    scale = max(1.0, width / 1280.0)
    for track in tracks:
        if len(track.points) < args.min_points:
            continue
        if track.duration < args.min_duration:
            continue
        if args.window > 0 and track.end < frame_time - args.window:
            continue
        if track.start > frame_time:
            continue
        points = np.array([[int(p.x * width), int(p.y * height)] for p in track.points], np.int32)
        if args.smooth >= 3 and len(points) >= args.smooth:
            kernel = np.ones(args.smooth, dtype=float) / args.smooth
            points = np.stack([
                np.convolve(points[:, 0], kernel, mode="valid"),
                np.convolve(points[:, 1], kernel, mode="valid"),
            ], axis=1).astype(np.int32)
        colour = track_colour((track.start - t0) / max(t1 - t0, 1e-6))
        cv2.polylines(overlay, [points], False, colour,
                      max(1, int(args.thickness * scale)), cv2.LINE_AA)
        cv2.circle(overlay, tuple(points[0]), int(3 * scale), (80, 220, 80), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(points[-1]), int(3 * scale), (60, 60, 240), -1, cv2.LINE_AA)
        drawn += 1

    blended = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
    label = f"tracks drawn: {drawn}   blue=early  red=late   green dot=start  red dot=end"
    cv2.rectangle(blended, (12, height - 40), (12 + 620, height - 12), (0, 0, 0), -1)
    cv2.putText(blended, label, (22, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)

    destination = args.out or (args.session / "trail_overlay.jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(destination))
    print(f"drawn={drawn} frame={index} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
