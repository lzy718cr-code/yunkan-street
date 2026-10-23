"""Render a heat map of observed presence onto a frame of the source video.

Each track contributes its visible duration, so the map reads as person-seconds
per pixel rather than raw detections. It is a visual summary, not a density in
people per square metre; the metric version needs a ground calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_app.space import load_tracks


def main() -> int:
    parser = argparse.ArgumentParser(description="把轨迹停留热力图叠加到原视频帧")
    parser.add_argument("session", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--classes", default="person,bicycle,motorcycle")
    parser.add_argument("--sigma", type=float, default=0.0, help="高斯核，0 表示按画面宽度自动")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    classes = [name.strip() for name in args.classes.split(",") if name.strip()] or None
    tracks, unit, _stats = load_tracks(args.session, None, classes)
    tracks = [track for track in tracks if track.duration >= args.min_duration] or tracks
    if not tracks:
        print("没有可用轨迹", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print("无法打开视频", file=sys.stderr)
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
    scale = 0.5
    grid = np.zeros((int(height * scale), int(width * scale)), dtype=np.float32)
    for track in tracks:
        points = track.points
        for i, point in enumerate(points):
            if i == 0:
                weight = 0.25
            else:
                weight = min(1.0, max(0.02, point.timestamp - points[i - 1].timestamp))
            x = int(point.x * width * scale)
            y = int(point.y * height * scale)
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                grid[y, x] += weight

    sigma = args.sigma if args.sigma > 0 else max(6.0, width * 0.018)
    grid = cv2.GaussianBlur(grid, (0, 0), sigma * scale)
    peak = float(grid.max())
    if peak <= 0:
        print("热力图没有累计值", file=sys.stderr)
        return 1
    normalized = np.clip(grid / peak, 0.0, 1.0)
    heat = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(frame, 1.0 - args.alpha, heat, args.alpha, 0)
    # Keep the scene readable where nothing happened.
    mask = cv2.resize(np.uint8(normalized > args.threshold) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    blended = np.where(mask[..., None] > 0, blended, frame)

    bar_width, bar_height = 180, 12
    bar_x, bar_y = 24, height - 44
    gradient = np.linspace(0, 1, bar_width, dtype=np.float32)[None, :]
    bar = cv2.applyColorMap(np.uint8(gradient * 255), cv2.COLORMAP_TURBO)
    bar = cv2.resize(bar, (bar_width, bar_height), interpolation=cv2.INTER_NEAREST)
    blended[bar_y:bar_y + bar_height, bar_x:bar_x + bar_width] = bar
    cv2.rectangle(blended, (bar_x - 1, bar_y - 1),
                  (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
    cv2.putText(blended, "low", (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, "high", (bar_x + bar_width - 34, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, f"presence heat (person-seconds), tracks={len(tracks)}", (bar_x, bar_y + bar_height + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    destination = args.out or (args.session / "heatmap.jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(destination))
    print(f"tracks={len(tracks)} frame={index} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


