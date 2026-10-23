"""Turn a recorded session into metric trajectories and space-use indicators.

The recorder already stores one row per detection per frame. This module is
deliberately offline: it reads those rows, optionally applies a surveyed
homography, stitches fragmented track IDs, and writes tracks, indicators and a
plan-view figure. Keeping it separate from the inference thread means a bad
calibration can never corrupt a live session.

Units
-----
With a calibration every coordinate is metres on the surveyed plane. Without
one, coordinates are normalised frame fractions in [0, 1] and distances are
reported as ``frame`` units; the summary records which mode was used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .calibration import Calibration


@dataclass
class TrackPoint:
    timestamp: float
    frame_index: int
    x: float
    y: float


@dataclass
class Track:
    track_id: int
    class_name: str
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.points[0].timestamp

    @property
    def end(self) -> float:
        return self.points[-1].timestamp

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _read_frame_size(session_dir: Path) -> tuple[int, int]:
    with (session_dir / "frames.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        raise ValueError("frames.csv 为空，无法确定画面尺寸")
    return int(float(row["width"])), int(float(row["height"]))


def load_tracks(
    session_dir: str | Path,
    calibration: Calibration | None = None,
    class_names: Iterable[str] | None = None,
) -> tuple[list[Track], str, dict]:
    """Read ``detections.csv`` and return tracks in metres or frame fractions.

    With a calibration, observations outside the surveyed control polygon are
    dropped: the homography still produces numbers there, but no control point
    verifies them, so including them would silently inflate the area covered.
    """
    directory = Path(session_dir)
    detections_path = directory / "detections.csv"
    if not detections_path.is_file():
        raise FileNotFoundError(f"找不到逐帧检测文件：{detections_path}")
    width, height = _read_frame_size(directory)
    wanted = None if class_names is None else {name.strip() for name in class_names}

    best: dict[int, dict[int, tuple[float, float, float, str]]] = defaultdict(dict)
    considered = 0
    outside = 0
    with detections_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            track_text = (row.get("track_id") or "").strip()
            if not track_text:
                continue
            name = (row.get("class_name") or "").strip()
            if wanted is not None and name not in wanted:
                continue
            track_id = int(float(track_text))
            frame = int(float(row["frame_index"]))
            timestamp = float(row["source_time_seconds"])
            confidence = float(row["confidence"])
            x = float(row["bottom_center_x_normalized"]) * width
            y = float(row["bottom_center_y_normalized"]) * height
            considered += 1
            if calibration is not None and not calibration.contains_image(x, y):
                outside += 1
                continue
            current = best[track_id].get(frame)
            if current is None or confidence > current[0]:
                best[track_id][frame] = (confidence, timestamp, x, y, name)

    tracks: list[Track] = []
    for track_id, frames in best.items():
        ordered = sorted(frames.items(), key=lambda item: item[1][1])
        track = Track(track_id=track_id, class_name=ordered[0][1][4])
        for _frame, (_confidence, timestamp, x, y, _name) in ordered:
            if calibration is not None:
                x, y = calibration.to_world(x, y)
            else:
                x, y = x / width, y / height
            track.points.append(TrackPoint(timestamp=timestamp,
                                           frame_index=_frame,
                                           x=x, y=y))
        if len(track.points) >= 2:
            tracks.append(track)
    tracks.sort(key=lambda track: (track.start, track.track_id))
    coordinate_unit = "metre" if calibration is not None else "frame_fraction"
    stats = {
        "detections_considered": considered,
        "detections_outside_calibration": outside,
    }
    return tracks, coordinate_unit, stats


def stitch_tracks(
    tracks: list[Track],
    max_gap_seconds: float = 2.0,
    max_distance: float = 1.0,
) -> tuple[list[Track], dict[int, int]]:
    """Join track fragments that end and restart close together in time/space.

    ByteTrack breaks a person into a new ID after a short occlusion. On the
    ground plane the two fragments are usually adjacent, so a greedy join can
    repair part of the error without retraining the tracker. The mapping from
    every original ID to its merged ID is returned for audit.
    """
    ordered = sorted(tracks, key=lambda track: track.start)
    merged: list[Track] = []
    owners: list[dict[int, int]] = []
    mapping: dict[int, int] = {}
    for track in ordered:
        candidates = []
        for index, host in enumerate(merged):
            gap = track.start - host.end
            if gap < 0 or gap > max_gap_seconds:
                continue
            distance = math.hypot(track.points[0].x - host.points[-1].x,
                                  track.points[0].y - host.points[-1].y)
            if distance <= max_distance:
                candidates.append((distance, gap, index))
        if not candidates:
            merged.append(Track(track_id=track.track_id, class_name=track.class_name,
                                points=list(track.points)))
            owners.append({track.track_id: track.track_id})
            mapping[track.track_id] = track.track_id
            continue
        _distance, _gap, index = min(candidates)
        host = merged[index]
        host.points.extend(track.points)
        host.points.sort(key=lambda point: point.timestamp)
        owners[index][track.track_id] = host.track_id
        mapping[track.track_id] = host.track_id
    return merged, mapping


def _grid_occupancy(tracks: list[Track], cell: float) -> dict[tuple[int, int], float]:
    """Seconds spent in each ground cell, distributed along each segment."""
    cells: dict[tuple[int, int], float] = defaultdict(float)
    for track in tracks:
        for previous, current in zip(track.points, track.points[1:]):
            dt = current.timestamp - previous.timestamp
            if dt <= 0 or dt > 10:
                continue
            steps = max(1, min(20, int(dt * 4)))
            for step in range(steps):
                fraction = (step + 0.5) / steps
                x = previous.x + (current.x - previous.x) * fraction
                y = previous.y + (current.y - previous.y) * fraction
                cells[(math.floor(x / cell), math.floor(y / cell))] += dt / steps
    return cells


def summarise(
    tracks: list[Track],
    coordinate_unit: str,
    *,
    cell_size: float,
    bin_seconds: float = 60.0,
    gap_seconds: float = 2.0,
    stitch_distance: float = 1.0,
    merged: list[Track] | None = None,
    mapping: dict[int, int] | None = None,
) -> dict:
    raw_ids = len(tracks)
    final = merged if merged is not None else tracks
    person_seconds = sum(track.duration for track in final)
    durations = [track.duration for track in final]
    paths = [sum(math.hypot(b.x - a.x, b.y - a.y)
                 for a, b in zip(track.points, track.points[1:])) for track in final]
    stationary = 0
    moving = 0
    speed_threshold = 0.2 if coordinate_unit == 'metre' else 0.02
    for track in final:
        for previous, current in zip(track.points, track.points[1:]):
            dt = current.timestamp - previous.timestamp
            if dt <= 0:
                continue
            speed = math.hypot(current.x - previous.x, current.y - previous.y) / dt
            if speed < speed_threshold:
                stationary += 1
            else:
                moving += 1
    bins: dict[int, dict[str, float]] = {}
    for track in final:
        index = int(track.start // bin_seconds)
        entry = bins.setdefault(index, {"present": 0.0, "unique": 0.0})
        entry["present"] += track.duration
    for track in final:
        for index in range(int(track.start // bin_seconds),
                           int(track.end // bin_seconds) + 1):
            bins.setdefault(index, {"present": 0.0, "unique": 0.0})["unique"] += 1
    occupancy = _grid_occupancy(final, cell_size)
    return {
        "coordinate_unit": coordinate_unit,
        "cell_size": cell_size,
        "tracks_raw": raw_ids,
        "tracks_after_stitch": len(final),
        "merged_fragments": raw_ids - len(final),
        "person_seconds": round(person_seconds, 2),
        "mean_dwell_seconds": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "median_dwell_seconds": round(float(np.median(durations)), 2) if durations else 0.0,
        "max_dwell_seconds": round(max(durations), 2) if durations else 0.0,
        "mean_path_length": round(sum(paths) / len(paths), 3) if paths else 0.0,
        "stationary_segments": stationary,
        "moving_segments": moving,
        "stationary_ratio": round(stationary / max(1, stationary + moving), 3),
        "occupied_cells": len(occupancy),
        "top_cells": [
            {"cell": [int(key[0]), int(key[1])], "seconds": round(value, 1)}
            for key, value in sorted(occupancy.items(), key=lambda item: -item[1])[:20]
        ],
        "time_bins": [
            {"start_seconds": round(index * bin_seconds, 1),
             "person_seconds": round(value["present"], 1),
             "track_count": int(value["unique"])}
            for index, value in sorted(bins.items())
        ],
        "stitch": {
            "gap_seconds": gap_seconds,
            "distance": stitch_distance,
            "id_map": mapping or {},
        },
        "notes": [
            "驻留时长是本会话内连续可见时长，不等于真实停留意愿。",
            "拼接只修复相邻断轨，不保证消除全部 ID 切换。",
        ],
    }


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(name.lower() in font.name.lower() for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_plan_view(
    tracks: list[Track],
    out_path: str | Path,
    *,
    title: str = "平面轨迹",
    unit: str = "m",
    background: str | Path | None = None,
) -> Path:
    """Draw all tracks on the ground plane; each dot is one observed second."""
    plt = _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(7.2, 6.4), dpi=180)
    if background:
        image = plt.imread(str(background))
        axis.imshow(image)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(tracks))))
    for color, track in zip(colors, tracks):
        if len(track.points) < 5 or track.duration < 1.0:
            continue
        xs = [point.x for point in track.points]
        ys = [point.y for point in track.points]
        axis.plot(xs, ys, color=color, linewidth=1.4, alpha=0.85)
        axis.scatter(xs[0], ys[0], color=color, marker="o", s=26, zorder=3)
        axis.scatter(xs[-1], ys[-1], color=color, marker="s", s=22, zorder=3)
        seconds = track.duration
        if seconds >= 5:
            index = int(np.argmax([point.timestamp for point in track.points]))
            axis.annotate(f"{track.track_id}·{seconds:.0f}s",
                          (xs[index], ys[index]), fontsize=7, color=color,
                          xytext=(3, 3), textcoords="offset points")
    axis.set_title(title)
    axis.set_xlabel(f"x / {unit}")
    axis.set_ylabel(f"y / {unit}")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.invert_yaxis()
    figure.tight_layout()
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def plot_occupancy(
    tracks: list[Track],
    out_path: str | Path,
    *,
    cell_size: float,
    title: str = "停留密度",
    unit: str = "m",
) -> Path:
    plt = _setup_matplotlib()
    cells = _grid_occupancy(tracks, cell_size)
    if not cells:
        raise ValueError("没有可绘制的占用格")
    xs = [key[0] for key in cells]
    ys = [key[1] for key in cells]
    values = np.array([cells[key] for key in cells])
    figure, axis = plt.subplots(figsize=(7.2, 6.4), dpi=180)
    scatter = axis.scatter(np.array(xs) * cell_size, np.array(ys) * cell_size,
                           c=values, s=90, marker="s", cmap="YlOrRd")
    figure.colorbar(scatter, ax=axis, label="累计秒数")
    axis.set_title(title)
    axis.set_xlabel(f"x / {unit}")
    axis.set_ylabel(f"y / {unit}")
    axis.set_aspect("equal", adjustable="datalim")
    axis.invert_yaxis()
    figure.tight_layout()
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)
    return destination


def write_tracks_csv(tracks: list[Track], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "timestamp", "frame_index", "x", "y"])
        for track in tracks:
            for point in track.points:
                writer.writerow([track.track_id, f"{point.timestamp:.3f}",
                                 point.frame_index, f"{point.x:.4f}", f"{point.y:.4f}"])
    return destination


def analyse_session(
    session_dir: str | Path,
    out_dir: str | Path,
    calibration: Calibration | None = None,
    *,
    class_names: Iterable[str] | None = ("person",),
    cell_size: float = 0.5,
    gap_seconds: float = 2.0,
    stitch_distance: float | None = None,
    background: str | Path | None = None,
) -> dict:
    """Run the full offline analysis and write tracks, metrics and figures."""
    directory = Path(session_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if stitch_distance is None:
        # One metre on a surveyed plane, or 2% of the frame width without one.
        stitch_distance = 1.0 if calibration is not None else 0.02
    tracks, unit, load_stats = load_tracks(directory, calibration, class_names)
    if not tracks:
        raise ValueError("该会话没有可用的带 ID 轨迹；请确认模型与跟踪已启用")
    merged, mapping = stitch_tracks(tracks, gap_seconds, stitch_distance)
    metrics = summarise(tracks, unit, cell_size=cell_size, bin_seconds=60.0,
                        gap_seconds=gap_seconds, stitch_distance=stitch_distance,
                        merged=merged, mapping=mapping)
    metrics.update(load_stats)
    metrics.update({
        "session_dir": str(directory.resolve()),
        "calibration_reprojection_rmse": (
            calibration.reprojection_rmse if calibration is not None else None
        ),
        "tracks_csv": str(write_tracks_csv(merged, output / "tracks.csv")),
    })
    unit_label = "m" if unit == "metre" else "画面"
    metrics["plan_view"] = str(plot_plan_view(
        merged, output / "plan_view.png",
        title="轨迹平面图" if unit == "metre" else "轨迹平面图（未标定，画面比例）",
        unit=unit_label, background=background,
    ))
    if unit == "metre":
        metrics["occupancy_map"] = str(plot_occupancy(
            merged, output / "occupancy.png", cell_size=cell_size, unit="m"))
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics

