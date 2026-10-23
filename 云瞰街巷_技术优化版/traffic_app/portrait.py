"""One-click space-use portrait for a recorded session.

The recorder stores one row per detection per frame. This module turns those
rows into the four things a planning reader actually asks about a small public
space: *where* people were, *how long* they stayed, *how* they moved, and
*where* they came close to a moving bicycle. No model is loaded here, so the
portrait can be rebuilt from an old session as many times as needed.

Everything in this module works without a ground calibration. Without one the
coordinates stay in frame fractions and every figure is labelled accordingly;
with one the same code produces metres. Nothing here guesses a scale.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .calibration import Calibration
from .space import Track, load_tracks, stitch_tracks, summarise, write_tracks_csv


VEHICLE_CLASSES = ("bicycle", "motorcycle", "bus", "truck", "car")
CLASS_LABELS = {
    "person": "行人",
    "bicycle": "自行车",
    "motorcycle": "电动车/摩托车",
    "car": "小汽车",
    "bus": "公交",
    "truck": "货车",
}

HEAT_BAR_LOW = "low"
HEAT_BAR_HIGH = "high"


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image through a byte buffer so non-ASCII paths work."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray, quality: int = 92) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower() or ".jpg"
    ok, buffer = cv2.imencode(suffix, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError(f"无法编码图片：{destination}")
    buffer.tofile(str(destination))
    return destination


def _caption(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Draw a translucent caption block along the bottom-left corner."""
    height, width = image.shape[:2]
    scale = max(0.6, min(1.6, width / 1400.0))
    thickness = max(1, int(round(scale)))
    line_height = int(26 * scale)
    box_height = line_height * len(lines) + int(14 * scale)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - box_height), (int(640 * scale), height), (8, 12, 20), -1)
    image = cv2.addWeighted(overlay, 0.72, image, 0.28, 0)
    for index, text in enumerate(lines):
        y = height - box_height + int((index + 1) * line_height) - int(4 * scale)
        cv2.putText(image, text, (int(14 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52 * scale, (238, 246, 255), thickness, cv2.LINE_AA)
    return image


# --------------------------------------------------------------------------
# presence heat
# --------------------------------------------------------------------------

def presence_grid(
    tracks: Sequence[Track],
    *,
    sigma: float | None = None,
    min_duration: float = 0.0,
    mask: np.ndarray | None = None,
    stationary_only: bool = False,
    speed_threshold: float = 0.02,
) -> np.ndarray:
    """Accumulate observed seconds per grid cell in frame-fraction space.

    Each segment contributes its own duration, so the map is person-seconds
    rather than detection count: one person standing for a minute outweighs
    six people walking past. With ``stationary_only`` only the segments where a
    person barely moved are counted, which separates *staying* from *passing*.
    """
    size = 1000
    grid = np.zeros((size, size), dtype=np.float32)
    for track in tracks:
        if track.duration < min_duration:
            continue
        points = track.points
        for index, point in enumerate(points):
            if index == 0:
                weight = 0.25
            else:
                dt = point.timestamp - points[index - 1].timestamp
                if stationary_only:
                    if dt <= 0:
                        continue
                    speed = math.hypot(point.x - points[index - 1].x,
                                       point.y - points[index - 1].y) / dt
                    if speed >= speed_threshold:
                        continue
                weight = min(1.0, max(0.02, dt))
            x = int(point.x * size)
            y = int(point.y * size)
            if 0 <= x < size and 0 <= y < size:
                grid[y, x] += weight
    kernel = sigma if sigma and sigma > 0 else max(5.0, size * 0.014)
    grid = cv2.GaussianBlur(grid, (0, 0), kernel)
    if mask is not None:
        grid *= mask.astype(np.float32)
    return grid


def _normalise(grid: np.ndarray, percentile: float = 97.0, gamma: float = 0.5) -> np.ndarray:
    """Scale by a high percentile, not by the single busiest pixel.

    Linear normalisation against the maximum lets one person who stood still for
    two minutes flatten the whole plaza to zero. Scaling against the 97th
    percentile and applying a gamma curve keeps the rest of the space readable.
    """
    positive = grid[grid > 0]
    if positive.size == 0:
        return grid
    scale = float(np.percentile(positive, percentile))
    normalised = np.clip(grid / max(scale, 1e-6), 0.0, 1.0)
    return np.power(normalised, gamma)


def render_heatmap(
    tracks: Sequence[Track],
    background: np.ndarray,
    *,
    alpha: float = 0.58,
    threshold: float = 0.15,
    min_duration: float = 2.0,
    mask_polygon: Sequence[tuple[float, float]] | None = None,
    stationary_only: bool = False,
    speed_threshold: float = 0.02,
    title: str = "observed presence, person-seconds",
) -> np.ndarray:
    height, width = background.shape[:2]
    mask = None
    if mask_polygon:
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        polygon = np.array([[int(x * 1000), int(y * 1000)] for x, y in mask_polygon], np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    grid = presence_grid(tracks, min_duration=min_duration, mask=mask,
                         stationary_only=stationary_only, speed_threshold=speed_threshold)
    normalized = _normalise(grid)
    heat = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(background, 1.0 - alpha, heat, alpha, 0)
    keep = cv2.resize(np.uint8(normalized > threshold) * 255, (width, height),
                      interpolation=cv2.INTER_NEAREST)
    image = np.where(keep[..., None] > 0, blended, background)

    scale = max(1.0, width / 1400.0)
    bar_width, bar_height = int(210 * scale), int(14 * scale)
    bar_x, bar_y = int(26 * scale), height - int(170 * scale)
    gradient = np.linspace(0, 1, bar_width, dtype=np.float32)[None, :]
    bar = cv2.applyColorMap(np.uint8(gradient * 255), cv2.COLORMAP_TURBO)
    bar = cv2.resize(bar, (bar_width, bar_height), interpolation=cv2.INTER_NEAREST)
    image[bar_y:bar_y + bar_height, bar_x:bar_x + bar_width] = bar
    cv2.rectangle(image, (bar_x - 1, bar_y - 1),
                  (bar_x + bar_width, bar_y + bar_height), (245, 250, 255), 1)
    cv2.putText(image, HEAT_BAR_LOW, (bar_x, bar_y - int(8 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (245, 250, 255), max(1, int(scale)), cv2.LINE_AA)
    cv2.putText(image, HEAT_BAR_HIGH, (bar_x + bar_width - int(44 * scale), bar_y - int(8 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (245, 250, 255), max(1, int(scale)), cv2.LINE_AA)
    if mask_polygon:
        polygon = np.array([[int(x * width), int(y * height)] for x, y in mask_polygon], np.int32)
        cv2.polylines(image, [polygon], True, (120, 220, 200), max(1, int(2 * scale)), cv2.LINE_AA)
    return _caption(image, [title, f"tracks: {len(tracks)}   threshold: {threshold:.2f}"])


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------

def _track_colour(fraction: float) -> tuple[int, int, int]:
    hue = int(round(120 * (1.0 - min(max(fraction, 0.0), 1.0))))
    patch = np.uint8([[[hue, 235, 255]]])
    blue, green, red = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(blue), int(green), int(red)


def render_trails(
    tracks: Sequence[Track],
    background: np.ndarray,
    *,
    min_points: int = 8,
    min_duration: float = 3.0,
    smooth: int = 5,
    max_tracks: int = 220,
    title: str = "observed trajectories",
) -> np.ndarray:
    height, width = background.shape[:2]
    usable = [track for track in tracks
              if len(track.points) >= min_points and track.duration >= min_duration]
    if len(usable) > max_tracks:
        usable = sorted(usable, key=lambda track: -track.duration)[:max_tracks]
    overlay = background.copy()
    scale = max(1.0, width / 1400.0)
    starts = [track.start for track in usable] or [0.0, 1.0]
    t0, t1 = min(starts), max(max(track.end for track in usable), min(starts) + 1e-6)
    for track in usable:
        points = np.array([[int(point.x * width), int(point.y * height)]
                           for point in track.points], np.int32)
        if smooth >= 3 and len(points) >= smooth:
            kernel = np.ones(smooth, dtype=float) / smooth
            points = np.stack([np.convolve(points[:, 0], kernel, mode="valid"),
                               np.convolve(points[:, 1], kernel, mode="valid")], axis=1).astype(np.int32)
        if len(points) < 2:
            continue
        colour = _track_colour((track.start - t0) / max(t1 - t0, 1e-6))
        cv2.polylines(overlay, [points], False, colour, max(1, int(2 * scale)), cv2.LINE_AA)
        cv2.circle(overlay, tuple(points[0]), int(3 * scale), (90, 226, 90), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(points[-1]), int(3 * scale), (70, 70, 244), -1, cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, 0.86, background, 0.14, 0)
    return _caption(blended, [title, f"drawn: {len(usable)}   green dot: start   red dot: end"])


def render_trail_frames(
    tracks: Sequence[Track],
    background: np.ndarray,
    *,
    seconds: float = 12.0,
    fps: float = 25.0,
    tail: float = 6.0,
    min_duration: float = 2.0,
) -> list[np.ndarray]:
    """Fading trail frames, for the playback clip.

    Each frame draws only the part of every trajectory that happened during the
    preceding ``tail`` seconds, so the picture moves the way the space was used
    instead of showing a static spaghetti plot.
    """
    height, width = background.shape[:2]
    usable = [track for track in tracks
              if len(track.points) >= 4 and track.duration >= min_duration]
    if not usable:
        return []
    start = min(track.start for track in usable)
    end = max(track.end for track in usable)
    total = max(1.0, min(seconds, end - start))
    step = 1.0 / max(1.0, fps)
    scale = max(1.0, width / 1400.0)
    frames: list[np.ndarray] = []
    moment = start
    while moment <= start + total:
        overlay = background.copy()
        for track in usable:
            if track.start > moment:
                continue
            points = [point for point in track.points
                      if moment - tail <= point.timestamp <= moment]
            if len(points) < 2:
                continue
            arrays = np.array([[int(point.x * width), int(point.y * height)]
                               for point in points], np.int32)
            age = max(0.0, moment - points[-1].timestamp)
            colour = _track_colour(1.0 - age / max(tail, 1e-6))
            cv2.polylines(overlay, [arrays], False, colour, max(1, int(3 * scale)), cv2.LINE_AA)
            cv2.circle(overlay, tuple(arrays[-1]), int(4 * scale), (60, 240, 240), -1, cv2.LINE_AA)
        image = cv2.addWeighted(overlay, 0.9, background, 0.1, 0)
        image = _caption(image, [f"window {moment - start:05.1f}s",
                                 f"tracking {tail:.0f}s of movement"])
        frames.append(image)
        moment += step
    return frames


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------

def _stationary_ratio(tracks: Sequence[Track], unit: str) -> tuple[float, int, int]:
    threshold = 0.2 if unit == "metre" else 0.02
    stationary = moving = 0
    for track in tracks:
        for previous, current in zip(track.points, track.points[1:]):
            dt = current.timestamp - previous.timestamp
            if dt <= 0:
                continue
            speed = math.hypot(current.x - previous.x, current.y - previous.y) / dt
            if speed < threshold:
                stationary += 1
            else:
                moving += 1
    return stationary / max(1, stationary + moving), stationary, moving


def class_breakdown(tracks: Sequence[Track]) -> dict[str, dict]:
    """Per-class counts and time, so a bicycle is never counted as a person."""
    grouped: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        grouped[track.class_name].append(track)
    result: dict[str, dict] = {}
    for name, group in sorted(grouped.items()):
        durations = [track.duration for track in group]
        result[name] = {
            "label": CLASS_LABELS.get(name, name),
            "tracks": len(group),
            "person_seconds": round(sum(durations), 1),
            "mean_dwell_seconds": round(sum(durations) / len(durations), 2) if durations else 0.0,
        }
    return result


def _sample_per_bucket(tracks: Sequence[Track], bucket: float = 0.5) -> dict[int, list[tuple[int, str, float, float, float]]]:
    """One representative position per track per bucket, plus how far it moved.

    The travel distance inside a bucket is what separates a bicycle riding past
    from a scooter parked against the wall: only the first can produce an
    interaction that a planner would care about.
    """
    buckets: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for track in tracks:
        for point in track.points:
            buckets[int(point.timestamp // bucket)][track.track_id].append((point.x, point.y))
    sampled: dict[int, list[tuple[int, str, float, float, float]]] = {}
    names = {track.track_id: track.class_name for track in tracks}
    for index, entries in buckets.items():
        rows = []
        for track_id, points in entries.items():
            xs = sum(point[0] for point in points) / len(points)
            ys = sum(point[1] for point in points) / len(points)
            travel = max((math.hypot(point[0] - xs, point[1] - ys) for point in points), default=0.0)
            rows.append((track_id, names.get(track_id, ""), xs, ys, travel))
        sampled[index] = rows
    return sampled


def near_miss_events(
    tracks: Sequence[Track],
    *,
    threshold: float = 0.03,
    bucket: float = 0.5,
    min_vehicle_travel: float = 0.01,
) -> dict:
    """Count pedestrian / *moving* bicycle pairs that came within ``threshold``.

    Distances are frame fractions when the session is uncalibrated, so the count
    is a screening indicator rather than a safety verdict: it says "these two
    were close on the ground plane while the bicycle was moving", not "this was
    a collision risk". A parked scooter next to a standing group is excluded by
    the ``min_vehicle_travel`` filter, because that is storage, not circulation.
    """
    sampled = _sample_per_bucket(tracks, bucket)
    pairs: set[tuple[int, int]] = set()
    moments: dict[tuple[int, int], float] = {}
    for rows in sampled.values():
        people = [row for row in rows if row[1] == "person"]
        vehicles = [row for row in rows
                    if row[1] in VEHICLE_CLASSES and row[4] >= min_vehicle_travel]
        if not people or not vehicles:
            continue
        people_array = np.array([[row[2], row[3]] for row in people], dtype=np.float32)
        for vehicle_id, _name, vx, vy, _travel in vehicles:
            distances = np.hypot(people_array[:, 0] - vx, people_array[:, 1] - vy)
            for person_index in np.nonzero(distances <= threshold)[0]:
                key = (int(people[person_index][0]), int(vehicle_id))
                pairs.add(key)
                moments[key] = moments.get(key, 0.0) + bucket
    involved_people = {key[0] for key in pairs}
    involved_vehicles = {key[1] for key in pairs}
    return {
        "threshold_frame_fraction": threshold,
        "bucket_seconds": bucket,
        "min_vehicle_travel": min_vehicle_travel,
        "events": len(pairs),
        "pedestrians_involved": len(involved_people),
        "vehicles_involved": len(involved_vehicles),
        "longest_overlap_seconds": round(max(moments.values()), 1) if moments else 0.0,
        "note": "只统计非机动车在运动中的近距接近；单位为画面宽比例，不代表碰撞风险判定。",
    }


def parked_vehicle_load(
    tracks: Sequence[Track],
    *,
    min_seconds: float = 30.0,
    speed_threshold: float | None = None,
) -> dict:
    """Long, nearly still bicycles: the storage demand side of the same corner."""
    threshold = speed_threshold if speed_threshold is not None else 0.02
    parked = []
    for track in tracks:
        if track.class_name not in VEHICLE_CLASSES:
            continue
        if track.duration < min_seconds or len(track.points) < 4:
            continue
        travelled = sum(math.hypot(b.x - a.x, b.y - a.y)
                        for a, b in zip(track.points, track.points[1:]))
        if travelled / max(track.duration, 1e-6) > threshold:
            continue
        parked.append(track)
    cells = {(int(track.points[0].x * 100), int(track.points[0].y * 100)) for track in parked}
    return {
        "min_seconds": min_seconds,
        "tracks": len(parked),
        "vehicle_seconds": round(sum(track.duration for track in parked), 1),
        "distinct_places": len(cells),
    }


def render_flow_field(
    tracks: Sequence[Track],
    background: np.ndarray,
    *,
    columns: int = 26,
    rows: int = 15,
    min_samples: int = 6,
    gain: float = 3.2,
    title: str = "dominant walking direction per cell",
) -> np.ndarray:
    """Arrow per grid cell: where people in that cell were heading.

    A spaghetti plot of every trajectory is unreadable at 1 200 tracks and tells
    a reader nothing about direction. Averaging each track segment into a cell
    turns the same data into a flow field, which is the figure a planner can use.
    """
    height, width = background.shape[:2]
    cell_width, cell_height = width / columns, height / rows
    sums = np.zeros((rows, columns, 2), dtype=np.float64)
    counts = np.zeros((rows, columns), dtype=np.int32)
    for track in tracks:
        for previous, current in zip(track.points, track.points[1:]):
            dt = current.timestamp - previous.timestamp
            if dt <= 0 or dt > 2.0:
                continue
            mx = (previous.x + current.x) / 2
            my = (previous.y + current.y) / 2
            col = int(mx * columns)
            row = int(my * rows)
            if not (0 <= col < columns and 0 <= row < rows):
                continue
            sums[row, col, 0] += (current.x - previous.x) / dt
            sums[row, col, 1] += (current.y - previous.y) / dt
            counts[row, col] += 1
    canvas = background.copy()
    overlay = background.copy()
    scale = max(1.0, width / 1400.0)
    peak = 0.0
    means = np.zeros_like(sums)
    for row in range(rows):
        for col in range(columns):
            if counts[row, col] < min_samples:
                continue
            means[row, col] = sums[row, col] / counts[row, col]
            peak = max(peak, float(np.hypot(*means[row, col])))
    if peak <= 0:
        return _caption(canvas, [title, "not enough movement to draw a field"])
    for row in range(rows):
        for col in range(columns):
            if counts[row, col] < min_samples:
                continue
            dx, dy = means[row, col]
            magnitude = float(np.hypot(dx, dy))
            if magnitude < peak * 0.06:
                continue
            length = min(cell_width, cell_height) * 0.46 * min(1.0, magnitude / peak * gain)
            norm = max(magnitude, 1e-9)
            ux, uy = dx / norm, dy / norm
            cx = (col + 0.5) * cell_width
            cy = (row + 0.5) * cell_height
            start = (int(cx - ux * length), int(cy - uy * length))
            end = (int(cx + ux * length), int(cy + uy * length))
            angle = math.degrees(math.atan2(-uy, ux)) % 360
            patch = np.uint8([[[int(angle / 2), 220, 250]]])
            blue, green, red = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
            colour = (int(blue), int(green), int(red))
            weight = int(min(1.0, magnitude / peak * 1.6) * counts[row, col] ** 0.35 * 1.1) + 2
            cv2.arrowedLine(overlay, start, end, colour, max(2, int(3 * scale)),
                            cv2.LINE_AA, tipLength=0.32)
            cv2.circle(overlay, (int(cx), int(cy)), max(2, weight // 2), colour, -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.9, background, 0.1, 0)
    return _caption(canvas, [title, f"cells: {int((counts >= min_samples).sum())}   length ~ mean speed"])


def render_time_curve(
    metrics: dict,
    width: int = 1500,
    height: int = 620,
    *,
    title: str = "presence per minute",
) -> np.ndarray:
    """One small chart: person-seconds per bin plus distinct tracks per bin."""
    bins = metrics.get("time_bins") or []
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    left, right, top, bottom = 96, width - 96, 92, height - 96
    cv2.putText(canvas, title, (left, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (30, 34, 44), 2, cv2.LINE_AA)
    if not bins:
        cv2.putText(canvas, "no data", (left, top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
        return canvas
    cv2.line(canvas, (left, top), (left, bottom), (60, 64, 74), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (60, 64, 74), 2)
    seconds = [float(item.get("person_seconds", 0.0)) for item in bins]
    counts = [float(item.get("track_count", 0.0)) for item in bins]
    labels = [float(item.get("start_seconds", 0.0)) for item in bins]
    peak_seconds = max(seconds + [1.0])
    peak_counts = max(counts + [1.0])
    slot = (right - left) / max(1, len(bins))
    bar_width = slot * 0.62
    for index, value in enumerate(seconds):
        x0 = int(left + slot * index + (slot - bar_width) / 2)
        x1 = int(x0 + bar_width)
        y0 = int(bottom - value / peak_seconds * (bottom - top))
        cv2.rectangle(canvas, (x0, y0), (x1, bottom), (196, 150, 78), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, bottom), (150, 112, 52), 1)
        cv2.putText(canvas, f"{value:.0f}", (x0 + 2, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (90, 74, 44), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{labels[index] / 60:.0f}", (x0 + 2, bottom + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 96, 108), 1, cv2.LINE_AA)
    points = []
    for index, value in enumerate(counts):
        x = int(left + slot * index + slot / 2)
        y = int(bottom - value / peak_counts * (bottom - top) * 0.86)
        points.append((x, y))
    if len(points) >= 2:
        cv2.polylines(canvas, [np.array(points, np.int32)], False, (170, 90, 40), 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, point, 6, (170, 90, 40), -1, cv2.LINE_AA)
    cv2.putText(canvas, "bars: person-seconds per minute", (left, height - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (196, 150, 78), 2, cv2.LINE_AA)
    cv2.putText(canvas, "line: distinct tracks per minute", (left + 430, height - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 90, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "minute", ((left + right) // 2 - 30, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (90, 96, 108), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

@dataclass
class Portrait:
    out_dir: Path
    metrics: dict
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def headline(self) -> list[tuple[str, str]]:
        """Short label/value pairs for a GUI panel or a report table."""
        m = self.metrics
        unit = "米" if m.get("coordinate_unit") == "metre" else "画面宽比例"
        people = (m.get("class_breakdown") or {}).get("person", {})
        return [
            ("坐标单位", unit),
            ("行人轨迹（拼接后）", f"{people.get('tracks', m.get('tracks_after_stitch', 0))} 条"),
            ("累计人·秒", f"{m.get('person_seconds', 0):.0f}"),
            ("平均停留", f"{m.get('mean_dwell_seconds', 0):.1f} 秒"),
            ("中位停留", f"{m.get('median_dwell_seconds', 0):.1f} 秒"),
            ("停留段占比", f"{m.get('stationary_ratio', 0) * 100:.0f}%"),
            ("占用网格", f"{m.get('occupied_cells', 0)} 个"),
            ("近距人车交织", f"{(m.get('near_miss') or {}).get('events', 0)} 次"),
            ("静止非机动车", f"{(m.get('parked_vehicles') or {}).get('tracks', 0)} 段"),
        ]


def build_portrait(
    session_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    calibration: Calibration | None = None,
    classes: Iterable[str] | None = ("person", "bicycle", "motorcycle"),
    background: str | Path | None = None,
    mask_polygon: Sequence[tuple[float, float]] | None = None,
    near_miss_threshold: float = 0.03,
    cell_size: float | None = None,
    gap_seconds: float = 2.0,
    playback_seconds: float = 0.0,
) -> Portrait:
    """Build the whole portrait; safe to call from a GUI thread."""
    session = Path(session_dir)
    output = Path(out_dir) if out_dir else session / "space"
    output.mkdir(parents=True, exist_ok=True)

    tracks, unit, load_stats = load_tracks(session, calibration, classes)
    if not tracks:
        raise ValueError("该会话没有可用的带 ID 轨迹；请确认模型与跟踪已启用，且已运行足够帧数")
    if cell_size is None:
        cell_size = 0.5 if calibration is not None else 0.02
    stitch_distance = 1.0 if calibration is not None else 0.02
    merged, mapping = stitch_tracks(tracks, gap_seconds, stitch_distance)
    metrics = summarise(tracks, unit, cell_size=cell_size, bin_seconds=60.0,
                        gap_seconds=gap_seconds, stitch_distance=stitch_distance,
                        merged=merged, mapping=mapping)
    metrics.update(load_stats)
    metrics["class_breakdown"] = class_breakdown(merged)
    metrics["near_miss"] = near_miss_events(merged, threshold=near_miss_threshold)
    metrics["parked_vehicles"] = parked_vehicle_load(
        merged, speed_threshold=0.2 if unit == "metre" else 0.02)
    # Dwell and presence are reported for pedestrians only: mixing bicycles into
    # "person-seconds" would quietly inflate every number a reader quotes.
    people = [track for track in merged if track.class_name == "person"]
    if people:
        durations = [track.duration for track in people]
        ratio, stationary, moving = _stationary_ratio(people, unit)
        metrics["person_seconds_all_targets"] = metrics["person_seconds"]
        metrics["person_seconds"] = round(sum(durations), 2)
        metrics["pedestrian_tracks"] = len(people)
        metrics["mean_dwell_seconds"] = round(sum(durations) / len(durations), 2)
        metrics["median_dwell_seconds"] = round(float(np.median(durations)), 2)
        metrics["max_dwell_seconds"] = round(max(durations), 2)
        metrics["stationary_ratio"] = round(ratio, 3)
        metrics["stationary_segments"] = stationary
        metrics["moving_segments"] = moving
        metrics.setdefault("notes", []).append(
            "人数、停留与停留占比只统计行人；自行车和电动车在分类构成中单列。")
    metrics["session_dir"] = str(session.resolve())
    metrics["calibration_reprojection_rmse"] = (
        calibration.reprojection_rmse if calibration is not None else None)
    metrics["generated_by"] = "traffic_app.portrait"

    files: dict[str, Path] = {}
    files["tracks"] = write_tracks_csv(merged, output / "tracks.csv")

    poster = Path(background) if background else session / "poster.jpg"
    frame = read_image(poster) if poster.is_file() else None
    warnings: list[str] = []
    if frame is None:
        warnings.append("会话没有封面帧，热力图与轨迹叠加已跳过；如需图形请在原视频上重跑。")
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
    # Keep figures light enough to paste into a document.
    height, width = frame.shape[:2]
    if width > 1600:
        frame = cv2.resize(frame, (1600, int(height * 1600 / width)), interpolation=cv2.INTER_AREA)
    files["heatmap"] = write_image(
        output / "heatmap.jpg",
        render_heatmap(merged, frame, mask_polygon=mask_polygon,
                       title="observed presence, person-seconds"))
    files["dwell_heat"] = write_image(
        output / "heatmap_dwell.jpg",
        render_heatmap(merged, frame, mask_polygon=mask_polygon, stationary_only=True,
                       speed_threshold=0.2 if unit == "metre" else 0.02,
                       title="where people stopped, stationary seconds"))
    files["trails"] = write_image(output / "trails.jpg", render_trails(merged, frame))
    files["flow"] = write_image(output / "flow.jpg", render_flow_field(merged, frame))
    files["time_curve"] = write_image(output / "time_curve.png", render_time_curve(metrics), quality=100)

    if playback_seconds > 0:
        frames = render_trail_frames(merged, frame, seconds=playback_seconds)
        if frames:
            files["playback"] = write_playback(frames, output / "playback.mp4")

    metrics["figures"] = {key: str(value) for key, value in files.items()}
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["metrics"] = output / "metrics.json"
    (output / "summary.md").write_text(
        _summary_markdown(metrics, files, session), encoding="utf-8")
    files["summary"] = output / "summary.md"
    return Portrait(out_dir=output, metrics=metrics, files=files, warnings=warnings)


def write_playback(frames: Sequence[np.ndarray], path: str | Path, fps: float = 25.0,
                   max_width: int = 1280) -> Path:
    """Write frames to an mp4 through a temp ASCII path, then move into place."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if frames and frames[0].shape[1] > max_width:
        scale = max_width / frames[0].shape[1]
        frames = [cv2.resize(frame, (max_width, int(frame.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA) for frame in frames]
    height, width = frames[0].shape[:2]
    temporary = destination.with_name("_playback_tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError("无法创建播放视频（缺少 mp4v 编码器）")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    if destination.exists():
        destination.unlink()
    temporary.replace(destination)
    return destination


def _summary_markdown(metrics: dict, files: dict[str, Path], session: Path) -> str:
    unit = "米制（已标定）" if metrics.get("coordinate_unit") == "metre" else "画面比例（未标定）"
    classes = metrics.get("class_breakdown") or {}
    lines = [
        "# 空间使用画像摘要",
        "",
        f"- 会话：`{session.name}`",
        f"- 坐标：{unit}",
        f"- 轨迹：原始 {metrics['tracks_raw']} 条，拼接后 {metrics['tracks_after_stitch']} 条"
        f"（合并 {metrics['merged_fragments']} 段）",
        f"- 累计人·秒：{metrics['person_seconds']}",
        f"- 平均停留 {metrics['mean_dwell_seconds']} 秒，中位 {metrics['median_dwell_seconds']} 秒，"
        f"最长 {metrics['max_dwell_seconds']} 秒",
        f"- 停留段占比：{metrics['stationary_ratio']}",
        f"- 占用网格：{metrics['occupied_cells']} 个",
        "",
        "## 分类构成",
        "",
        "| 类别 | 轨迹数 | 累计秒数 | 平均停留（秒） |",
        "| --- | --- | --- | --- |",
    ]
    for entry in classes.values():
        lines.append(f"| {entry['label']} | {entry['tracks']} | {entry['person_seconds']} |"
                     f" {entry['mean_dwell_seconds']} |")
    near = metrics.get("near_miss") or {}
    parked = metrics.get("parked_vehicles") or {}
    lines += [
        "",
        "## 交织与停放",
        "",
        f"- 近距人车交织：{near.get('events', 0)} 次，涉及行人 {near.get('pedestrians_involved', 0)} 位、"
        f"非机动车 {near.get('vehicles_involved', 0)} 辆",
        f"- 筛选阈值：{near.get('threshold_frame_fraction', 0)} 画面宽比例，"
        f"按 {near.get('bucket_seconds', 0)} 秒分桶；{near.get('note', '')}",
        f"- 静止非机动车：{parked.get('tracks', 0)} 段，累计 {parked.get('vehicle_seconds', 0)} 秒，"
        f"分布在 {parked.get('distinct_places', 0)} 个位置",
        "",
        "## 输出文件",
        "",
    ]
    for key, value in files.items():
        lines.append(f"- `{value.name}`（{key}）")
    lines += ["", "## 口径", ""]
    for note in metrics.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"
