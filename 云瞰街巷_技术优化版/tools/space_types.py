"""Classify space into use types from the recorded tracks.

Each grid cell gets a three-bit signature - stayed / passed / parked - and the
combination defines a space type. This is the same move as coding a facade with
a binary string: detection is not the result, the typology built on top of it is.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from traffic_app.portrait import read_image
from traffic_app.space import load_tracks, stitch_tracks


SESSION = Path(r"C:\Users\96901\Desktop\城垣杯加yolo\_site743\sessions"
               r"\session_20261022T124514Z_6d65ea9e3332")
OUT = Path(r"C:\Users\96901\Desktop\城垣杯加yolo\成果_实测报告\附件_图表")
CELL = 0.04
STATIONARY_SPEED = 0.02      # frame fraction per second
VEHICLES = ("bicycle", "motorcycle")

TYPE_COLOURS = {
    "停留主导型": (198, 106, 58),
    "穿行主导型": (38, 140, 128),
    "停放占用型": (214, 160, 60),
    "混合使用型": (31, 78, 121),
    "偶发经过型": (170, 178, 190),
}

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def cell_key(x: float, y: float) -> tuple[int, int]:
    return int(x / CELL), int(y / CELL)


def accumulate(tracks):
    stayed = defaultdict(float)
    passed = defaultdict(float)
    parked = defaultdict(float)
    people = defaultdict(float)
    for track in tracks:
        for a, b in zip(track.points, track.points[1:]):
            dt = b.timestamp - a.timestamp
            if dt <= 0 or dt > 5:
                continue
            speed = math.hypot(b.x - a.x, b.y - a.y) / dt
            steps = max(1, min(8, int(dt * 3)))
            for step in range(steps):
                f = (step + 0.5) / steps
                key = cell_key(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f)
                share = dt / steps
                if track.class_name == "person":
                    people[key] += share
                    if speed < STATIONARY_SPEED:
                        stayed[key] += share
                    else:
                        passed[key] += share
                elif track.class_name in VEHICLES and speed < STATIONARY_SPEED:
                    parked[key] += share
                    people[key] += share
    return stayed, passed, parked, people


def classify(stayed, passed, parked, key) -> str | None:
    total = stayed[key] + passed[key] + parked[key]
    if total < 6:
        return None
    if parked[key] >= 8 and parked[key] / total >= 0.35:
        return "停放占用型"
    ratio = stayed[key] / max(1e-6, stayed[key] + passed[key])
    if stayed[key] >= 6 and ratio >= 0.5:
        return "停留主导型"
    if passed[key] >= 6 and ratio <= 0.2:
        return "穿行主导型"
    if stayed[key] >= 4 and passed[key] >= 4:
        return "混合使用型"
    return "偶发经过型"


def draw_map(poster: Path, cells, stats, out: Path) -> None:
    background = read_image(poster)
    height, width = background.shape[:2]
    from PIL import Image as PILImage
    canvas = PILImage.fromarray(background[:, :, ::-1]).convert("RGBA")
    overlay = PILImage.new("RGBA", canvas.size, (0, 0, 0, 0))
    painter = ImageDraw.Draw(overlay)
    for (cx, cy), kind in cells.items():
        colour = TYPE_COLOURS[kind] + (150,)
        painter.rectangle([cx * CELL * width, cy * CELL * height,
                           (cx + 1) * CELL * width, (cy + 1) * CELL * height], fill=colour)
    composed = PILImage.alpha_composite(canvas, overlay).convert("RGB")
    painter = ImageDraw.Draw(composed)
    bold = ImageFont.truetype(FONT_BOLD, 26)
    small = ImageFont.truetype(FONT, 20)
    box_w, box_h = 420, 40 + 34 * len(stats)
    painter.rectangle([26, height - box_h - 26, 26 + box_w, height - 26],
                      fill=(255, 255, 255, 235), outline=(200, 206, 214))
    painter.text((44, height - box_h - 6), "空间类型（格数 / 面积占比）", font=bold, fill=(32, 38, 48))
    for index, item in enumerate(stats):
        y = height - box_h + 34 + index * 34
        painter.rectangle([44, y + 4, 66, y + 24], fill=TYPE_COLOURS[item["type"]])
        painter.text((78, y), f'{item["type"]}　{item["cells"]} 格 · {item["share"]*100:.0f}%',
                     font=small, fill=(40, 46, 56))
    composed.save(out)


def main() -> int:
    tracks, unit, _stats = load_tracks(SESSION, None, ["person"] + list(VEHICLES))
    merged, mapping = stitch_tracks(tracks, 2.0, 0.02)
    stayed, passed, parked, people = accumulate(merged)

    keys = set(stayed) | set(passed) | set(parked)
    cells: dict[tuple[int, int], str] = {}
    for key in keys:
        kind = classify(stayed, passed, parked, key)
        if kind:
            cells[key] = kind

    grouped = defaultdict(lambda: {"cells": 0, "person_seconds": 0.0, "stayed": 0.0, "passed": 0.0})
    for key, kind in cells.items():
        entry = grouped[kind]
        entry["cells"] += 1
        entry["person_seconds"] += people[key]
        entry["stayed"] += stayed[key]
        entry["passed"] += passed[key]
    total_cells = sum(item["cells"] for item in grouped.values())
    stats = []
    for kind, entry in sorted(grouped.items(), key=lambda kv: -kv[1]["cells"]):
        stats.append({
            "type": kind,
            "cells": entry["cells"],
            "share": entry["cells"] / total_cells,
            "person_seconds": round(entry["person_seconds"], 1),
            "stayed_seconds": round(entry["stayed"], 1),
            "passed_seconds": round(entry["passed"], 1),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    draw_map(SESSION / "poster.jpg", cells, stats, OUT / "13_空间类型分布.png")
    payload = {
        "cell": CELL,
        "stationary_speed": STATIONARY_SPEED,
        "tracks": len(merged),
        "occupied_cells": len(cells),
        "types": stats,
    }
    (OUT.parent / "空间类型.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
