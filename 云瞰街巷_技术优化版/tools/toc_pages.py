"""Locate headings by colour and map them to page numbers for a static TOC.

LibreOffice (and Word) paint the Heading styles in one accent blue while body
text, captions and table fills are black, grey or pale blue. Blue text that
starts at the left margin and is narrower than a figure is therefore a heading;
the ordered heading list from the rebuild says which one it is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


ACCENT = np.array([31, 78, 121])
LEFT_MARGIN_PX = (86, 150)
MAX_WIDTH_PX = 640


def heading_bands(image_path: Path, tolerance: int = 34) -> list[tuple[int, int, int]]:
    image = np.array(Image.open(image_path).convert("RGB")).astype(np.int16)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    # Per-channel closeness, plus "actually blue" so dark grey text cannot pass.
    mask = ((np.abs(red - ACCENT[0]) < tolerance)
            & (np.abs(green - ACCENT[1]) < tolerance)
            & (np.abs(blue - ACCENT[2]) < tolerance)
            & (blue - red > 45))
    rows = np.nonzero(mask.sum(axis=1) > 4)[0]
    if rows.size == 0:
        return []
    groups = []
    start = previous = rows[0]
    for y in rows[1:]:
        if y - previous > 4:
            groups.append((start, previous))
            start = y
        previous = y
    groups.append((start, previous))

    bands = []
    for top, bottom in groups:
        height = bottom - top
        if not (11 <= height <= 26):
            continue
        columns = np.nonzero(mask[top:bottom + 1].sum(axis=0) > 0)[0]
        if columns.size == 0:
            continue
        left, right = int(columns.min()), int(columns.max())
        if not (LEFT_MARGIN_PX[0] <= left <= LEFT_MARGIN_PX[1]):
            continue
        if not (60 <= right - left <= 420):
            continue
        bands.append((top, bottom, right - left))
    return bands


def level_of(height: int) -> int:
    # 16 pt chapter titles render clearly taller than 13 pt section titles;
    # the single 11.5 pt sub-heading is deliberately left out of the contents.
    return 1 if height >= 16 else 2


def main() -> int:
    render_dir = Path(r"_render_toc")
    dump = json.loads(Path(r"_plan_dump.json").read_text(encoding="utf-8"))
    headings: list[tuple[int, str]] = []
    for index, block in enumerate(dump):
        if index < 32 or block["type"] != "text":
            continue
        text = block["text"]
        if text == "参考资料" or re.match(r"^第[一二三四五六七八九十]+章", text):
            if text.startswith("第四章"):
                headings.append((2, "3.10 硬件与现场作业"))
            headings.append((1, text))
        elif re.match(r"^\d+\.\d+\.\d+[\s　]", text):
            continue  # sub-headings stay out of the contents page
        else:
            match = re.match(r"^(\d+\.\d+)[\s　](.+)$", text)
            if match and len(text) <= 26 and not re.search(r"[。；：！？]", match.group(2)):
                headings.append((2, text))
    headings.insert(len(headings) - 1, (1, "3.10 硬件与现场作业")) if False else None

    detected: list[tuple[int, int]] = []
    for number, page in enumerate(sorted(render_dir.glob("page-*.png")), start=1):
        if number <= 2:
            continue
        for top, bottom, _width in heading_bands(page):
            detected.append((number, level_of(bottom - top)))

    print("detected", len(detected), "expected", len(headings))
    result = []
    cursor = 0
    for page, level in detected:
        if cursor >= len(headings):
            break
        expected_level, text = headings[cursor]
        if expected_level != level:
            print(f"!! page {page}: detected level {level} but expected {expected_level} ({text})")
            continue
        result.append({"level": level, "text": text, "page": page})
        cursor += 1
    print("mapped", cursor, "of", len(headings))
    Path(r"_toc.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for item in result:
        print(item["page"], "  " * (item["level"] - 1), item["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
