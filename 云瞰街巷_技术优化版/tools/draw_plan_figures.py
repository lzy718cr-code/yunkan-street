"""Draw the three figures the plan was missing.

A technology roadmap, a two-year Gantt and a funding pie, all in the same
palette and typeface as the rest of the document so they read as one system.
Everything is generated from numbers held in this file, so re-runs after a
budget change are a one-line edit rather than a redraw.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
INK = (32, 38, 48)
MUTED = (108, 118, 132)
ACCENT = (31, 78, 121)
TEAL = (38, 140, 128)
WARM = (198, 106, 58)
GOLD = (214, 160, 60)
PALE = (238, 244, 250)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def rounded(draw, box, radius, fill=None, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, colour, width=3, head=11):
    draw.line([start, end], fill=colour, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for sign in (1, -1):
        draw.line([end,
                   (end[0] - head * math.cos(angle + sign * 0.42),
                    end[1] - head * math.sin(angle + sign * 0.42))],
                  fill=colour, width=width)


def roadmap(path: Path) -> None:
    width, height = 1700, 400
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    stages = [
        ("① 采集", "固定机位视频", "点位与机位记录"),
        ("② 检测跟踪", "人 / 自行车 / 电动车", "逐帧 ID 与框"),
        ("③ 逐帧记录", "CSV：帧号、时间、坐标", "可重算的原始记录"),
        ("④ 轨迹重建", "断轨拼接", "带时长与路径的轨迹"),
        ("⑤ 画像指标", "强度 · 停留 · 流向 · 交织", "指标表与三张图"),
        ("⑥ 规划判断", "问题地图与更新建议", "诊断报告"),
    ]
    box_width, box_height, gap = 236, 190, 46
    left, top = 46, 96
    for index, (title, middle, output) in enumerate(stages):
        x = left + index * (box_width + gap)
        fill = PALE if index < 5 else (255, 244, 236)
        outline = ACCENT if index < 5 else WARM
        rounded(draw, (x, top, x + box_width, top + box_height), 14, fill, outline, 2)
        draw.text((x + box_width / 2, top + 24), title, font=font(24, True),
                  fill=outline, anchor="ma")
        draw.multiline_text((x + box_width / 2, top + 66), middle, font=font(18),
                            fill=INK, anchor="ma", align="center", spacing=6)
        draw.line([(x + 16, top + box_height - 52), (x + box_width - 16, top + box_height - 52)],
                  fill=(214, 220, 228), width=1)
        draw.text((x + box_width / 2, top + box_height - 26), output, font=font(17),
                  fill=MUTED, anchor="mm")
        if index < len(stages) - 1:
            arrow(draw, (x + box_width + 6, top + box_height / 2),
                  (x + box_width + gap - 6, top + box_height / 2), (150, 160, 175), 3)

    draw.text((46, 30), "从视频到规划判断：六步一条流水线", font=font(26, True), fill=INK)
    draw.text((46, top + box_height + 28),
              "每一步的产物都能被下一步单独检查；任何一步不成立，后面的数字就不出。",
              font=font(19), fill=MUTED)
    image.save(path)


def gantt(path: Path) -> None:
    width, height = 1700, 620
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    rows = [
        ("点位与采集", 0, 6, TEAL),
        ("软件与画像层", 0, 9, ACCENT),
        ("标定与米制化", 3, 9, GOLD),
        ("试点报告与复测", 6, 18, WARM),
        ("商务与报价体系", 9, 24, MUTED),
    ]
    months = 24
    left, right = 260, width - 60
    top, row_height = 156, 72
    draw.text((46, 30), "两年推进节奏（月）", font=font(26, True), fill=INK)
    draw.text((46, 68), "同一个团队并行推进，但每一行的交付物是独立的，可以单独验收。",
              font=font(19), fill=MUTED)

    for step in range(0, months + 1, 3):
        x = left + (right - left) * step / months
        draw.line([(x, top - 16), (x, top + row_height * len(rows))], fill=(228, 232, 238), width=1)
        draw.text((x, top - 34), f"{step}", font=font(17), fill=MUTED, anchor="ma")
    draw.text((right, top - 34), "月", font=font(17), fill=MUTED, anchor="lm")

    for index, (name, start, end, colour) in enumerate(rows):
        y = top + index * row_height
        if index % 2 == 0:
            draw.rectangle([40, y, right + 10, y + row_height - 8], fill=(249, 250, 252))
        draw.text((52, y + 26), name, font=font(20, True), fill=INK)
        x0 = left + (right - left) * start / months
        x1 = left + (right - left) * end / months
        rounded(draw, (x0, y + 14, x1, y + 50), 8, colour, colour, 1)
        draw.text(((x0 + x1) / 2, y + 32), f"{start}–{end} 月", font=font(16),
                  fill=(255, 255, 255), anchor="mm")

    milestones = [(6, "首个点位报告"), (9, "米制版上线"), (18, "5 个点位"), (24, "20 个点位")]
    for month, label in milestones:
        x = left + (right - left) * month / months
        y = top + row_height * len(rows) - 4
        draw.polygon([(x, y), (x - 7, y + 14), (x + 7, y + 14)], fill=WARM)
        draw.text((x, y + 20), label, font=font(16), fill=WARM, anchor="ma")

    image.save(path)


def budget(path: Path) -> None:
    size = (1000, 660)
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    items = [
        ("采集与推理设备", 3.0, ACCENT),
        ("周转与人力", 3.0, TEAL),
        ("市场与培训", 1.2, GOLD),
        ("预留", 0.8, MUTED),
    ]
    total = sum(value for _, value, _ in items)
    centre = (360, 356)
    radius = 200
    start = -90.0
    draw.text((46, 30), "首期 8 万元怎么花", font=font(26, True), fill=INK)
    for label, value, colour in items:
        sweep = 360.0 * value / total
        draw.pieslice([centre[0] - radius, centre[1] - radius,
                       centre[0] + radius, centre[1] + radius],
                      start, start + sweep, fill=colour, outline=(255, 255, 255), width=4)
        angle = math.radians(start + sweep / 2)
        tx = centre[0] + math.cos(angle) * radius * 0.62
        ty = centre[1] + math.sin(angle) * radius * 0.62
        draw.text((tx, ty), f"{value / total * 100:.0f}%", font=font(22, True),
                  fill=(255, 255, 255), anchor="mm")
        start += sweep

    legend_x, legend_y = 620, 210
    for index, (label, value, colour) in enumerate(items):
        y = legend_y + index * 62
        draw.rounded_rectangle([legend_x, y, legend_x + 26, y + 26], radius=6, fill=colour)
        draw.text((legend_x + 42, y - 2), label, font=font(21), fill=INK)
        draw.text((legend_x + 42, y + 26), f"{value:.1f} 万元", font=font(18), fill=MUTED)
    draw.text((legend_x, legend_y + 4 * 62 + 16),
              "合计 8.0 万元，设备按 3 年折旧计入单点位成本。", font=font(17), fill=MUTED)
    image.save(path)


def main() -> int:
    out = Path(r"成果_实测报告\附件_图表")
    out.mkdir(parents=True, exist_ok=True)
    roadmap(out / "10_技术路线流程图.png")
    gantt(out / "11_两年甘特图.png")
    budget(out / "12_经费结构饼图.png")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

