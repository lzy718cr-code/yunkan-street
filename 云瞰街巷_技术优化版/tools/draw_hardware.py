"""Draw the two deployment schematics used in the hardware note.

Everything is drawn with Pillow and a Chinese TrueType font, because OpenCV's
built-in text renderer cannot draw Chinese and a mostly-English figure in a
Chinese document reads badly.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
INK = (32, 38, 48)
MUTED = (110, 120, 134)
ACCENT = (31, 78, 121)
WARM = (198, 106, 58)
GROUND = (232, 236, 241)
TEAL = (38, 140, 128)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def arrow(draw: ImageDraw.ImageDraw, start, end, colour, width=3, head=12):
    draw.line([start, end], fill=colour, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for sign in (1, -1):
        draw.line([end,
                   (end[0] - head * math.cos(angle + sign * 0.42),
                    end[1] - head * math.sin(angle + sign * 0.42))],
                  fill=colour, width=width)


def dimension(draw, start, end, text, colour=MUTED, anchor="mm"):
    arrow(draw, start, end, colour, 2, 9)
    arrow(draw, end, start, colour, 2, 9)
    if not text:
        return
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    draw.text(mid, text, font=font(22), fill=colour, anchor=anchor)


def section_view(path: Path) -> None:
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    ground_y = 600
    draw.rectangle([0, ground_y, width, height], fill=GROUND)
    draw.line([(0, ground_y), (width, ground_y)], fill=(150, 158, 170), width=3)
    draw.text((60, ground_y + 18), "地面（观察区所在平面）", font=font(24), fill=MUTED)

    # pole and camera
    pole_x = 250
    camera_y = 210
    draw.line([(pole_x, ground_y), (pole_x, camera_y)], fill=(120, 128, 140), width=8)
    draw.rectangle([pole_x - 34, camera_y - 22, pole_x + 34, camera_y + 22],
                   fill=ACCENT, outline=INK, width=2)
    draw.text((pole_x, camera_y - 60), "固定相机", font=font(24, True), fill=ACCENT, anchor="mm")

    near_x, far_x = 520, 1290
    draw.line([(pole_x + 30, camera_y), (near_x, ground_y)], fill=WARM, width=3)
    draw.line([(pole_x + 30, camera_y), (far_x, ground_y)], fill=WARM, width=3)
    draw.polygon([(pole_x + 30, camera_y), (near_x, ground_y), (far_x, ground_y)],
                 fill=(255, 236, 224))

    # angle
    draw.arc([pole_x - 120, camera_y - 120, pole_x + 120, camera_y + 120], 20, 62,
             fill=TEAL, width=3)
    draw.text((pole_x + 132, camera_y + 78), "俯角 θ", font=font(24, True), fill=TEAL)

    # height
    dimension(draw, (pole_x - 90, ground_y), (pole_x - 90, camera_y), "")
    draw.text((pole_x - 108, (ground_y + camera_y) / 2), "H", font=font(26, True),
              fill=MUTED, anchor="rm")
    draw.text((pole_x - 120, (ground_y + camera_y) / 2 + 34), "3～6 m", font=font(21),
              fill=MUTED, anchor="rm")

    # coverage
    dimension(draw, (near_x, ground_y + 52), (far_x, ground_y + 52), "")
    draw.text(((near_x + far_x) / 2, ground_y + 84), "有效观测范围（通常 15～40 m）",
              font=font(23), fill=WARM, anchor="mm")

    # calibration rectangle drawn in perspective on the ground
    quad = [(700, 600), (900, 600), (860, 556), (716, 556)]
    draw.polygon(quad, fill=(255, 255, 255), outline=TEAL)
    for point in quad:
        draw.ellipse([point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7], fill=TEAL)
    centre = ((quad[0][0] + quad[2][0]) / 2, (quad[0][1] + quad[2][1]) / 2)
    draw.ellipse([centre[0] - 9, centre[1] - 9, centre[0] + 9, centre[1] + 9],
                 fill=(245, 200, 90), outline=INK)
    draw.text((quad[1][0] + 18, quad[1][1] + 6),
              "第 5 点：矩形中心（校核）", font=font(21), fill=(150, 110, 20))
    draw.text((quad[3][0] - 40, quad[3][1] - 46),
              "地面标定矩形 4 m × 3 m（实量）", font=font(22, True), fill=TEAL)

    # annotation
    draw.text((40, 30),
              "要点：画面下缘落在观察区近边，上缘略超远边；矩形四个角都要在画面内且无遮挡。",
              font=font(21), fill=MUTED)
    image.save(path)


def plan_view(path: Path) -> None:
    width, height = 1500, 780
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    pass

    area = [(250, 210), (1010, 250), (960, 640), (210, 590)]
    draw.polygon(area, fill=(238, 244, 250), outline=ACCENT, width=3)
    draw.text((300, 232), "观察区（广场 / 出入口 / 停放带）", font=font(23), fill=ACCENT)

    for index, point in enumerate(area, 1):
        draw.ellipse([point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10],
                     fill=TEAL, outline=INK)
        draw.text((point[0] + 16, point[1] - 22), str(index), font=font(24, True), fill=TEAL)

    centre = (sum(p[0] for p in area) / 4, sum(p[1] for p in area) / 4)
    draw.ellipse([centre[0] - 11, centre[1] - 11, centre[0] + 11, centre[1] + 11],
                 fill=(245, 200, 90), outline=INK)
    draw.text((centre[0] + 18, centre[1] - 8), "5 中心校核点", font=font(21), fill=(150, 110, 20))

    camera = (1180, 150)
    draw.polygon([camera, (area[1][0] + 40, area[1][1]), (area[2][0] + 10, area[2][1])],
                 fill=(255, 240, 228))
    draw.ellipse([camera[0] - 16, camera[1] - 16, camera[0] + 16, camera[1] + 16],
                 fill=ACCENT, outline=INK, width=2)
    draw.text((camera[0] - 30, camera[1] + 26), "相机位（二级平台 / 灯杆 / 楼角）",
              font=font(21), fill=ACCENT, anchor="ma")

    draw.text((250, 690),
              "布点原则：① 正对观察区短边，避免逆光；② 前后 30 分钟内机位不动；"
              "③ 画面里同时出现地面、出入口和停放带。",
              font=font(21), fill=MUTED)
    image.save(path)


def main() -> int:
    out = Path(r"成果_实测报告\附件_图表")
    out.mkdir(parents=True, exist_ok=True)
    section_view(out / "07_点位剖面示意.png")
    plan_view(out / "08_点位平面示意.png")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

