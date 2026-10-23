"""Build the observation-hardware note (.docx).

The note is deliberately operational: what to buy, where to put it, how to
calibrate, what to check before leaving the site, and what it costs per point.
Prices are labelled as reference values, not quotations.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


BODY_FONT = "宋体"
HEAD_FONT = "黑体"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
FIG = Path(r"成果_实测报告\附件_图表")


def set_run_font(run, name=BODY_FONT, size=None, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def body(document, text, *, size=10.5, indent=True, space_after=6, bullet=False):
    paragraph = document.add_paragraph(style="List Bullet" if bullet else None)
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    paragraph.paragraph_format.line_spacing = 1.32
    if indent and not bullet:
        paragraph.paragraph_format.first_line_indent = Pt(size * 2)
    run = paragraph.add_run(text)
    set_run_font(run, BODY_FONT, size)
    return paragraph


def heading(document, text, level=1):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(13 if level == 1 else 9)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, HEAD_FONT, {1: 14, 2: 12, 3: 11}[level], bold=True, color=ACCENT)
    return paragraph


def caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(9)
    run = paragraph.add_run(text)
    set_run_font(run, BODY_FONT, 9, color=RGBColor(0x44, 0x44, 0x44))


def shade(cell, hex_color):
    properties = OxmlElement("w:shd")
    properties.set(qn("w:val"), "clear")
    properties.set(qn("w:color"), "auto")
    properties.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(properties)


def table(document, headers, rows, *, widths=None, size=8.8, header_fill="DCE6F1"):
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    created = document.add_table(rows=1, cols=len(headers))
    created.style = "Table Grid"
    created.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, text in enumerate(headers):
        cell = created.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        set_run_font(run, HEAD_FONT, size, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(cell, header_fill)
    for values in rows:
        row = created.add_row()
        OxmlElement("w:cantSplit")
        for index, value in enumerate(values):
            cell = row.cells[index]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(value))
            set_run_font(run, BODY_FONT, size)
    for row in created.rows[:-1]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True
    if widths:
        for row in created.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)
    return created


def figure(document, path: Path, width_cm: float, label: str):
    if not path.is_file():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(document, label)


def build(out: Path) -> Path:
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.4)
    section.top_margin = section.bottom_margin = Cm(2.3)
    style = document.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(8)
    run = title.add_run("观测点位硬件方案")
    set_run_font(run, HEAD_FONT, 20, bold=True, color=ACCENT)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("——一个点位从“到场”到“能出图”需要带什么、怎么摆、怎么验")
    set_run_font(run, BODY_FONT, 11, color=RGBColor(0x44, 0x44, 0x44))
    document.add_paragraph()

    heading(document, "一、为什么要有硬件包", 1)
    body(document, "软件谁都能装，难的是“拍回来的素材能不能用”。我们复盘的三个返工原因都不是算法："
                   "机位被撞动了几厘米、画面里有逆光、地面没有一个能实量的矩形。"
                   "这三件事在软件里无法事后补救——机位一动，前后两段轨迹就不能比；没有实量尺寸，"
                   "所有指标只能停在画面比例。所以把硬件和流程固定下来，是这套方法能不能复制的前提。")
    body(document, "硬件包的目标不是“买贵的设备”，而是三件事：同一机位、可实量、可复现。"
                   "同一机位靠夹持与配平，可实量靠卷尺和地面矩形，可复现靠一张验收清单——"
                   "换人去做，也要拍出能进同一条流水线的片子。")

    heading(document, "二、三档配置", 1)
    body(document, "按点位条件和预算分三档。轻量包用现成手机，标准包是主力，边缘包用于需要现场出结果或没有网络的点位。"
                   "单价为 2026 年参考价，含税费与常规配件，实际采购按当地报价调整。")
    table(document,
          ["档次", "组成", "参考单价", "适用场景", "局限"],
          [["轻量包", "手机夹具、1.6 m 三脚架、20000 mAh 移动电源、128 GB 存储、30 m 卷尺",
            "约 500 元", "临时踏勘、谈判前的快速取证", "画质与稳定性一般；夜景噪点大"],
           ["标准包（主力）", "4K 运动相机、防水壳、大力夹＋魔术臂、2.5 m 立杆夹具、备用电池、"
            "移动电源、256 GB 卡、激光测距仪", "约 2 900 元", "正式点位诊断，按天连续拍摄", "需要夹持点；广角边缘有畸变"],
           ["边缘包", "标准包 ＋ 迷你主机（或 Jetson 级板卡）、4G 路由、PoE 供电防水箱、512 GB SSD",
            "约 8 700 元", "无网络现场出结果；街道自建点位", "成本高；需一次调试"]],
          widths=[2.6, 6.6, 2.2, 3.6, 3.4])
    caption(document, "表 1　三档配置与参考单价")
    body(document, "建议起步只买标准包。轻量包是为了“先去看一眼”，边缘包等有街道或物业愿意自建点位时再上——"
                   "这两档都不影响方法本身。")

    heading(document, "三、机位怎么摆", 1)
    body(document, "一句话原则：站在观察区外，俯视观察区，画面里同时出现地面、出入口和停放带。"
                   "高度 3～6 m，俯角 20°～35°，一台相机覆盖一个观察区；宁可靠近一点拍清楚，"
                   "不要为了拍全而把人都拍成小点。")
    figure(document, FIG / "07_点位剖面示意.png", 16.2,
           "图 A　点位剖面：相机高度、俯角与地面标定矩形")
    figure(document, FIG / "08_点位平面示意.png", 16.2,
           "图 B　平面布置：一个观察区对应一台相机与一组标定点")
    body(document, "四个容易忽略的约束：其一，画面下缘要压在观察区近边，否则近处的人只剩半截；"
                   "其二，正对观察区短边，避免逆光——下午拍摄尤其要避开西向；"
                   "其三，前后 30 分钟内机位不能动，三脚架要配重、夹具要拧紧；"
                   "其四，广角镜头边缘畸变明显，标定矩形尽量放在画面中部三分之一。")

    heading(document, "四、安装与标定六步", 1)
    table(document,
          ["步骤", "做什么", "工具", "产出"],
          [["1 踏勘", "确认观察区范围、人流方向、可用的夹持点与供电条件", "平面图、手机拍照", "点位草图"],
           ["2 架设", "固定相机，调俯角与构图，画面下缘压近边", "三脚架 / 大力夹、配重", "构图确认照"],
           ["3 实量", "在画面中部选一块平整地面，量出矩形边长（如 4 m × 3 m）", "激光测距仪或卷尺", "矩形尺寸记录"],
           ["4 拍摄", "连续录制，单段不超过 30 分钟；开机后不再触碰机身", "相机、备用电池", "视频文件"],
           ["5 标定", "在软件里依次点左下、右下、右上、左上，再点矩形中心", "软件四点标定", "calibration.json 与中心校核误差"],
           ["6 验收", "按第七部分清单逐项打勾，现场补拍", "验收表", "签字确认"]],
          widths=[2.4, 7.4, 4.0, 3.4])
    caption(document, "表 2　安装与标定六步")
    body(document, "第 5 步的顺序不能反：四个角点必须按“左下→右下→右上→左上”点，"
                   "程序按这个顺序把它们对应到矩形的四个角；第 5 点用来自查，"
                   "中心校核误差超过 10 cm 说明点歪了或有畸变，要重新点。")

    heading(document, "五、供电、存储与天气", 1)
    for text in (
        "续航：4K/30 fps 约 60～80 分钟一块电池。一个点位通常需要 2 块电池 + 一块 20000 mAh 移动电源边充边录。",
        "存储：4K/30 fps 约 350 MB/分钟，30 分钟约 10 GB。按“两个时段 × 30 分钟”准备，一张 256 GB 卡足够。",
        "分段：单段不超过 30 分钟，避免单文件过大和断电丢整段；分段处留 5 秒重叠，便于拼接。",
        "天气：小雨用防水壳，中雨以上改期——雨伞和雨衣会改变人的行为，数据不可比。",
        "光线：阴天最稳；晴天的正午阴影硬，行人容易与阴影粘连；傍晚要防止逆光过曝。",
        "清洁：到场先擦镜头。指纹在画面里就是一片永远不动的“热区”，会直接污染热力图。",
    ):
        body(document, text, bullet=True, space_after=4)

    heading(document, "六、成本与折旧", 1)
    body(document, "计划书里的单点位直接成本按“设备折旧 0.05 万元”计，这个数偏保守："
                   "标准包约 2 900 元，按 3 年、每年 6 个点位轮转，摊到每个点位约 160 元；"
                   "再算上损耗与配件补充，300 元/点位已经足够。保留 500 元的预算，是因为前两年会有返工与丢失。")
    table(document,
          ["项目", "金额（元）", "说明"],
          [["标准包一次性投入", "2 900", "相机、夹具、电源、存储、测距仪"],
           ["年维护与配件", "300", "电池衰减、卡损耗、夹具更换"],
           ["单点位折旧（3 年 × 6 点位/年）", "约 160", "（2900 + 300×3）÷ 18"],
           ["计划书取值", "500", "含返工与丢失裕量"]],
          widths=[6.4, 3.2, 6.2])
    caption(document, "表 3　单点位硬件成本")
    body(document, "不买设备也能起步：先用手机拍一段、用卷尺量一块地面，就能跑完整条流程。"
                   "硬件包解决的是“多个点位、多人执行、结果可比”，不是“没有设备就不能做”。")

    heading(document, "七、现场验收清单", 1)
    table(document,
          ["检查项", "合格标准", "不合格的处理"],
          [["机位稳定", "三脚架/夹具无晃动，轻推机身不位移", "加配重、换夹持点"],
           ["构图", "观察区近边在画面下 1/6，出入口与停放带均入画", "调俯角或后退"],
           ["逆光", "主体无明显过曝，人脸不呈剪影", "换时段或改机位"],
           ["镜头", "画面无指纹、水渍、蛛网", "擦净重拍"],
           ["标定矩形", "四角在画面中部、地面平整、无遮挡", "重选矩形"],
           ["实量尺寸", "两边长已记录到厘米，且与地面特征对得上", "复量"],
           ["录制", "连续无中断，时长覆盖完整时段", "补拍"],
           ["文件", "文件名含点位、日期、时段；已备份一份", "当场改名前备份"]],
          widths=[3.2, 8.4, 5.4])
    caption(document, "表 4　现场验收清单")

    heading(document, "八、与软件的接口", 1)
    body(document, "硬件只负责“拍对”，软件负责“算对”。流程是：把视频拷回电脑，在软件里打开，"
                   "填点位与采集起点，需要米制时做四点标定，然后开始分析；"
                   "分析结束会自动生成空间画像，结果全部落在会话目录里。")
    body(document, "一个容易踩的坑：手机或相机默认会写 HEVC/H.265 编码，部分机器上解码会缺帧。"
                   "现场用 H.264 录制，或者回来先转一遍，能省掉大量“为什么帧数对不上”的排查时间。")
    body(document, "另一个坑是文件名带空格或特殊符号。软件支持中文路径，但批量处理脚本更容易在空格上出错，"
                   "建议统一命名：点位_日期_时段.mp4。")

    note = document.add_table(rows=1, cols=1)
    note.style = "Table Grid"
    cell = note.rows[0].cells[0]
    cell.text = ""
    run = cell.paragraphs[0].add_run(
        "本方案的作用是让“换一个点位、换一个人”也能拍出可比较的素材。"
        "价格与选型会随市场变化，方法与验收标准不会。")
    set_run_font(run, BODY_FONT, 9, color=RGBColor(0x40, 0x40, 0x40))
    shade(cell, "F4F6FA")

    out.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(out))
    return out


def main() -> int:
    path = build(Path(r"成果_实测报告\观测点位硬件方案.docx"))
    print("已生成：", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
