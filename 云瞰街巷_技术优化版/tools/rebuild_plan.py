"""Rebuild 附件2 as one coherent 3.0 document.

The text is carried over block by block from ``_plan_dump.json`` so nothing is
re-typed and nothing is lost. What changes is the structure: real Word heading
styles (so an automatic table of contents works), running header, page numbers,
a new hardware section, and figure/table numbering that follows the new order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT,
                            WD_TAB_LEADER)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


DUMP = Path(r"_plan_dump.json")
OUT = Path(r"材料_附件2技术完善版\附件2：创业计划书_20261022.docx")
FIG = Path(r"成果_实测报告\附件_图表")
BODY_FONT = "宋体"
HEAD_FONT = "黑体"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)

IMAGES = [
    FIG / "05_软件界面.png",
    FIG / "06_画像面板.png",
    FIG / "01_存在强度热力图.jpg",
    FIG / "03_主流向场.jpg",
    FIG / "07_点位剖面示意.png",
    FIG / "08_点位平面示意.png",
    FIG / "10_技术路线流程图.png",
    FIG / "11_两年甘特图.png",
    FIG / "12_经费结构饼图.png",
]

# Figure captions in document order; numbers are assigned as they are emitted,
# so inserting a figure never leaves a stale number behind.
FIGURE_TEXT = {
    6: "图 1　从视频到规划判断的六步流水线",
    7: "图 2　两年推进节奏",
    8: "图 3　首期 8 万元资金结构",
}

# Text replacements keyed by dump index.
OVERRIDES: dict[int, str] = {
    38: ("车辆观测能力（四类机动车检测、ByteTrack 跟踪、方向过线、区域连续驻留、逐帧 CSV 与参数快照）"
         "在 2.1 版已交付；3.0 版在其之上增加了人行观测与空间画像层，自动化测试 63 项通过，"
         "并有打包运行验证记录。"),
    151: ("团队由城乡规划、计算机视觉与软件工程背景成员组成，按“规划提问题、技术做工具、"
          "调研验结果”的方式协作。下表列出分工，姓名与联系方式见报名表。"),
    48: ("这条路线有三个刻意的取舍。其一，先把“人”看懂，再谈“车”：车辆模块保留，"
         "用于临停占用和人车交织的辅助判断，不再作为主线。其二，不把“识别”当成果："
         "识别只是第二步，真正的成果是后面的轨迹、指标与设计判断。其三，每条结论都标注"
         "证据强度：哪些是实测、哪些是估算、哪些还不能说，在报告里逐条写清楚，"
         "不用检测数量替代空间品质判断。"),
    155: ("指导教师：王一睿（讲师，硕士生导师，研究方向为城市更新、空间感知与全龄友好城市）、"
          "张睿婕（副教授，硕士生导师，研究方向为城市更新与智慧城市）。两位老师主要指导"
          "规划问题定义、案例选择与成果表达；本项目所用方法、数据与结论由团队独立完成。"),
    147: ("公司尚未注册，拟以团队形式起步，依托长安大学建筑学院，办公地点设在西安。"
          "注册主体、注册地址与成立时间待公司注册后填写；在此之前以项目团队名义承接业务、签订技术服务合同。"),
    153: "表 9　团队角色与分工",
    179: ("首期资金需求 8 万元：采集与推理设备 3 万元（两套标准采集包 0.58 万元、一套边缘计算包 0.87 万元、"
          "移动工作站 1.4 万元，配置与单价见 3.10），周转与人力 3 万元，市场与培训 1.2 万元，"
          "预留 0.8 万元。"),
}

TABLE_OVERRIDES = {
    152: [
        ["角色", "姓名", "专业背景", "主要职责"],
        ["项目负责人", "刘正一", "城乡规划", "整体统筹、客户对接、报告审定"],
        ["算法负责人", "王梓轩", "计算机 / 自动化", "检测跟踪、标定、指标实现"],
        ["软件开发", "王思懿", "软件工程", "桌面程序、导出、打包与测试"],
        ["现场调研", "王泓森", "城乡规划 / 社会学", "点位踏勘、拍摄、访谈、人工核验"],
        ["运营财务", "鲁兴文", "管理学 / 经济学", "报价、合同、成本与回款"],
    ],
}

# Blocks appended after a given dump index.
INSERT_AFTER = 133
EXTRA_INSERTS = {
    49: [( "image", 6), ("caption", 6)],
    65: [( "image", 7), ("caption", 7)],
    179: [("image", 8), ("caption", 8)],
    208: [
        ("body", "[6] Zhang Y., Sun P., Jiang Y., et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV, 2022."),
        ("body", "[7] Ultralytics. YOLO11 模型与官方文档. https://docs.ultralytics.com, 2024."),
        ("body", "[8] Lin T.-Y., Maire M., Belongie S., et al. Microsoft COCO: Common Objects in Context. ECCV, 2014."),
        ("body", "附录　随附材料清单"),
        ("table", [
            ["序号", "材料", "用途"],
            ["1", "附件1 参赛项目报名表", "报名与团队信息"],
            ["2", "附件2 创业计划书（本文件）", "项目方案与可行性"],
            ["3", "《空间使用画像实测报告》（10 页）", "实测数据、图表与口径说明"],
            ["4", "《观测点位硬件方案》（4 页）", "采集配置、安装标定与验收"],
            ["5", "云瞰街巷软件及使用说明", "可运行成品与操作指引"],
            ["6", "轨迹回放视频（60 秒）", "结论的直观演示"],
            ["7", "软著、专利、获奖证书与采纳证明", "前期成果佐证"],
            ["8", "源码与模型来源说明", "可复现与第三方信息"],
        ]),
    ],
}


def set_font(run, name=BODY_FONT, size=None, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def shade(cell, hex_color):
    properties = OxmlElement("w:shd")
    properties.set(qn("w:val"), "clear")
    properties.set(qn("w:color"), "auto")
    properties.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(properties)


def field(paragraph, instruction, placeholder=""):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instr, separate, text, end):
        run._r.append(element)
    return run


class Builder:
    def __init__(self):
        self.doc = Document()
        self.figure_count = 0
        self._setup()

    def _setup(self):
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.left_margin, section.right_margin = Cm(2.8), Cm(2.6)
        section.top_margin, section.bottom_margin = Cm(2.5), Cm(2.4)
        section.different_first_page_header_footer = True

        normal = self.doc.styles["Normal"]
        normal.font.name = BODY_FONT
        normal.font.size = Pt(10.5)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        normal.paragraph_format.line_spacing = 1.42
        normal.paragraph_format.space_after = Pt(3)

        for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
            style = self.doc.styles[name]
            style.font.name = HEAD_FONT
            style.font.size = Pt(size)
            style.font.bold = True
            style.font.color.rgb = ACCENT
            style.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
            style.paragraph_format.first_line_indent = Pt(0)
            style.paragraph_format.space_before = Pt(14 if name != "Heading 1" else 0)
            style.paragraph_format.space_after = Pt(8)
        self.doc.styles["Heading 1"].paragraph_format.page_break_before = True

        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = header.add_run("云瞰街巷 · 创业计划书")
        set_font(run, BODY_FONT, 8.5, color=RGBColor(0x60, 0x6A, 0x78))
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field(footer, "PAGE", "")
        for run in footer.runs:
            set_font(run, BODY_FONT, 9, color=RGBColor(0x60, 0x6A, 0x78))

    # ---- building blocks -------------------------------------------------
    def cover(self):
        lines = [
            ("长安大学第十七届“创业先锋”挑战赛", 13, False, 26),
            ("创 业 计 划 书", 30, True, 16),
            ("", 10, False, 10),
            ("作品类别：  ☑ G. 现代服务与社会治理", 11, False, 4),
            ("项目名称：云瞰街巷", 13, True, 4),
            ("——面向老城区小微公共空间更新的交通感知与空间诊断系统", 11.5, False, 4),
            ("申报单位：长安大学建筑学院", 11, False, 4),
            ("团队负责人：刘正一", 11, False, 4),
            ("指导教师：王一睿、张睿婕", 11, False, 4),
            ("2026 年 10 月", 11, False, 4),
        ]
        paragraph = self.doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(90)
        for text, size, bold, gap in lines:
            paragraph = self.doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.space_after = Pt(gap)
            run = paragraph.add_run(text)
            set_font(run, HEAD_FONT if bold else BODY_FONT, size, bold=bold,
                     color=ACCENT if bold else RGBColor(0x33, 0x33, 0x33))
        self.doc.add_page_break()

    def contents(self, entries):
        title = self.doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.first_line_indent = Pt(0)
        title.paragraph_format.space_after = Pt(14)
        run = title.add_run("目　　录")
        set_font(run, HEAD_FONT, 18, bold=True, color=ACCENT)
        for level, text, page in entries:
            paragraph = self.doc.add_paragraph()
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.left_indent = Cm(0 if level == 1 else 0.8)
            paragraph.paragraph_format.space_after = Pt(1.5)
            paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            paragraph.paragraph_format.line_spacing = Pt(15.5 if level == 1 else 14)
            paragraph.paragraph_format.tab_stops.add_tab_stop(
                Cm(15.4), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            run = paragraph.add_run(f"{text}\t{page if page else ''}")
            set_font(run, HEAD_FONT if level == 1 else BODY_FONT, 10.5 if level == 1 else 9.5,
                     bold=(level == 1),
                     color=ACCENT if level == 1 else RGBColor(0x33, 0x33, 0x33))
        self.doc.add_page_break()

    def heading(self, text, level):
        paragraph = self.doc.add_paragraph(style=f"Heading {level}")
        paragraph.paragraph_format.first_line_indent = Pt(0)
        run = paragraph.add_run(text)
        set_font(run, HEAD_FONT, {1: 16, 2: 13, 3: 11.5}[level], bold=True, color=ACCENT)

    def body(self, text, bold=False):
        paragraph = self.doc.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Pt(21)
        run = paragraph.add_run(text)
        set_font(run, BODY_FONT, 10.5, bold=bold)

    def caption(self, text):
        match = re.match(r"^图\s*(\d+)(.*)$", text)
        if match:
            self.figure_count += 1
            text = f"图 {self.figure_count}{match.group(2)}"
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(10)
        run = paragraph.add_run(text)
        set_font(run, BODY_FONT, 9, color=RGBColor(0x44, 0x44, 0x44))

    def image(self, path: Path, width_cm=14.6):
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.paragraph_format.space_before = Pt(6)
        paragraph.paragraph_format.keep_with_next = True
        paragraph.add_run().add_picture(str(path), width=Cm(width_cm))

    def table(self, rows):
        columns = max(len(row) for row in rows)
        created = self.doc.add_table(rows=0, cols=columns)
        created.style = "Table Grid"
        created.alignment = WD_TABLE_ALIGNMENT.CENTER
        for index, values in enumerate(rows):
            row = created.add_row()
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for column in range(columns):
                cell = row.cells[column]
                cell.text = ""
                text = values[column] if column < len(values) else ""
                run = cell.paragraphs[0].add_run(text)
                set_font(run, HEAD_FONT if index == 0 else BODY_FONT, 8.8,
                         bold=index == 0)
                cell.paragraphs[0].paragraph_format.first_line_indent = Pt(0)
                cell.paragraphs[0].paragraph_format.line_spacing = 1.05
                cell.paragraphs[0].paragraph_format.space_after = Pt(0)
                if index == 0:
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    shade(cell, "DCE6F1")
        trPr = created.rows[0]._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:tblHeader"))
        for row in created.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True
        return created


def classify(text: str) -> tuple[str, int]:
    if text == "参考资料":
        return "heading", 1
    if re.match(r"^第[一二三四五六七八九十]+章", text):
        return "heading", 1
    if re.match(r"^\d+\.\d+\.\d+[\s　]", text):
        return "heading", 3
    match = re.match(r"^(\d+\.\d+)[\s　](.+)$", text)
    # "3.0 版已经把对象…" also starts with a decimal, so a section title must
    # additionally be short and free of sentence punctuation.
    if match and len(text) <= 26 and not re.search(r"[。；：！？]", match.group(2)):
        return "heading", 2
    if re.match(r"^(表|图)\s*\d", text):
        return "caption", 0
    return "body", 0


def renumber(text: str) -> str:
    match = re.match(r"^表\s*(\d+)(.*)$", text)
    if match and int(match.group(1)) >= 8:
        return f"表 {int(match.group(1)) + 1}{match.group(2)}"
    return text


HARDWARE = [
    ("heading", "3.10 硬件与现场作业"),
    ("body", "一个点位能不能出可用数据，一半取决于软件，另一半取决于现场怎么拍。我们复盘的返工原因都不是算法："
             "机位被撞动了几厘米、画面里有逆光、地面没有一个能实量的矩形。机位一动，前后两段轨迹就不能比；"
             "没有实量尺寸，所有指标只能停在画面比例。所以硬件和作业流程要一起固定下来。"),
    ("body", "按点位条件和预算分三档。轻量包约 500 元（手机夹具、三脚架、移动电源、存储卡、卷尺），"
             "用于临时踏勘和谈判前的快速取证；标准包约 2 900 元（4K 运动相机、防水壳、大力夹与立杆夹具、"
             "备用电池、移动电源、256 GB 存储卡、激光测距仪），是正式点位的主力；边缘包约 8 700 元，"
             "在标准包基础上加迷你主机、4G 路由与 PoE 供电防水箱，用于没有网络、需要现场出结果的点位。"),
    ("image", 4),
    ("caption", "图 5　点位剖面：相机高度、俯角与地面标定矩形"),
    ("image", 5),
    ("caption", "图 6　平面布置：一个观察区对应一台相机与一组标定点"),
    ("body", "机位三句话：高度 3～6 米，俯角 20～35 度，一台相机覆盖一个观察区；画面下缘压在观察区近边，"
             "画面里同时出现地面、出入口和停放带；正对观察区短边，避开逆光。"),
    ("table", [
        ["档次", "组成", "参考单价", "适用场景"],
        ["轻量包", "手机夹具、三脚架、移动电源、存储卡、卷尺", "约 500 元", "临时踏勘、快速取证"],
        ["标准包（主力）", "4K 运动相机、防水壳、夹具、备用电池、256 GB 卡、激光测距仪", "约 2 900 元",
         "正式点位诊断，连续拍摄"],
        ["边缘包", "标准包 + 迷你主机、4G 路由、PoE 防水箱、512 GB SSD", "约 8 700 元",
         "无网络现场出结果；街道自建点位"],
    ]),
    ("caption", "表 8　采集硬件三档配置与参考单价"),
    ("body", "安装与标定六步：踏勘、架设、实量、拍摄、标定、验收。第 5 步的顺序不能反——四个角点必须按"
             "“左下、右下、右上、左上”的顺序点，程序按这个顺序把它们对应到矩形的四个角；第 5 点用来自查，"
             "中心校核误差超过 10 厘米说明点歪了或有畸变，要重新点。"),
    ("body", "现场还有两个容易忽略的坑。一是编码：手机和相机默认写 H.265，部分机器解码会缺帧，"
             "统一用 H.264 录制或回来先转一遍。二是命名：文件名统一为“点位_日期_时段.mp4”，"
             "避免批量处理时在空格和特殊符号上出错。"),
    ("body", "硬件的价值不在设备本身，而在“换一个点位、换一个人”也能拍出可比较的素材。"
             "设备价格随市场变化，验收标准不变。"),
]


def heading_list(blocks) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for index, block in enumerate(blocks):
        if index < 32 or block["type"] != "text":
            continue
        text = block["text"]
        if text == "参考资料" or re.match(r"^第[一二三四五六七八九十]+章", text):
            if text.startswith("第四章"):
                headings.append((2, "3.10 硬件与现场作业"))
            headings.append((1, text))
        elif re.match(r"^\d+\.\d+\.\d+[\s　]", text):
            continue
        else:
            match = re.match(r"^(\d+\.\d+)[\s　](.+)$", text)
            if match and len(text) <= 26 and not re.search(r"[。；：！？]", match.group(2)):
                headings.append((2, text))
    return headings


def main() -> int:
    blocks = json.loads(DUMP.read_text(encoding="utf-8"))
    headings = heading_list(blocks)
    pages: dict[str, int] = {}
    if Path("_toc.json").is_file():
        for item in json.loads(Path("_toc.json").read_text(encoding="utf-8")):
            pages[item["text"]] = item["page"]

    builder = Builder()
    builder.cover()
    builder.contents([(level, text, pages.get(text, 0)) for level, text in headings])

    image_index = 0
    for index, block in enumerate(blocks):
        if index < 32:
            continue
        kind = block["type"]
        if kind == "blank":
            continue
        if kind == "table":
            builder.table(TABLE_OVERRIDES.get(index, block["rows"]))
            continue
        if kind == "image":
            if image_index < 4:
                builder.image(IMAGES[image_index])
                image_index += 1
            continue
        text = OVERRIDES.get(index, block["text"])
        role, level = classify(text)
        if role == "heading":
            builder.heading(text, level)
        elif role == "caption":
            builder.caption(renumber(text))
        else:
            builder.body(text, bold=block.get("bold", False))

        if index in EXTRA_INSERTS:
            for kind_name, payload in EXTRA_INSERTS[index]:
                if kind_name == "image":
                    builder.image(IMAGES[payload], 15.0 if payload != 8 else 12.4)
                elif kind_name == "caption":
                    builder.caption(FIGURE_TEXT[payload])
                elif kind_name == "body":
                    bold = payload.startswith("附录")
                    builder.body(payload, bold=bold)
                elif kind_name == "table":
                    builder.table(payload)
        if index == INSERT_AFTER:
            for item in HARDWARE:
                if item[0] == "heading":
                    builder.heading(item[1], 3)
                elif item[0] == "body":
                    builder.body(item[1])
                elif item[0] == "caption":
                    builder.caption(item[1])
                elif item[0] == "image":
                    builder.image(IMAGES[item[1]])
                elif item[0] == "table":
                    builder.table(item[1])

    # ask Word to refresh the table of contents on open
    settings = builder.doc.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)

    builder.doc.save(str(OUT))
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

