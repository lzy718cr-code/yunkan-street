"""Build the measured space-use report (.docx) from a portrait folder.

Every number is read from ``metrics.json`` so the document can be regenerated
after a re-run without hand-copying figures. The narrative is written once, but
the tables, captions and inline values follow the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


BODY_FONT = "宋体"
HEAD_FONT = "黑体"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def set_run_font(run, name=BODY_FONT, size=None, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def body(document, text, *, size=10.5, first_indent=True, space_after=6, align=None):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    paragraph.paragraph_format.line_spacing = 1.35
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Pt(size * 2)
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    set_run_font(run, BODY_FONT, size)
    return paragraph


def heading(document, text, level=1):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, HEAD_FONT, {1: 15, 2: 12.5, 3: 11}[level], bold=True, color=ACCENT)
    return paragraph


def caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run(text)
    set_run_font(run, BODY_FONT, 9, color=RGBColor(0x44, 0x44, 0x44))
    return paragraph


def shade(cell, hex_color):
    properties = OxmlElement("w:shd")
    properties.set(qn("w:val"), "clear")
    properties.set(qn("w:color"), "auto")
    properties.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(properties)


def no_split(row):
    properties = OxmlElement("w:cantSplit")
    row._tr.get_or_add_trPr().append(properties)


def table(document, headers, rows, *, widths=None, size=9, header_fill="DCE6F1"):
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    created = document.add_table(rows=1, cols=len(headers))
    created.style = "Table Grid"
    created.alignment = WD_TABLE_ALIGNMENT.CENTER
    header = created.rows[0]
    for index, text in enumerate(headers):
        cell = header.cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        set_run_font(run, HEAD_FONT, size, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(cell, header_fill)
    no_split(header)
    for values in rows:
        row = created.add_row()
        no_split(row)
        for index, value in enumerate(values):
            cell = row.cells[index]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(value))
            set_run_font(run, BODY_FONT, size)
    if widths:
        for row in created.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)
    # Keep the whole table on one page: a table split across a page break is
    # the single most common defect in a short report.
    for row in created.rows[:-1]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True
    return created


def figure(document, path: Path, width_cm: float, label: str):
    if not path.is_file():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(document, label)


def note_box(document, text):
    created = document.add_table(rows=1, cols=1)
    created.style = "Table Grid"
    cell = created.rows[0].cells[0]
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    set_run_font(run, BODY_FONT, 9, color=RGBColor(0x40, 0x40, 0x40))
    shade(cell, "F4F6FA")
    document.add_paragraph().paragraph_format.space_after = Pt(0)


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------

def build(space: Path, out: Path, *, screenshot: Path | None = None,
          dialog: Path | None = None) -> Path:
    metrics = json.loads((space / "metrics.json").read_text(encoding="utf-8"))
    bins = metrics.get("time_bins") or []
    span = (max(item["start_seconds"] for item in bins) + 60) if bins else 600
    minutes = int(round(span / 60))
    classes = metrics.get("class_breakdown") or {}
    near = metrics.get("near_miss") or {}
    parked = metrics.get("parked_vehicles") or {}
    pedestrians = classes.get("person", {})

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.4)
    section.top_margin = section.bottom_margin = Cm(2.4)

    style = document.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    # ---- title block ----
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(60)
    run = title.add_run("小空间人群使用画像")
    set_run_font(run, HEAD_FONT, 26, bold=True, color=ACCENT)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("——基于现有视频分析工具的一次再开发与实测")
    set_run_font(run, HEAD_FONT, 13, color=RGBColor(0x44, 0x44, 0x44))

    for text in (
        "观测点位：校园商业街角（便利店门前广场及非机动车停放带）",
        f"观测时长：{minutes} 分钟（{span:.0f} 秒，{int(span * 30)} 帧）",
        "素材性质：既有固定机位拍摄，非摆拍；无地面标定",
        "生成方式：由分析软件输出的 metrics.json 自动排版",
    ):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(text)
        set_run_font(run, BODY_FONT, 10.5)

    document.add_page_break()

    # ---- 1 ----
    heading(document, "一、为什么把对象从“车”换成“人”", 1)
    body(document, "现有软件回答的是“过去了几辆、什么车、在哪条线上过去”，输出是一组计数。"
                   "这个口径在路段和路口是有效的，但放到一个几十米见方的公共空间里，它回答不了设计真正关心的问题："
                   "人愿不愿意停下来、停在哪、待多久、和非机动车在什么地方互相让路。"
                   "而且校园里几乎没有机动车，如果继续以车为对象，画面里大部分活动都会被丢掉。")
    body(document, "因此这次没有再训练任何模型。检测与跟踪继续使用通用权重，"
                   "真正被替换的是“记录什么”和“算什么”：把每一帧的检测结果原样落盘，"
                   "再在这一层数据上重建轨迹、算出画像指标。这样做的直接好处是——"
                   "同一份录像可以在不改代码的情况下反复重算，换一个点位也不需要重新标注。")

    # ---- 2 ----
    heading(document, "二、素材与点位", 1)
    table(document,
          ["项目", "内容"],
          [["相机", "运动相机，固定机位，二层高度俯视，广角"],
           ["原始规格", "3840×2160，29.97 fps，30 分钟"],
           ["机位稳定性", "首尾帧配准位移 0.1 像素，旋转 0.0°"],
           ["分析规格", "缩放至 1920×1080 后分析，逐帧处理"],
           ["观测时段", "原片第 4:00–14:00，共 10 分钟"],
           ["场景构成", "便利店门前广场、两个出入口、自行车与电动车停放带、东西向穿行带"]],
          widths=[3.6, 12.4])
    caption(document, "表 1　观测素材与点位")
    body(document, "机位稳定性这一项值得单独说明：30 分钟内首尾帧的配准位移只有 0.1 像素、旋转为 0，"
                   "说明这是一台真正意义上的固定机位。固定机位是后续所有工作的前提——"
                   "同一个画面坐标在不同时刻指的是同一个地面位置，停留、占用、流向才有意义。"
                   "代价是，这次没有做地面标定，所以下文所有距离与面积指标都以“画面宽比例”表达，不做米制换算。")

    # ---- 3 ----
    heading(document, "三、方法：从“检出”到“画像”的四步", 1)
    heading(document, "3.1　检测与跟踪：通用权重，不重训", 2)
    body(document, "只取 COCO 通用权重中的行人、自行车、电动车（摩托车）三类，跟踪沿用 ByteTrack。"
                   "选择通用权重而不是自训练权重，是基于实拍对比：在同样一段校园素材上，"
                   "自训练的多类别权重对远处行人的召回明显更低，而通用权重虽然类别更粗，"
                   "但“人”这一类足够稳。更重要的是，通用权重不依赖本项目的数据，"
                   "换一个点位、换一台机器，不需要重新准备训练集。")
    heading(document, "3.2　逐帧落盘", 2)
    body(document, "每一帧、每一个被跟踪到的目标写一行：帧号、源时间戳、跟踪 ID、类别、置信度、检测框、脚点归一化坐标；"
                   "另外每一帧写一行画面尺寸与当帧参数。这一步是整套东西的地基——"
                   "后面的任何指标都可以从这张表重算，而原始画面不必保存。")
    heading(document, "3.3　轨迹重建", 2)
    body(document, f"跟踪器在短时遮挡后会给出新的 ID。用“时间上相邻、空间上接近”的规则把断掉的轨迹接回来："
                   f"本次原始轨迹 {metrics['tracks_raw']} 条，拼接后 {metrics['tracks_after_stitch']} 条，"
                   f"合并了 {metrics['merged_fragments']} 段。拼接只修复相邻断轨，不保证消除全部 ID 切换，"
                   f"这个限制在第五部分会再说明。")
    heading(document, "3.4　画像指标", 2)
    body(document, "存在强度用人·秒表达：一条轨迹存在多久，就把多久累加到它脚点所在的格子上。"
                   "停留按相邻观测之间的速度判定——低于阈值即视为停留，并据此计算停留段占比。"
                   "流向把轨迹段按网格求平均速度矢量。交织统计“运动中的非机动车与行人”在同一时间桶内脚点距离小于阈值的组合。"
                   "停放则找出长时间几乎不动的非机动车轨迹。")

    # ---- 4 ----
    heading(document, "四、结果", 1)
    heading(document, "4.1　总览", 2)
    table(document,
          ["指标", "数值", "口径"],
          [["观测时长", f"{minutes} 分钟（{span:.0f} 秒）", "连续观测窗口"],
           ["检测记录", f"{metrics['detections_considered']:,} 行", "每帧每目标一行"],
           ["行人轨迹", f"{metrics.get('pedestrian_tracks', 0):,} 条", "拼接后，含断轨修复"],
           ["累计存在", f"{metrics['person_seconds']:,.0f} 人·秒", "仅行人"],
           ["平均 / 中位停留", f"{metrics['mean_dwell_seconds']:.1f} / {metrics['median_dwell_seconds']:.1f} 秒", "单条轨迹可见时长"],
           ["最长停留", f"{metrics['max_dwell_seconds']:.1f} 秒", "单条轨迹"],
           ["停留段占比", f"{metrics['stationary_ratio'] * 100:.1f}%", "速度低于阈值的观测段占比"],
           ["占用网格", f"{metrics['occupied_cells']} 个", "0.02 画面宽比例网格"],
           ["近距人车交织", f"{near.get('events', 0)} 次", f"阈值 {near.get('threshold_frame_fraction', 0)} 画面宽"],
           ["静止非机动车", f"{parked.get('tracks', 0)} 段 / {parked.get('vehicle_seconds', 0):.0f} 秒",
            f"分布在 {parked.get('distinct_places', 0)} 处"]],
          widths=[4.2, 6.0, 5.8])
    caption(document, "表 2　核心指标")
    body(document, f"平均停留 {metrics['mean_dwell_seconds']:.1f} 秒、中位 {metrics['median_dwell_seconds']:.1f} 秒，"
                   f"说明这个街角以“短时经过”为主；但停留段占比达到 {metrics['stationary_ratio'] * 100:.0f}%，"
                   f"说明停留并不是个别现象，而是这个空间稳定的一部分使用方式。"
                   f"只看通过人数会完全丢掉后一半信息。")

    heading(document, "4.2　空间使用强度", 2)
    figure(document, space / "heatmap.jpg", 16.0,
           "图 1　空间使用强度（人·秒）。暖色表示单位面积累计停留时间更长")
    body(document, "强度最高的带状区沿便利店门前展开，并向右侧延伸到非机动车通行带。"
                   "左侧自行车停放带本身也形成一条独立的高值带——那不是“人喜欢站在自行车堆里”，"
                   "而是取车、停车、挪车的动作让脚点在同一小片区域反复出现。"
                   "这一点很关键：热力高不等于“应该在这里做停留空间”，要先分清是停留、是穿行、还是操作。")

    heading(document, "4.3　停留与穿行的分离", 2)
    figure(document, space / "heatmap_dwell.jpg", 16.0,
           "图 2　仅统计停留段的空间分布。与图 1 的差异即“通过”与“停留”的分界")
    body(document, "把所有停留段单独累加后，高值区收缩到便利店门前与两处花池、座椅边缘，"
                   "穿行带上不再有成片高值。图 1 减去图 2 的那部分，就是这个空间里“只是路过”的区域。"
                   "两张图放在一起，才有可能讨论“停留设施该放哪、不该放哪”。")

    heading(document, "4.4　主流向", 2)
    figure(document, space / "flow.jpg", 16.0,
           "图 3　各网格的平均移动方向。箭头长度对应平均速度")
    body(document, "东西向穿越是这个街角的主导动线，便利店门前出现指向出入口的短矢量。"
                   "对设计而言，这意味着任何加在穿越带中段的花池、展架或隔离设施，"
                   "影响的不只是“美观”，而是整条动线的通行效率。")

    heading(document, "4.5　时间节律", 2)
    figure(document, space / "time_curve.png", 16.0,
           "图 4　逐分钟存在强度与轨迹条数")
    table(document,
          ["分钟", "人·秒", "轨迹条数"],
          [[f"{int(item['start_seconds'] // 60) + 1}", f"{item['person_seconds']:.0f}", f"{item['track_count']}"]
           for item in bins],
          widths=[3.0, 5.0, 5.0], size=9)
    caption(document, "表 3　逐分钟分箱")
    if bins:
        def average(item):
            return item["person_seconds"] / max(1, item["track_count"])
        busiest = max(bins, key=lambda item: item["track_count"])
        longest = max(bins, key=average)
        body(document, f"存在强度在 {min(item['person_seconds'] for item in bins):.0f}–"
                       f"{max(item['person_seconds'] for item in bins):.0f} 人·秒之间波动，"
                       f"轨迹条数在 {min(item['track_count'] for item in bins)}–"
                       f"{max(item['track_count'] for item in bins)} 条之间。"
                       f"两者并不同步：第 {int(busiest['start_seconds'] // 60) + 1} 分钟轨迹条数最多"
                       f"（{busiest['track_count']} 条），但人均只在 {average(busiest):.1f} 秒；"
                       f"第 {int(longest['start_seconds'] // 60) + 1} 分钟轨迹只有 {longest['track_count']} 条，"
                       f"人均却达到 {average(longest):.1f} 秒。"
                       f"“人多”和“待得久”是两件事——这正是只看计数会丢掉的信息。")

    heading(document, "4.6　与非机动车的交织", 2)
    body(document, f"以“运动中的非机动车 + 行人脚点距离小于 {near.get('threshold_frame_fraction', 0)} 画面宽”为条件，"
                   f"共筛出 {near.get('events', 0)} 组近距事件，涉及行人 {near.get('pedestrians_involved', 0)} 位、"
                   f"非机动车 {near.get('vehicles_involved', 0)} 辆，最长的一次持续 {near.get('longest_overlap_seconds', 0):.1f} 秒。"
                   f"判断时加了“非机动车必须在运动”这一条：停着的电动车旁边站一群人，那是停放，不是交织。")
    note_box(document, "口径提醒：这不是“险情次数”，而是“人车在同一时刻进入了同一小块地面”的组数。"
                       "它的用途是定位需要回看复核的点位，不能当作安全评价。")
    body(document, "从分布上看，交织集中在穿越带与非机动车通行带的重叠段，"
                   "也就是便利店门前向右延伸、人流与车流共用同一片地面的地方。"
                   "这为“把减速与让行提示放在这一段、而不是全场铺开”提供了依据。")

    heading(document, "4.7　非机动车停放占用", 2)
    body(document, f"长时间几乎不动的非机动车轨迹共 {parked.get('tracks', 0)} 段、累计 {parked.get('vehicle_seconds', 0):.0f} 秒，"
                   f"集中在 {parked.get('distinct_places', 0)} 处。数量不算大，但停放带在画面里已经压到人行边界，"
                   f"与 4.2 中左侧那条独立高值带是同一现象的两面："
                   f"停放越往北挤，取车的人就越往穿行带里站。")

    # ---- 5 ----
    heading(document, "五、口径与可信边界", 1)
    body(document, "下表列出六类口径问题，以及本次对每一类的具体处理。")
    table(document,
          ["问题", "本次处理", "对结论的影响"],
          [["没有地面标定", "全部用画面宽比例，不出米制密度与速度", "不能直接与规范阈值比较；补一次矩形量测即可升级"],
           ["跟踪 ID 断裂", "拼接减少但不消除", f"{metrics.get('pedestrian_tracks', 0)} 条轨迹不等于同等数量的人，只能做相对比较"],
           ["停留不等于意愿", "只报“会话内连续可见时长”", "不能据此推断“愿意待多久”"],
           ["单机位遮挡", "花池、灯杆、人群互相遮挡", "密集时段人数被低估"],
           ["年龄与性别", "未纳入任何结论", "只在可见人脸子集上可能，样本偏差大，不进入判断"],
           ["交织口径", "距离筛查，非冲突判定", "只能用于定位复核点，不能作为安全评价"]],
          widths=[3.4, 6.2, 6.4], size=8.5)
    caption(document, "表 4　口径与局限")
    body(document, "其中“年龄与性别”这一条需要单独说：它看起来是加分项，实际上是最容易出错的一项。"
                   "俯视机位下绝大多数人背对或侧对镜头，可见人脸的比例很低，"
                   "在这个子集上统计出的比例不能外推到全体。"
                   "因此这次只把它列进能力清单，不写进任何结论。")

    # ---- 6 ----
    heading(document, "六、三条可用结论", 1)
    body(document, "以下三条都标注了证据强度与待验证项，可以直接作为设计讨论的输入，但不要单独引用数字。")

    heading(document, "结论一　使用由穿越主导，停留集中在门前一条窄带（证据强）", 3)
    body(document, f"证据：图 1 与图 2 的对比；停留段占比 {metrics['stationary_ratio'] * 100:.0f}%，"
                   f"但停留高值区面积远小于存在高值区面积。")
    body(document, "可用：任何改造都应以“不打断东西向穿越”为前提；停留设施应压在门前带，而不是放在穿越带中间。")
    body(document, "待验证：需要一次地面标定，把“窄带”换算成米，才能判断它是否同时容得下停留与通行。")

    heading(document, "结论二　人车交织集中在穿越带与通行带的重叠段（证据中）", 3)
    body(document, f"证据：{near.get('events', 0)} 组近距事件，涉及 {near.get('pedestrians_involved', 0)} 位行人、"
                   f"{near.get('vehicles_involved', 0)} 辆非机动车，分布位置集中在门前向右延伸的共用段。")
    body(document, "可用：优先在这一段设置减速与让行提示，而不是全场铺设减速设施。")
    body(document, "待验证：需要连续多个时段的数据，才能判断这是全天特征还是某个时段的偶发。")

    heading(document, "结论三　非机动车停放已贴着人行边界生长（证据中）", 3)
    body(document, f"证据：{parked.get('distinct_places', 0)} 处静止聚集、累计 {parked.get('vehicle_seconds', 0):.0f} 秒；"
                   f"左侧停放带在存在热力上形成独立高值带。")
    body(document, "可用：把停放带边界与门前停留带一起重画，比单纯增加车位数更有效。")
    body(document, "待验证：需要一次现场停放数量核对，确认视频中的静止轨迹没有把“临时停车”误判为“长期停放”。")

    # ---- 7 ----
    heading(document, "七、局限与下一步", 1)
    for text in (
        "补一次地面矩形量测（例如门前铺装的一块 4 m×3 m 区域），即可把全部指标升级为米制，"
        "并直接与人均面积、通道宽度等常用指标对话。",
        "把观测框换成贴合地面的多边形遮罩，可以进一步去掉立面与树冠上的渗色；"
        "本报告已经用观测框做过一次遮罩，属于粗粒度处理。",
        "多时段（早、中、晚各一段）能把“节律”从一个 10 分钟窗口扩展到一天，"
        "这也是判断峰值是否需要管理措施的前提。",
        "与公开行人数据集的交叉校验，用来量化“通用权重在本场景下的召回水平”，"
        "这比再训练一个模型更能说明方法的可迁移性。",
    ):
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(text)
        set_run_font(run, BODY_FONT, 10.5)

    # ---- 8 ----
    heading(document, "八、复现方式", 1)
    body(document, "软件为单文件可执行程序，本地运行，不上传任何画面。复现路径有两条：")
    body(document, "界面：打开程序 → 载入随包分发的示例片段（或自选视频）→ 点击“开始分析”→ "
                   "分析结束后自动生成空间画像，弹出热力图与关键指标。", first_indent=False)
    body(document, "命令行：对任意已记录的会话目录执行后处理，可指定播放片段长度、遮罩多边形与分类。",
         first_indent=False)
    note_box(document, "python tools/space_pipeline.py <会话目录> --video <原视频> --playback 24")
    body(document, "输出包括：轨迹表 tracks.csv、指标 metrics.json、三张图（热力、停留热力、流向）、"
                   "时间曲线、轨迹回放视频 playback.mp4，以及一份可直接贴进汇报的 summary.md。")

    if screenshot is not None and screenshot.is_file():
        heading(document, "附　软件界面", 1)
        figure(document, screenshot, 15.5, "图 5　分析软件运行界面（示例片段，人行观测模式）")
        body(document, "界面左侧为实时画面与数量曲线，右侧为场景模式、点位、标定与观测框设置。"
                       "人行观测为默认模式，直接使用通用行人权重；标定与观测框均可选，"
                       "不标定也能出结果，只是单位退化为画面比例。分析结束后会自动生成空间画像，"
                       "顶部“空间画像”按钮可随时重算。")
        note_box(document, "每次会话在 space/ 下写入：tracks.csv（轨迹表）、metrics.json（全部指标）、"
                           "heatmap.jpg（存在强度）、heatmap_dwell.jpg（停留）、flow.jpg（流向）、"
                           "time_curve.png（节律）、playback.mp4（回放）与 summary.md（文字摘要）。"
                           "底图 poster.jpg 在会话开始时自动保存，因此原视频移走后仍可重算。")
        if dialog is not None and dialog.is_file():
            figure(document, dialog, 15.5, "图 6　分析结束后自动弹出的空间画像面板")
            body(document, "面板左侧是这一次观测的热力图，右侧是读者会追问的九个数字，"
                           "底部写明结果目录。所有数值都能在该目录的 metrics.json 里查到，"
                           "不需要回到软件里截图取证。")

    out.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="由 metrics.json 生成空间使用画像实测报告")
    parser.add_argument("--space", type=Path, required=True, help="包含 metrics.json 与图片的目录")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--dialog", type=Path, default=None, help="空间画像面板截图")
    args = parser.parse_args()
    path = build(args.space.resolve(), args.out.resolve(),
                 screenshot=args.screenshot.resolve() if args.screenshot else None,
                 dialog=args.dialog.resolve() if args.dialog else None)
    print(f"已生成：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
