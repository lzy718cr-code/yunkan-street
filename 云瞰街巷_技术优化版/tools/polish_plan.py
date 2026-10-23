"""Editorial pass on 附件2: drop the stale 2.2 wording and add real figures.

The plan still described the person module as future work and used the old
ten-class weight; it also had no images at all. This script fixes the wording
and inserts four screenshots/figures produced by the delivered build.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph


DOC = Path(r"材料_附件2技术完善版\附件2：创业计划书_20261022.docx")
FIG = Path(r"成果_实测报告\附件_图表")

FIGURES = {
    "ui": FIG / "05_软件界面.png",
    "dialog": FIG / "06_画像面板.png",
    "heat": FIG / "01_存在强度热力图.jpg",
    "flow": FIG / "03_主流向场.jpg",
}


def replace_text(paragraph: Paragraph, text: str) -> None:
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for run in runs[1:]:
        run._element.getparent().remove(run._element)


def find(document: Document, prefix: str) -> Paragraph:
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise KeyError(prefix)


class Inserter:
    """Append paragraphs after an anchor, reusing an existing paragraph's format."""

    def __init__(self, body_template: Paragraph, caption_template: Paragraph):
        self.body_template = body_template
        self.caption_template = caption_template

    def _new(self, anchor: Paragraph, template: Paragraph) -> Paragraph:
        element = deepcopy(template._p)
        anchor._p.addnext(element)
        created = Paragraph(element, anchor._parent)
        for run in list(created.runs):
            run._element.getparent().remove(run._element)
        return created

    def text(self, anchor: Paragraph, text: str) -> Paragraph:
        created = self._new(anchor, self.body_template)
        created.add_run(text)
        return created

    def picture(self, anchor: Paragraph, path: Path, width_cm: float = 14.6) -> Paragraph:
        created = self._new(anchor, self.body_template)
        created.alignment = WD_ALIGN_PARAGRAPH.CENTER
        created.paragraph_format.first_line_indent = Pt(0)
        created.add_run().add_picture(str(path), width=Cm(width_cm))
        return created

    def caption(self, anchor: Paragraph, text: str) -> Paragraph:
        created = self._new(anchor, self.caption_template)
        created.alignment = WD_ALIGN_PARAGRAPH.CENTER
        created.add_run(text)
        return created


def main() -> int:
    document = Document(str(DOC))
    paragraphs = document.paragraphs
    body_template = paragraphs[39]
    caption_template = paragraphs[42]
    insert = Inserter(body_template, caption_template)

    # ---- stale wording ----
    replace_text(find(document, "表 1  本次更新内容一览"), "表 1  本次更新内容一览（2.1 → 3.0）")
    replace_text(find(document, "第一步，检测行人"), 
                 "第一步，检测行人。人行模式使用 COCO 通用权重，只取行人、自行车、电动车三类；"
                 "车辆模式保留原有四类机动车权重。两类权重都来自成熟的开源模型，"
                 "本项目不重新设计网络结构，也不依赖本项目自己的标注数据，换点位不需要重新标注。")
    replace_text(find(document, "第三步，地面标定"),
                 "第三步，地面标定。在画面里选四个已知地面点，再加一个中心点校核，计算单应矩阵，"
                 "把像素坐标换成米制坐标。标定结果带重投影误差；只统计控制点围合范围内的轨迹，范围外不外推。"
                 "这一步不需要训练，是几何计算，也是“空间尺度”能否成立的关键。"
                 "本次实测尚未做地面标定，因此报告中的距离按画面宽比例表达；补一次矩形量测即可升级为米制。")
    replace_text(find(document, "第四步，指标"),
                 "第四步，指标。把轨迹汇总成使用强度（人·秒，标定后人·秒/平方米）、停留节点、"
                 "穿越与停留比、设施占用率和分时段对比。指标口径写进数据字典，"
                 "保证不同人、不同时间算出来是同一件事。")
    replace_text(find(document, "6 个月：完成一处点位的人行模块验证"),
                 "6 个月：校园点位已完成人行实测，转入老城区一处临街空间补米制标定复测，"
                 "形成可复现的“标定—轨迹—指标”流程。")
    replace_text(find(document, "与通用行人计数工具的区别在于"),
                 "与通用行人计数工具的区别在于：通用工具给人数曲线和热力图，本项目给可复核、可复测、"
                 "能落到平面图上的轨迹与指标（完成标定后即可换算为米制），并把每一条指标的边界写清楚。"
                 "创新点不在网络结构，而在“空间归算 + 可复核的指标口径 + 规划转译”。")
    replace_text(find(document, "使用流程为：选择视频或摄像头"),
                 "使用流程为：选择视频或摄像头，填写点位与采集时间，开始分析，分析结束后软件自动生成空间画像，"
                 "需要米制时再补四点标定。3.0 版把标定拾取、轨迹重建、热力图与指标合成一步，"
                 "不再依赖额外的分析脚本。")
    replace_text(find(document, "每次分析输出 session.json"),
                 "每次分析输出 session.json、frames.csv、detections.csv、events.csv、summary.json，"
                 "并在会话目录的 space/ 下生成 tracks.csv（轨迹表）、metrics.json（全部指标）、"
                 "heatmap.jpg（存在强度）、heatmap_dwell.jpg（仅停留段）、flow.jpg（主流向）、"
                 "time_curve.png（节律）、playback.mp4（轨迹回放）与 summary.md（文字摘要）。"
                 "同一份录像可以反复重算，不需要重新跑模型。")

    # ---- table 1 header ----
    for row in document.tables[0].rows:
        for cell in row.cells:
            if cell.text.strip() == "2.2 版（本次更新）":
                paragraph = cell.paragraphs[0]
                replace_text(paragraph, "3.0 版（本次更新）")

    # ---- figures in 3.2 ----
    anchor = find(document, "使用边界明确写入软件与报告")
    anchor = insert.text(anchor,
                         "下面两张图是软件的实际界面：左侧是运行状态，右侧是分析结束后自动弹出的画像面板。")
    anchor = insert.picture(anchor, FIGURES["ui"])
    anchor = insert.caption(anchor, "图 1　分析软件运行界面（人行观测模式，示例片段）")
    anchor = insert.picture(anchor, FIGURES["dialog"])
    anchor = insert.caption(anchor, "图 2　分析结束后自动生成的空间画像面板")

    # ---- figures in 3.9.1 ----
    anchor = find(document, "必须同时说明边界")
    while not anchor.text.strip().startswith("必须同时说明边界"):
        break
    # the block is long; attach to the paragraph that ends with the report pointer
    for paragraph in document.paragraphs:
        if paragraph.text.strip().endswith("见随附的《空间使用画像实测报告》。"):
            anchor = paragraph
    anchor = insert.text(anchor, "这次实测的直接产出如下两图，原图与回放视频随报告一并提交。")
    anchor = insert.picture(anchor, FIGURES["heat"])
    anchor = insert.caption(anchor, "图 3　空间使用强度（人·秒）：暖色表示单位面积累计停留时间更长")
    anchor = insert.picture(anchor, FIGURES["flow"])
    anchor = insert.caption(anchor, "图 4　各网格的主流移动方向")

    # ---- delivery section ----
    anchor = find(document, "每个点位按五步交付")
    replace_text(anchor,
                 "每个点位按五步交付，客户在任何一步都能看到可核对的中间成果，不用等到最后才知道数据长什么样。"
                 "本次实测已经把其中两件交付物做成了实物：一份十页的《空间使用画像实测报告》和一段六十秒的"
                 "轨迹回放视频——客户拿到的是可以直接翻阅、也可以直接插进汇报的材料，不是一堆 CSV。")

    document.save(str(DOC))
    print("saved", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
