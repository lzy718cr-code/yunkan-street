"""One-off editorial pass on 附件2: reflect the delivered person-mode work.

The document was written while the person module was still "in development".
This script updates only status wording and inserts one measured-results block,
so the plan stops describing as future work something that now exists.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph


DOC = Path(r"材料_附件2技术完善版\附件2：创业计划书_20261022.docx")


def replace_text(paragraph: Paragraph, text: str) -> None:
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for run in runs[1:]:
        run._element.getparent().remove(run._element)


def insert_after(anchor: Paragraph, text: str, template: Paragraph) -> Paragraph:
    new_element = deepcopy(template._p)
    anchor._p.addnext(new_element)
    created = Paragraph(new_element, anchor._parent)
    runs = created.runs
    if runs:
        runs[0].text = text
        for run in runs[1:]:
            run._element.getparent().remove(run._element)
    else:
        created.add_run(text)
    return created


def table_status(document: Document, table_index: int, first_cell: str, changes: dict[int, str]) -> None:
    for row in document.tables[table_index].rows:
        if row.cells[0].text.strip() == first_cell:
            for column, value in changes.items():
                cell = row.cells[column]
                paragraph = cell.paragraphs[0]
                if paragraph.runs:
                    paragraph.runs[0].text = value
                    for run in paragraph.runs[1:]:
                        run._element.getparent().remove(run._element)
                else:
                    paragraph.add_run(value)


def main() -> int:
    document = Document(str(DOC))
    paragraphs = document.paragraphs

    # ---- status of the person module ----
    for paragraph in paragraphs:
        text = paragraph.text.strip()
        if text.startswith("正在开发的 2.2 版把对象从“车”扩展到“人”"):
            replace_text(paragraph,
                         "3.0 版已经把对象从“车”扩展到“人”，并完成了第一次实测：行人检测与跟踪、"
                         "断轨拼接、使用强度与停留指标、流向场、人车交织筛查，全部从逐帧记录离线重算，"
                         "不需要重新训练模型。它回答的不再是“画面里有几辆车”，而是"
                         "“这块空间哪里有人用、哪里没人去、人从哪里走、在哪里停”。")
        elif text.startswith("当前 2.1 版已交付车辆观测能力"):
            replace_text(paragraph,
                         "车辆观测能力（四类机动车检测、ByteTrack 跟踪、方向过线、区域连续驻留、"
                         "逐帧 CSV 与参数快照）在 2.1 版已交付；3.0 版在其之上增加了人行观测与空间画像层，"
                         "自动化测试 63 项通过，并有打包运行验证记录。")
        elif text.startswith("本次 2.1 版完成 48 项自动化测试"):
            replace_text(paragraph,
                         "软件测试累计 63 项，覆盖计数口径、时间戳、ROI 中断、CSV 与摘要一致性、"
                         "热更新、线程退出、来源切换、旧信号隔离、标定几何以及新增的轨迹拼接与画像指标。"
                         "打包程序在 CUDA 与 CPU 两种模式下完成软件验证，核对逐帧记录、源时间、"
                         "参数快照、事件与摘要一致性。软件验证不等于检测精度评测，两者在报告中分开表述。")

    # ---- capability table ----
    table_status(document, 0, "观测对象", {3: "已实测（校园街角 10 分钟）"})
    table_status(document, 0, "空间定位", {3: "模块完成，未做现场标定"})
    table_status(document, 0, "轨迹", {3: "已完成并实测"})
    table_status(document, 0, "停留", {3: "已完成并实测"})
    table_status(document, 0, "导出", {3: "已完成（含热力图与回放）"})
    table_status(document, 0, "测试", {2: "63 项"})

    # ---- validation plan table ----
    table_status(document, 6, "地面标定", {4: "方法已实现，待现场量测"})
    table_status(document, 6, "断轨拼接", {4: "已完成：1708→1404 条，合并 304 段"})
    table_status(document, 6, "行人检测", {4: "软件侧已跑通，精度对照待标注"})
    table_status(document, 6, "跟踪稳定", {4: "待标注评测"})
    table_status(document, 6, "人数统计", {4: "已有实测结果，缺人工真值对照"})
    table_status(document, 6, "驻留时长", {4: "已有实测结果，缺人工计时对照"})

    # ---- insert the measured block after the discipline paragraph ----
    anchor = next(p for p in document.paragraphs
                  if p.text.strip().startswith("验证纪律："))
    body_template = next(p for p in document.paragraphs
                         if p.text.strip().startswith("软件能跑不等于结果可信"))
    heading_template = next(p for p in document.paragraphs
                            if p.text.strip().startswith("3.9 验证方案"))
    heading_copy = deepcopy(heading_template._p)
    # strip numbering leftovers by keeping only the run formatting
    heading = insert_after(anchor, "3.9.1　已完成的一次实测（校园商业街角）", heading_template)

    blocks = [
        "2025 年 4 月的一段既有固定机位素材被用作第一次实测对象：四层楼下俯视校园商业街角，"
        "画面内包含便利店门前广场、两个出入口、自行车与电动车停放带和一条东西向穿行带。"
        "机位 30 分钟内首尾帧配准位移 0.1 像素、旋转 0.0°，属于真正意义上的固定机位；"
        "取其中 10 分钟（18000 帧、30 fps）做逐帧分析，全流程不重训任何模型。",

        "结果如下：检出记录 22.6 万行，原始轨迹 1708 条，断轨拼接后 1404 条（合并 304 段），"
        "其中行人 1161 条；行人累计存在 7967 人·秒，平均停留 6.9 秒、中位 4.0 秒、最长 82.3 秒，"
        "停留段占比 33.5%，占用网格 859 个；筛出运动中的非机动车与行人的近距接近 124 组，"
        "涉及行人 116 位、非机动车 96 辆；长时间几乎不动的非机动车 7 段、累计 348 秒、集中在 3 处。",

        "这组数字支撑三个判断：第一，这个街角的使用由东西向穿越主导，停留集中在便利店门前一条窄带，"
        "两张热力图相减即可看出“通过”与“停留”的分界；第二，人车交织集中在穿越带与通行带的重叠段，"
        "而不是全场均匀分布，减速与让行提示应当只放在这一段；第三，非机动车停放已经贴着人行边界生长，"
        "停放带与门前停留带需要一起重画。三条都标注了证据强度与待验证项。",

        "必须同时说明边界：本次没有做地面标定，所以全部距离与面积以“画面宽比例”表达，不出米制密度；"
        "跟踪 ID 会断裂，1161 条轨迹不等于 1161 个人；停留是本会话内连续可见时长，不等于停留意愿；"
        "交织是距离筛查值，不是冲突或安全评价；年龄与性别只在可见人脸子集上可能，样本偏差大，"
        "未纳入任何结论。补一次地面矩形量测即可把全部指标升级为米制，这也是下一步最先要做的事。"
        "完整的实测过程、图表与口径说明见随附的《空间使用画像实测报告》。",
    ]
    current = heading
    for text in blocks:
        current = insert_after(current, text, body_template)

    document.save(str(DOC))
    print(f"已更新：{DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
