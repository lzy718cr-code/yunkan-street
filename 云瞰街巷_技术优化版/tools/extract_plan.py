"""Extract 附件2 into a structured dump so it can be rebuilt cleanly.

Rebuilding from a dump (instead of hand-retyping) guarantees that no fact,
number or table survived only in the old file: every block is carried over and
only the explicitly overridden paragraphs change.
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


SRC = Path(r"材料_附件2技术完善版\附件2：创业计划书_20261022.docx")
OUT = Path(r"_plan_dump.json")
DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"


def has_drawing(paragraph) -> bool:
    return bool(paragraph._p.findall(f".//{DRAWING}"))


def main() -> int:
    document = Document(str(SRC))
    blocks: list[dict] = []
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            from docx.text.paragraph import Paragraph
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if has_drawing(paragraph):
                blocks.append({"type": "image"})
                continue
            if not text:
                blocks.append({"type": "blank"})
                continue
            bold = any(run.bold for run in paragraph.runs)
            blocks.append({"type": "text", "text": text, "bold": bold})
        elif child.tag == qn("w:tbl"):
            from docx.table import Table
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = []
                seen = set()
                for cell in row.cells:
                    key = cell._tc
                    if key in seen:
                        continue
                    seen.add(key)
                    cells.append(cell.text.strip())
                rows.append(cells)
            blocks.append({"type": "table", "rows": rows})
    OUT.write_text(json.dumps(blocks, ensure_ascii=False, indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block["type"]] = counts.get(block["type"], 0) + 1
    print("blocks", len(blocks), counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
