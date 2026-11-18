"""Load business documents (PDF, DOCX, Markdown, TXT, CSV) into locatable sections."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = (".pdf", ".docx", ".md", ".markdown", ".txt", ".csv")


@dataclass
class Section:
    text: str
    page: int | None = None
    heading: str | None = None  # breadcrumb such as "Leave Types > Sick Leave"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> str:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return " > ".join(t for _, t in stack)


def _load_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    crumb: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(body, None, crumb))
        buf.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            flush()
            crumb = _push_heading(stack, len(m.group(1)), m.group(2))
        else:
            buf.append(line)
    flush()
    return sections


def _load_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            sections.append(Section(txt, page=i))
    if not sections:
        raise ValueError("No extractable text found (is this a scanned PDF? OCR is required).")
    return sections


def _load_docx(data: bytes) -> list[Section]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    crumb: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(body, None, crumb))
        buf.clear()

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "") if para.style is not None else ""
            m = re.match(r"Heading (\d)", style)
            if m or style == "Title":
                flush()
                crumb = _push_heading(stack, int(m.group(1)) if m else 1, text)
            elif style.lower().startswith("list"):
                buf.append(f"- {text}")
            else:
                buf.append(text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, doc)
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                if any(cells):
                    buf.append(" | ".join(cells))
    flush()
    return sections


def _load_csv(text: str) -> list[Section]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    lines = [" | ".join(f"{h}: {v}" for h, v in zip(header, r) if v.strip()) for r in body]
    return [Section("\n".join(l for l in lines if l), None, "Table")]


def load_bytes(name: str, data: bytes) -> list[Section]:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        sections = _load_pdf(data)
    elif ext == ".docx":
        sections = _load_docx(data)
    elif ext in (".md", ".markdown"):
        sections = _load_markdown(_decode(data))
    elif ext == ".txt":
        sections = [Section(_decode(data).strip())]
    elif ext == ".csv":
        sections = _load_csv(_decode(data))
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED)}")
    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise ValueError(f"'{name}' contains no readable text.")
    return sections


def load_path(path: str | Path) -> tuple[str, list[Section]]:
    p = Path(path)
    return p.name, load_bytes(p.name, p.read_bytes())
