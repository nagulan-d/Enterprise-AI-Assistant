"""Split document sections into overlapping, citation-ready chunks."""

from __future__ import annotations

from dataclasses import dataclass

from .loaders import Section
from .text import split_sentences


@dataclass
class Chunk:
    doc: str
    index: int
    text: str
    page: int | None = None
    heading: str | None = None

    @property
    def index_text(self) -> str:
        """Text used for retrieval: the heading breadcrumb helps match section-level questions."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text

    @property
    def locator(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.heading:
            parts.append(self.heading)
        return " · ".join(parts) if parts else "document"


def _units(text: str, max_chars: int) -> list[str]:
    """Break text into units no longer than max_chars (blocks -> lines -> sentences -> words)."""
    units: list[str] = []
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        if len(block) <= max_chars:
            units.append(block)
            continue
        for sent in split_sentences(block) or [block]:
            if len(sent) <= max_chars:
                units.append(sent)
            else:  # extremely long line: hard-wrap on words
                words, cur = sent.split(), ""
                for w in words:
                    if cur and len(cur) + len(w) + 1 > max_chars:
                        units.append(cur)
                        cur = w
                    else:
                        cur = f"{cur} {w}".strip()
                if cur:
                    units.append(cur)
    return units


def chunk_sections(
    doc: str, sections: list[Section], max_chars: int = 900, overlap: int = 160
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        buf: list[str] = []
        size = 0

        def emit() -> None:
            body = "\n".join(buf).strip()
            if body:
                chunks.append(Chunk(doc, len(chunks), body, sec.page, sec.heading))

        for unit in _units(sec.text, max_chars):
            if buf and size + len(unit) + 1 > max_chars:
                emit()
                tail = buf[-1] if len(buf[-1]) <= overlap else ""
                buf = [tail] if tail else []
                size = len(tail)
            buf.append(unit)
            size += len(unit) + 1
        emit()
    return chunks
