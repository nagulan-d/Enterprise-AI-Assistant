"""KnowledgeCopilot: RAG question answering with citations and follow-up memory,
document summarization, multi-document comparison, and decision / action-item generation.

Every feature has two paths:
  * LLM path        - Claude, grounded strictly on retrieved passages.
  * Extractive path - deterministic, offline fallback (no API key, or the call failed).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunker import Chunk, chunk_sections
from .llm import LLM, LLMError
from .loaders import SUPPORTED, Section, load_bytes
from .retriever import Hit, HybridRetriever
from .text import (
    content_tokens,
    numbers_in,
    rank_by_centroid,
    raw_tokens,
    split_sentences,
    surface_counts,
    STOP,
)

# --------------------------------------------------------------------------- data


@dataclass
class Source:
    id: int
    doc: str
    locator: str
    text: str
    score: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.doc} · {self.locator}"


@dataclass
class Answer:
    text: str
    sources: list[Source]
    grounded: bool
    confidence: str  # High | Medium | Low | None
    mode: str  # llm | extractive | none
    standalone_question: str
    note: str = ""


@dataclass
class Summary:
    text: str
    doc: str
    mode: str
    key_terms: list[str]
    words: int
    note: str = ""


@dataclass
class Comparison:
    text: str
    docs: list[str]
    sources: list[Source]
    mode: str
    table: list[dict] = field(default_factory=list)
    note: str = ""


@dataclass
class ActionPlan:
    summary: str
    recommendation: str
    items: list[dict]
    risks: list[str]
    sources: list[Source]
    mode: str
    note: str = ""


@dataclass
class DocRecord:
    name: str
    sections: list[Section]
    chunk_count: int
    words: int
    pages: int | None
    added: str

    def full_text(self) -> str:
        parts = []
        for s in self.sections:
            label = s.heading or (f"Page {s.page}" if s.page else None)
            parts.append(f"## {label}\n{s.text}" if label else s.text)
        return "\n\n".join(parts)


class _SourceBook:
    """Assigns stable [S#] ids to chunks as they are cited."""

    def __init__(self) -> None:
        self.items: list[Source] = []
        self._ids: dict[tuple[str, int], int] = {}

    def cite(self, obj: Hit | Chunk, score: float | None = None) -> int:
        chunk = obj.chunk if isinstance(obj, Hit) else obj
        if score is None and isinstance(obj, Hit):
            score = obj.relevance
        key = (chunk.doc, chunk.index)
        if key not in self._ids:
            self.items.append(
                Source(len(self.items) + 1, chunk.doc, chunk.locator, chunk.text, round(score or 0.0, 3))
            )
            self._ids[key] = len(self.items)
        return self._ids[key]

    def labels(self, ids: list[int]) -> str:
        by_id = {s.id: s for s in self.items}
        return "; ".join(by_id[i].label for i in ids if i in by_id)


# ------------------------------------------------------------------------ prompts

QA_SYSTEM = """You are the Enterprise Knowledge Copilot for an organisation's internal documents.
Rules:
1. Answer ONLY from the numbered context passages provided. Do not use outside knowledge for facts about the organisation.
2. Cite every factual statement with its passage number in square brackets, e.g. [S1] or [S1][S3].
3. If the passages do not contain the answer, say so plainly and state what is missing. Do not guess.
4. If passages from different documents disagree (for example an older and a newer policy), point out the conflict and say which document states what.
5. Be concise and professional. Use short bullet points for lists, steps or conditions. Quote numbers, limits and deadlines exactly.
6. Use the earlier conversation to resolve follow-up questions."""

CONDENSE_SYSTEM = (
    "Rewrite the user's latest message as one standalone question that can be understood without the "
    "conversation. Keep names, numbers and document titles. Reply with the rewritten question only."
)

SUMMARY_STYLES = {
    "executive": "Write an executive summary: a 2-sentence overview, then 5-7 key takeaways as bullets, then a 'Risks / open questions' bullet list if the document raises any.",
    "detailed": "Write a section-by-section summary using ## headings that follow the document's structure. Keep every number, limit, date and owner.",
    "bullets": "Write a concise bullet-point summary of 8-12 bullets. Keep every number, limit, date and owner.",
}
SUMMARY_SYSTEM = (
    "You summarise business documents accurately for busy employees. Use only the document text. "
    "Never invent facts. Preserve numbers, dates, thresholds and responsibilities exactly."
)

COMPARE_SYSTEM = """You compare organisational documents for decision makers, using ONLY the numbered passages provided.
Cite passage numbers like [S2] after every factual claim. Produce markdown with these sections:
## Overview (2-3 sentences)
## Comparison table (markdown table: one row per topic, one column per document; write 'Not stated' where a document is silent)
## Key similarities
## Key differences and conflicts
## Recommendation (what the reader should do or clarify next)
Never guess beyond the passages."""

PLAN_SYSTEM = """You are a decision-support assistant for an organisation. Using ONLY the numbered passages provided,
assess the goal or scenario and return ONLY valid JSON (no prose, no code fences) with this shape:
{"summary": "2-3 sentence grounded assessment with [S#] citations",
 "recommendation": "the recommended decision or course of action, with reasoning and conditions, citing [S#]",
 "action_items": [{"action": "specific step", "owner": "role/team or Unassigned", "priority": "High|Medium|Low",
                   "deadline": "as stated in the documents, or Not specified", "source": "S1"}],
 "risks": ["gaps, conflicts, approvals or risks the reader must be aware of"]}
If the passages do not contain enough information, say so in "summary" and keep action_items to what is supported."""

NOT_FOUND = (
    "I couldn't find anything relevant to that in the uploaded documents. "
    "Try rephrasing the question, or upload the document that covers this topic."
)

_FOLLOWUP_START = re.compile(r"^\s*(and|also|what about|how about|why|so|then|but|what if)\b", re.I)
_PRONOUN = re.compile(r"\b(it|its|that|this|they|them|their|those|these|he|she|there)\b", re.I)
_QUANT = re.compile(
    r"\b(how many|how much|how long|how often|limit|cap|maximum|minimum|max|min|deadline|when|"
    r"days?|hours?|weeks?|months?|amount|cost|threshold|rate)\b",
    re.I,
)
_OBLIG = re.compile(
    r"\b(must|shall|should|required?|requires?|need(?:s)? to|ensure|mandatory|responsible for|submit|"
    r"complete[sd]?|report(?:ed)?|approv(?:al|ed|e)\s+(?:by|from|before|of)|obtain|notify|review(?:ed)?|"
    r"no later than|prior to|"
    r"within \d+|by \d|deadline|not (?:be )?(?:allowed|permitted)|prohibited)\b",
    re.I,
)
_HIGH = re.compile(
    r"\b(must|shall|mandatory|immediately|critical|prohibited|within (?:1|one|2|two|24) "
    r"(?:hour|day)s?|not (?:be )?(?:allowed|permitted))\b",
    re.I,
)
_DEADLINE = re.compile(
    r"(within\s+\d+\s*(?:working |business |calendar )?(?:hours?|days?|weeks?|months?)"
    r"|no later than [^,.;]+|by\s+\d{1,2}\s+\w+(?:\s+\d{4})?|by\s+(?:end of\s+)?\w+\s+\d{4}"
    r"|\d+\s*(?:working )?days? (?:prior|before|in advance)[^,.;]*|every\s+(?:\d+\s+)?\w+"
    r"|quarterly|annually|monthly|weekly)",
    re.I,
)
_OWNERS = [
    "line manager", "department head", "project manager", "security team", "steering committee",
    "team lead", "employee", "employees", "manager", "managers", "HR", "IT", "finance", "CFO", "CTO",
    "legal", "vendor", "admin", "PMO", "sponsor",
]


# ---------------------------------------------------------------------- helpers


def _cited_ids(text: str) -> list[int]:
    ids: list[int] = []
    for grp in re.findall(r"\[([^\]]*)\]", text):
        ids += [int(x) for x in re.findall(r"S(\d+)", grp)]
    return sorted(set(ids))


def _clean_history(history: list[dict], n: int = 6) -> list[dict]:
    """Keep the last n turns, strictly alternating, starting with 'user' and ending with 'assistant'."""
    out: list[dict] = []
    for m in history[-n:]:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + content
        else:
            out.append({"role": role, "content": content})
    while out and out[0]["role"] != "user":
        out.pop(0)
    if out and out[-1]["role"] == "user":
        out.pop()
    return out


def _context_block(sources: list[Source], limit: int = 1500) -> str:
    return "\n\n".join(f"[S{s.id}] {s.doc} — {s.locator}\n{s.text[:limit]}" for s in sources)


def _segments(text: str, size: int) -> list[str]:
    """Cut text into pieces of at most `size` characters, preferring paragraph, then line, then word breaks."""
    pieces: list[str] = []
    for para in text.split("\n\n"):
        while len(para) > size:
            cut = max(para.rfind("\n", 0, size), para.rfind(" ", 0, size))
            cut = cut if cut > size // 2 else size
            pieces.append(para[:cut])
            para = para[cut:].lstrip()
        pieces.append(para)
    segments, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > size:
            segments.append(cur)
            cur = ""
        cur += p + "\n\n"
    if cur.strip():
        segments.append(cur)
    return segments


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ").strip()


# ------------------------------------------------------------------- the copilot


class KnowledgeCopilot:
    def __init__(
        self,
        llm: LLM | None = None,
        top_k: int = 6,
        chunk_chars: int = 900,
        chunk_overlap: int = 160,
        min_relevance: float = 0.35,
    ) -> None:
        self.llm = llm or LLM()
        self.top_k = top_k
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.min_relevance = min_relevance
        self.retriever = HybridRetriever()
        self.docs: dict[str, DocRecord] = {}

    # ------------------------------------------------------------ documents
    def add_document(self, name: str, data: bytes) -> DocRecord:
        sections = load_bytes(name, data)
        if name in self.docs:
            self.remove_document(name)
        chunks = chunk_sections(name, sections, self.chunk_chars, self.chunk_overlap)
        self.retriever.add(chunks)
        rec = DocRecord(
            name=name,
            sections=sections,
            chunk_count=len(chunks),
            words=sum(len(s.text.split()) for s in sections),
            pages=max((s.page for s in sections if s.page), default=None),
            added=time.strftime("%Y-%m-%d %H:%M"),
        )
        self.docs[name] = rec
        return rec

    def add_path(self, path: str | Path) -> DocRecord:
        p = Path(path)
        return self.add_document(p.name, p.read_bytes())

    def load_folder(self, folder: str | Path) -> tuple[list[str], dict[str, str]]:
        loaded, errors = [], {}
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in SUPPORTED and p.is_file():
                try:
                    loaded.append(self.add_path(p).name)
                except Exception as exc:  # keep going; report per file
                    errors[p.name] = str(exc)
        return loaded, errors

    def remove_document(self, name: str) -> None:
        self.docs.pop(name, None)
        self.retriever.remove_doc(name)

    def doc_names(self) -> list[str]:
        return list(self.docs)

    def key_terms(self, doc: str, n: int = 8) -> list[str]:
        names = list(self.docs)
        if doc not in self.docs:
            return []
        texts = [self.docs[d].full_text() for d in names]
        try:
            vec = TfidfVectorizer(analyzer=content_tokens, sublinear_tf=True)
            X = vec.fit_transform(texts)
        except ValueError:
            return []
        idx = names.index(doc)
        row = X[idx].toarray().ravel()
        vocab = vec.get_feature_names_out()
        surf = surface_counts(texts[idx])
        out: list[str] = []
        for j in row.argsort()[::-1]:
            if row[j] <= 0:
                break
            term = vocab[j]
            if term.isdigit() or len(term) < 3:
                continue
            forms = surf.get(term)
            out.append(forms.most_common(1)[0][0] if forms else term)
            if len(out) == n:
                break
        return out

    def stats(self) -> list[dict]:
        return [
            {
                "Document": r.name,
                "Type": Path(r.name).suffix.lstrip(".").upper(),
                "Pages": r.pages if r.pages else "–",
                "Words": r.words,
                "Chunks": r.chunk_count,
                "Key terms": ", ".join(self.key_terms(r.name, 6)),
                "Added": r.added,
            }
            for r in self.docs.values()
        ]

    # ------------------------------------------------------------------ Q&A
    def _top_relevance(self, query: str, docs: list[str] | None, need_any: set[str] | None = None) -> float:
        """Best relevance for a query; optionally the top chunks must contain one of `need_any`."""
        for h in self.retriever.search(query, 3, docs):
            if need_any is None or need_any & set(content_tokens(h.chunk.index_text)):
                return h.relevance
        return 0.0

    def _offline_standalone(self, question: str, prior: str, docs: list[str] | None) -> str:
        """Resolve follow-ups without an LLM. The question is used as-is when it stands on its own;
        the previous question is only added when that demonstrably improves retrieval."""
        words = question.split()
        pronoun = len(words) <= 8 and bool(_PRONOUN.search(question))
        marker = bool(_FOLLOWUP_START.match(question))
        own = self._top_relevance(question, docs)
        if own >= 0.5 and not pronoun:
            return question
        if not (marker or pronoun or len(content_tokens(question)) <= 3):
            return question
        combined = f"{prior} {question}"
        # unless the user used a pronoun ("what about it?"), the new wording must still match something
        new_terms = None if pronoun else set(content_tokens(question))
        return combined if self._top_relevance(combined, docs, new_terms) > own else question

    def _condense(self, question: str, history: list[dict], docs: list[str] | None = None) -> str:
        prior = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
        if not prior:
            return question
        if self.llm.available:
            transcript = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
                for m in _clean_history(history + [{"role": "assistant", "content": "."}])
            )
            try:
                out = self.llm.complete(
                    CONDENSE_SYSTEM,
                    [{"role": "user", "content": f"{transcript}\n\nLatest message: {question}"}],
                    max_tokens=200,
                    temperature=0,
                )
                return out.strip().strip('"') or question
            except LLMError:
                pass
        return self._offline_standalone(question, prior[-1], docs)

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        docs: list[str] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        question = question.strip()
        history = history or []
        if not self.docs:
            return Answer(
                "No documents are loaded yet. Upload documents (or load the sample set) from the sidebar.",
                [], False, "None", "none", question,
            )
        standalone = self._condense(question, history, docs)
        hits = [
            h
            for h in self.retriever.search(standalone, top_k or self.top_k, docs)
            if h.relevance >= self.min_relevance
        ]
        mode = "llm" if self.llm.available else "extractive"
        if not hits:
            return Answer(NOT_FOUND, [], False, "None", mode, standalone)

        book = _SourceBook()
        for h in hits:
            book.cite(h)

        note = "" if self.llm.available else self.llm.reason
        text = ""
        if self.llm.available:
            user = f"Context passages:\n\n{_context_block(book.items)}\n\nQuestion: {standalone}"
            if standalone != question:
                user += f"\n(The user's original wording was: {question})"
            try:
                text = self.llm.complete(
                    QA_SYSTEM, _clean_history(history) + [{"role": "user", "content": user}], max_tokens=1100
                )
            except LLMError as exc:
                mode, note = "extractive", f"Model call failed ({exc}); showing an extractive answer instead."
        if not text:
            mode = "extractive"
            text = self._extractive_answer(standalone, hits)

        cited = _cited_ids(text)
        shown = [s for s in book.items if s.id in cited] or book.items[:3]
        top = hits[0].relevance
        confidence = "High" if top >= 0.6 else "Medium" if top >= 0.4 else "Low"
        return Answer(text, shown, True, confidence, mode, standalone, note)

    def _extractive_answer(self, query: str, hits: list[Hit]) -> str:
        q = set(content_tokens(query))
        quant = bool(_QUANT.search(query))
        cands = []
        for rank, h in enumerate(hits):
            head = set(content_tokens(h.chunk.heading or ""))
            for s in split_sentences(h.chunk.text):
                toks = set(content_tokens(s))
                if len(toks) < 2:
                    continue
                # a sentence inherits some credit from its section heading ("Risks", "Sick Leave")
                own = len(q & toks) / len(q) if q else 0.0
                with_head = len(q & (toks | head)) / len(q) if q else 0.0
                cov = own + 0.6 * (with_head - own)
                score = cov + (0.15 if quant and re.search(r"\d", s) else 0.0) - 0.03 * rank
                cands.append((score, rank, s))
        cands.sort(key=lambda x: -x[0])
        floor = max(0.3, 0.5 * cands[0][0]) if cands else 0.3  # drop sentences far weaker than the best
        picked, seen = [], set()
        for score, rank, s in cands:
            if score < floor or s.lower() in seen:
                continue
            seen.add(s.lower())
            picked.append((rank, s))
            if len(picked) == 4:
                break
        if not picked:
            picked = [(0, s) for s in split_sentences(hits[0].chunk.text)[:2]]
        picked.sort(key=lambda x: x[0])
        lines = [f"- {s} [S{rank + 1}]" for rank, s in picked]
        text = "From the documents:\n" + "\n".join(lines)

        # flag disagreements between documents (e.g. an old and a revised policy)
        by_doc: dict[str, set[str]] = {}
        for rank, s in picked:
            by_doc.setdefault(hits[rank].chunk.doc, set()).update(numbers_in(s))
        if len(by_doc) > 1 and len({frozenset(v) for v in by_doc.values()}) > 1:
            text += (
                "\n\n⚠ **Documents differ:** "
                + ", ".join(by_doc)
                + " give different figures for this. Check which version is current before acting."
            )
        return text

    # ------------------------------------------------------------ summarise
    def summarize(self, doc: str, style: str = "executive") -> Summary:
        if doc not in self.docs:
            raise KeyError(f"Unknown document: {doc}")
        rec = self.docs[doc]
        terms = self.key_terms(doc)
        style = style if style in SUMMARY_STYLES else "executive"
        note = "" if self.llm.available else self.llm.reason
        if self.llm.available:
            try:
                return Summary(self._llm_summary(rec, style), doc, "llm", terms, rec.words)
            except LLMError as exc:
                note = f"Model call failed ({exc}); showing an extractive summary instead."
        return Summary(self._extractive_summary(rec, style), doc, "extractive", terms, rec.words, note)

    def _llm_summary(self, rec: DocRecord, style: str) -> str:
        text = rec.full_text()
        instruction = SUMMARY_STYLES[style]
        if len(text) <= 30_000:
            return self.llm.complete(
                SUMMARY_SYSTEM,
                [{"role": "user", "content": f"Document: {rec.name}\n\n{text}\n\n{instruction}"}],
                max_tokens=1500,
            )
        # map-reduce for long documents
        segments = _segments(text, max(12_000, math.ceil(len(text) / 12)))
        partials = [
            self.llm.complete(
                SUMMARY_SYSTEM,
                [{"role": "user", "content": (
                    f"Part {i} of {len(segments)} of '{rec.name}'. Summarise in at most 200 words, "
                    f"keeping all numbers, dates and owners.\n\n{s}")}],
                max_tokens=500,
            )
            for i, s in enumerate(segments, 1)
        ]
        joined = "\n\n".join(f"Part {i}: {p}" for i, p in enumerate(partials, 1))
        return self.llm.complete(
            SUMMARY_SYSTEM,
            [{"role": "user", "content": f"Partial summaries of '{rec.name}':\n\n{joined}\n\n{instruction}"}],
            max_tokens=1500,
        )

    def _extractive_summary(self, rec: DocRecord, style: str) -> str:
        items: list[tuple[int, str]] = []
        seen: set[str] = set()
        for si, sec in enumerate(rec.sections):
            for s in split_sentences(sec.text):
                if len(content_tokens(s)) >= 4 and s.lower() not in seen:
                    seen.add(s.lower())
                    items.append((si, s))
        if not items:
            return "This document has too little text to summarise."
        scores = rank_by_centroid([s for _, s in items])

        if style == "detailed":
            out = []
            for si, sec in enumerate(rec.sections):
                idx = [i for i, (sj, _) in enumerate(items) if sj == si]
                if not idx:
                    continue
                best = sorted(sorted(idx, key=lambda i: -scores[i])[:2])
                label = sec.heading or (f"Page {sec.page}" if sec.page else "Document")
                out.append(f"### {label}\n" + "\n".join(f"- {items[i][1]}" for i in best))
            return "\n\n".join(out)

        n = 7 if style == "executive" else 10
        best = sorted(np.argsort(-scores)[:n])
        title = "Key takeaways" if style == "executive" else "Summary"
        return f"**{title}** (key sentences extracted from the document)\n\n" + "\n".join(
            f"- {items[i][1]}" for i in best
        )

    # -------------------------------------------------------------- compare
    def _representative_chunks(self, doc: str, n: int = 4) -> list[Chunk]:
        chunks = self.retriever.chunks_for(doc)
        if len(chunks) <= n:
            return chunks
        scores = rank_by_centroid([c.text for c in chunks])
        return [chunks[i] for i in sorted(np.argsort(-scores)[:n])]

    def compare(self, docs: list[str], aspect: str | None = None) -> Comparison:
        docs = [d for d in docs if d in self.docs]
        if len(docs) < 2:
            raise ValueError("Select at least two documents to compare.")
        aspect = (aspect or "").strip() or None

        evidence: dict[str, list[Chunk]] = {}
        for d in docs:
            if aspect:
                found = self.retriever.search(aspect, 4, [d])
                floor = max(0.35, 0.5 * found[0].relevance) if found else 1.0  # drop loosely related sections
                evidence[d] = [h.chunk for h in found if h.relevance >= floor]
            else:
                evidence[d] = self._representative_chunks(d, 4)

        note = "" if self.llm.available else self.llm.reason
        if self.llm.available:
            book = _SourceBook()
            blocks = []
            for d in docs:
                if not evidence[d]:
                    blocks.append(f"DOCUMENT: {d}\n(no passages relevant to the topic were found)")
                    continue
                body = "\n\n".join(
                    f"[S{book.cite(c, 0.0)}] {c.locator}\n{c.text[:1500]}" for c in evidence[d]
                )
                blocks.append(f"DOCUMENT: {d}\n{body}")
            focus = f"Focus on: {aspect}." if aspect else "Cover the most important topics in the documents."
            try:
                text = self.llm.complete(
                    COMPARE_SYSTEM,
                    [{"role": "user", "content": f"{focus}\n\n" + "\n\n".join(blocks)}],
                    max_tokens=1800,
                )
                cited = _cited_ids(text)
                used = [s for s in book.items if s.id in cited] or book.items
                return Comparison(text, docs, used, "llm", [], note)
            except LLMError as exc:
                note = f"Model call failed ({exc}); showing an extractive comparison instead."

        text, table, sources = self._extractive_compare(docs, evidence, aspect)
        return Comparison(text, docs, sources, "extractive", table, note)

    def _doc_sentences(self, doc: str, chunks: list[Chunk]) -> list[tuple[str, Chunk]]:
        out, seen = [], set()
        for c in chunks:
            for s in split_sentences(c.text):
                if s.lower() not in seen and len(content_tokens(s)) >= 3:
                    seen.add(s.lower())
                    out.append((s, c))
        return out

    def _extractive_compare(
        self, docs: list[str], evidence: dict[str, list[Chunk]], aspect: str | None
    ) -> tuple[str, list[dict], list[Source]]:
        book = _SourceBook()
        sents: dict[str, list[tuple[str, Chunk]]] = {}
        for d in docs:
            pool = evidence[d] if aspect else self.retriever.chunks_for(d)
            sents[d] = self._doc_sentences(d, pool)

        base, table, parts = docs[0], [], []
        title = f"Comparison of {len(docs)} documents" + (f" on “{aspect}”" if aspect else "")
        parts.append(f"**{title}** (extractive mode: matched sentences, no AI synthesis)")
        for other in docs[1:]:
            rows = self._pair_rows(sents[base], sents[other], book)
            for r in rows:
                r.update({"doc_a": base, "doc_b": other})
            table += rows
            parts.append(f"### {base} vs {other}")
            if not rows:
                parts.append("_No closely matching statements were found between these two documents._")
                continue
            lines = [f"| Topic | {_cell(base)} | {_cell(other)} | Status |", "|---|---|---|---|"]
            for r in rows:
                lines.append(
                    f"| {_cell(r['topic'])} | {_cell(r['a'])} [S{r['a_id']}] | "
                    f"{_cell(r['b'])} [S{r['b_id']}] | {r['status']} |"
                )
            parts.append("\n".join(lines))

        sets = {d: set(self.key_terms(d, 12)) for d in docs}
        shared = set.intersection(*sets.values()) if sets else set()
        parts.append("### Themes")
        parts.append(f"- **Shared:** {', '.join(sorted(shared)) or '–'}")
        for d in docs:
            others = set().union(*(sets[o] for o in docs if o != d))
            parts.append(f"- **Distinctive to {d}:** {', '.join(sorted(sets[d] - others)) or '–'}")

        diffs = sum(1 for r in table if r["status"] == "Differs")
        parts.append(f"\n**{diffs} statement(s) differ** (different figures or limits). Review those rows first.")
        return "\n\n".join(parts), table, book.items

    def _pair_rows(
        self,
        a_sents: list[tuple[str, Chunk]],
        b_sents: list[tuple[str, Chunk]],
        book: _SourceBook,
        min_jaccard: float = 0.35,
        limit: int = 12,
    ) -> list[dict]:
        b_tokens = [set(content_tokens(s)) for s, _ in b_sents]
        used: set[int] = set()
        found = []
        for sa, ca in a_sents:
            ta = set(content_tokens(sa))
            best = None
            for j, tb in enumerate(b_tokens):
                if j in used or not tb:
                    continue
                jac = len(ta & tb) / len(ta | tb)
                if best is None or jac > best[0]:
                    best = (jac, j)
            if best and best[0] >= min_jaccard:
                used.add(best[1])
                sb, cb = b_sents[best[1]]
                if numbers_in(sa) != numbers_in(sb):
                    status = "Differs"
                elif sa.lower() == sb.lower():
                    status = "Same"
                else:
                    status = "Similar"
                ha = (ca.heading or "").split(" > ")[-1]
                hb = (cb.heading or "").split(" > ")[-1]
                if ha and hb:
                    topic = ha if ha == hb else f"{ha} / {hb}"
                else:
                    shared = [t for t in raw_tokens(sa) if t not in STOP and content_tokens(t)[:1] and
                              content_tokens(t)[0] in (ta & b_tokens[best[1]])]
                    topic = " ".join(list(dict.fromkeys(shared))[:5]) or "–"
                found.append((status, best[0], topic, sa, ca, sb, cb))
        order = {"Differs": 0, "Similar": 1, "Same": 2}
        found.sort(key=lambda r: (order[r[0]], -r[1]))
        return [
            {"topic": t, "a": sa, "b": sb, "status": st, "a_id": book.cite(ca), "b_id": book.cite(cb)}
            for st, _, t, sa, ca, sb, cb in found[:limit]
        ]

    # --------------------------------------------- decisions & action items
    def _obligation_chunks(self, docs: list[str], per_doc: int = 4) -> list[Chunk]:
        picked: list[Chunk] = []
        for d in docs:
            chunks = self.retriever.chunks_for(d)
            scored = sorted(chunks, key=lambda c: -len(_OBLIG.findall(c.text)))
            picked += [c for c in scored[:per_doc] if _OBLIG.search(c.text)]
        return picked

    def action_plan(self, goal: str = "", docs: list[str] | None = None) -> ActionPlan:
        goal = goal.strip()
        scope = [d for d in (docs or self.doc_names()) if d in self.docs]
        if not scope:
            return ActionPlan("No documents are loaded yet.", "", [], [], [], "none")

        if goal:
            chunks = [h.chunk for h in self.retriever.search(goal, 8, scope) if h.relevance >= 0.25]
        else:
            chunks = self._obligation_chunks(scope)
        if not chunks:
            return ActionPlan(
                "I couldn't find passages in the selected documents that relate to this goal.",
                "", [], [], [], "extractive",
            )

        book = _SourceBook()
        for c in chunks:
            book.cite(c)
        note = "" if self.llm.available else self.llm.reason

        if self.llm.available:
            target = goal or "Extract the key obligations, deadlines and next steps from these documents."
            try:
                raw = self.llm.complete(
                    PLAN_SYSTEM,
                    [{"role": "user", "content": f"Goal / scenario: {target}\n\nContext passages:\n\n{_context_block(book.items)}"}],
                    max_tokens=1800,
                )
                data = _parse_json(raw)
                if data is None:
                    return ActionPlan(raw, "", [], [], book.items, "llm", "The model did not return structured JSON.")
                items = []
                for it in data.get("action_items") or []:
                    if not isinstance(it, dict):
                        continue
                    pr = str(it.get("priority", "Medium")).strip().title()
                    items.append({
                        "action": str(it.get("action", "")).strip(),
                        "owner": str(it.get("owner") or "Unassigned"),
                        "priority": pr if pr in ("High", "Medium", "Low") else "Medium",
                        "deadline": str(it.get("deadline") or "Not specified"),
                        "source": book.labels(_cited_ids(f"[{it.get('source', '')}]")),
                    })
                risks = [str(r) for r in (data.get("risks") or [])]
                cited = set(_cited_ids(raw)) | {
                    int(x) for it in data.get("action_items") or [] if isinstance(it, dict)
                    for x in re.findall(r"S(\d+)", str(it.get("source", "")))
                }
                used = [s for s in book.items if s.id in cited] or book.items
                return ActionPlan(
                    str(data.get("summary", "")), str(data.get("recommendation", "")),
                    [i for i in items if i["action"]], risks, used, "llm", note,
                )
            except LLMError as exc:
                note = f"Model call failed ({exc}); showing extractive action items instead."

        return self._extractive_plan(goal, chunks, book, note)

    def _extractive_plan(self, goal: str, chunks: list[Chunk], book: _SourceBook, note: str) -> ActionPlan:
        items, seen, used = [], set(), set()
        for c in chunks:
            for s in split_sentences(c.text):
                if not _OBLIG.search(s) or s.lower() in seen or len(s) < 20:
                    continue
                seen.add(s.lower())
                # the owner is normally the grammatical subject: the role mentioned earliest
                found = [
                    (m.start(), -len(o), o)
                    for o in _OWNERS
                    if (m := re.search(rf"\b{re.escape(o)}\b", s, re.I))
                ]
                owner = min(found)[2] if found else "Unassigned"
                dl = _DEADLINE.search(s)
                items.append({
                    "action": s,
                    "owner": owner.title() if owner.islower() else owner,
                    "priority": "High" if _HIGH.search(s) else "Medium",
                    "deadline": dl.group(1) if dl else "Not specified",
                    "source": book.labels([book.cite(c)]),
                })
                used.add(book.cite(c))
        rank = {"High": 0, "Medium": 1}
        items.sort(key=lambda i: rank[i["priority"]])
        items = items[:10]
        high = sum(1 for i in items if i["priority"] == "High")
        dated = [i for i in items if i["deadline"] != "Not specified"]
        summary = (
            f"Found {len(items)} obligation(s) in the relevant passages"
            + (f" for: {goal}" if goal else "")
            + f" ({high} high priority)."
        )
        reco = (
            "Start with the High-priority items"
            + (f" and the time-bound ones ({len(dated)} have deadlines)" if dated else "")
            + ". Confirm owners for items marked Unassigned. Connect an API key for a reasoned "
            "recommendation tailored to your scenario."
        )
        cited = [src for src in book.items if src.id in used]  # only passages the items came from
        return ActionPlan(summary, reco, items, [], cited or book.items, "extractive", note)
