"""Hybrid retriever: BM25 (exact terms, numbers, acronyms) fused with TF-IDF cosine
(word + bigram overlap). Runs fully locally; no embedding model download required."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunker import Chunk
from .text import GENERIC_TERMS, QUERY_NOISE, SYNONYMS, content_tokens, ngram_analyzer


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused rank score, relative to the best hit in this search (0-1)
    cosine: float  # absolute TF-IDF cosine
    coverage: float  # IDF-weighted fraction of query terms present in the chunk
    unseen: float = 0.0  # fraction of query terms that occur nowhere in the corpus

    @property
    def relevance(self) -> float:
        """Absolute 0-1 relevance used for the 'is this answerable from the documents' check.
        Query words the corpus has never seen ("CEO") usually mean the topic is not covered."""
        base = 0.6 * self.coverage + 0.4 * min(1.0, self.cosine / 0.35)
        return max(0.0, min(1.0, base * (1 - 0.7 * self.unseen)))


def _norm(a: np.ndarray) -> np.ndarray:
    m = a.max() if a.size else 0.0
    return a / m if m > 0 else a


class HybridRetriever:
    K1, B = 1.5, 0.75

    def __init__(self, bm25_weight: float = 0.5) -> None:
        self.w = bm25_weight
        self.chunks: list[Chunk] = []
        self._dirty = True

    # ----- index management -------------------------------------------------
    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._dirty = True

    def remove_doc(self, doc: str) -> None:
        self.chunks = [c for c in self.chunks if c.doc != doc]
        self._dirty = True

    def chunks_for(self, doc: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc == doc]

    def _ensure(self) -> None:
        if not self._dirty:
            return
        self._tok = [content_tokens(c.index_text) for c in self.chunks]
        self._tokset = [set(t) for t in self._tok]
        self._dl = np.array([len(t) for t in self._tok], dtype=float)
        self._avgdl = float(self._dl.mean()) if len(self._dl) else 1.0
        inv: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(self._tok):
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                inv.setdefault(t, []).append((i, tf))
        self._inv = inv
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in inv.items()}
        if self.chunks:
            self._vec = TfidfVectorizer(analyzer=ngram_analyzer, sublinear_tf=True)
            self._X = self._vec.fit_transform([c.index_text for c in self.chunks])
        self._dirty = False

    # ----- search -----------------------------------------------------------
    def search(self, query: str, k: int = 6, docs: list[str] | None = None) -> list[Hit]:
        self._ensure()
        n = len(self.chunks)
        q_terms = set(content_tokens(query)) - QUERY_NOISE
        if n == 0 or not q_terms:
            return []

        # thesaurus: swap an unknown everyday word for its policy wording when the corpus has it
        lowered = query.lower()
        extra: list[str] = []
        for word, syns in SYNONYMS.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                found = {t for s in syns for t in content_tokens(s) if t in self._idf}
                if found:
                    extra += [s for s in syns]
                    q_terms |= found
                    q_terms -= {t for t in content_tokens(word) if t not in self._idf}
        if extra:
            query = f"{query} {' '.join(extra)}"
        q_terms -= {t for t in q_terms if t in GENERIC_TERMS and t not in self._idf}
        if not q_terms:
            return []

        # unseen query words are treated as specific (rare) so an unanswerable question cannot
        # pass the relevance gate on a generic word such as "company" alone
        unseen_w = 0.75 * max(self._idf.values(), default=1.0)
        weights = {
            t: self._idf.get(t, unseen_w) * (0.3 if t in GENERIC_TERMS else 1.0) for t in q_terms
        }
        total_w = sum(weights.values()) or 1.0
        unseen = sum(1 for t in q_terms if t not in self._idf) / len(q_terms)
        cos = (self._X @ self._vec.transform([query]).T).toarray().ravel()
        bm = np.zeros(n)
        for term in q_terms:
            postings = self._inv.get(term)
            if not postings:
                continue
            idf = weights[term] if term in self._idf else 0.0
            for i, tf in postings:
                denom = tf + self.K1 * (1 - self.B + self.B * self._dl[i] / self._avgdl)
                bm[i] += idf * tf * (self.K1 + 1) / denom

        if docs:
            allowed = np.array([c.doc in set(docs) for c in self.chunks])
            cos, bm = cos * allowed, bm * allowed
        fused = self.w * _norm(bm) + (1 - self.w) * _norm(cos)

        hits: list[Hit] = []
        for i in np.argsort(-fused)[: k * 3]:
            if fused[i] <= 0:
                break
            # IDF-weighted coverage: matching only a generic word ("company") counts for little
            cov = sum(w for t, w in weights.items() if t in self._tokset[i]) / total_w
            hits.append(Hit(self.chunks[i], float(fused[i]), float(cos[i]), cov, unseen))
            if len(hits) == k:
                break
        return hits
