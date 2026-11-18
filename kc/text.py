"""Text utilities shared by the chunker, retriever and offline (extractive) modes."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

STOP = frozenset(
    """a an the and or of to in on for with at by from is are was were be been being it its
    this that these those as if then than so such not no do does did can could should would may
    might will shall must i you he she we they them their our your my me us what which who whom
    whose when where why how about into over under per via also any all each every both more
    most other some only own same too very just there here have has had having""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def stem(word: str) -> str:
    """Very small suffix stripper so that leave/leaves/leaving share one index term."""
    w = word
    if len(w) > 3 and not w.endswith(("ss", "us", "is")):
        for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)] + rep
                break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def raw_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Stemmed, stop-word-free tokens."""
    return [stem(t) for t in raw_tokens(text) if t not in STOP]


def ngram_analyzer(text: str) -> list[str]:
    t = content_tokens(text)
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


# Question filler that says nothing about the topic ("how many ... do I get"): ignored in queries.
QUERY_NOISE = frozenset(
    stem(w) for w in "get got gets getting give tell need want know please show explain describe list many much "
    "mean meaning kind sort".split()
)
# Generic nouns/verbs that match everywhere ("rate", "apply"): kept, but weighted down in retrieval.
GENERIC_TERMS = frozenset(
    stem(w) for w in "requirement requirements rule rules policy policies process procedure limit rate "
    "detail details information info amount number length apply applies".split()
)

# Small domain thesaurus so everyday wording still reaches policy wording (lexical retrieval has no
# notion of meaning). Add your organisation's own jargon here.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "vacation": ("annual leave",), "holiday": ("leave",), "holidays": ("leave",), "pto": ("annual leave",),
    "time off": ("leave",), "maternity": ("parental",), "paternity": ("parental",), "ill": ("sick",),
    "illness": ("sick",), "sickness": ("sick",), "wfh": ("remote work",), "telework": ("remote work",),
    "telecommute": ("remote work",), "reimbursement": ("claim", "expense"), "reimburse": ("claim", "expense"),
    "refund": ("reimburse",), "boss": ("manager",), "supervisor": ("manager",), "passcode": ("password",),
    "2fa": ("multi factor authentication",), "mfa": ("multi factor authentication",),
    "breach": ("incident",), "hacked": ("incident",), "quit": ("resignation",), "resign": ("resignation",),
    "hotel": ("accommodation",), "per diem": ("daily allowance",), "food": ("meal",),
    "attrition": ("turnover",), "sales": ("revenue",), "income": ("revenue",),
}


def clean_line(line: str) -> str:
    return _BULLET.sub("", line).strip()


def split_sentences(text: str) -> list[str]:
    """Split on newlines (bullets, table rows) and then on sentence punctuation."""
    out: list[str] = []
    for line in text.splitlines():
        line = clean_line(line)
        if not line:
            continue
        out.extend(s.strip() for s in _SENT_END.split(line) if s.strip())
    return out


def rank_by_centroid(texts: list[str]) -> np.ndarray:
    """Score each text by similarity to the centroid of all texts (extractive salience)."""
    if not texts:
        return np.zeros(0)
    try:
        vec = TfidfVectorizer(analyzer=content_tokens, sublinear_tf=True)
        X = vec.fit_transform(texts)
    except ValueError:  # empty vocabulary
        return np.zeros(len(texts))
    centroid = np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X @ centroid).ravel()


def surface_counts(text: str) -> dict[str, Counter]:
    """stem -> Counter(surface form) so key terms can be shown as readable words."""
    table: dict[str, Counter] = {}
    for t in raw_tokens(text):
        if t in STOP or len(t) < 3 or t.isdigit():
            continue
        table.setdefault(stem(t), Counter())[t] += 1
    return table


def numbers_in(text: str) -> set[str]:
    return {n.strip(".,") for n in re.findall(r"\d[\d,\.]*", text)}
