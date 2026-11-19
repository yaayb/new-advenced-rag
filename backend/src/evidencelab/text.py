"""Tokenisation and chunking primitives."""

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[^\W_]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were what which who whom with how why when where does do did into than then there "
    "le la les un une des du de et en est sont ce cet cette ces au aux par pour sur dans qui que quoi quel quelle "
    "quels quelles comment pourquoi quand ou où avec sans sous il elle ils elles on nous vous je tu mon ma mes "
    "son sa ses leur leurs ne pas plus se s l d c j n m t qu a ont été être avoir fait".split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS] if drop_stopwords else toks


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True)
class RawChunk:
    index: int
    text: str
    token_count: int


def chunk_text(text: str, size: int, overlap: int) -> list[RawChunk]:
    """Sliding word-window chunker that prefers to break on sentence boundaries."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks: list[RawChunk] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        body = " ".join(current)
        chunks.append(RawChunk(len(chunks), body, count_tokens(body)))
        # carry overlap words into next chunk
        words = body.split()
        tail = words[-overlap:] if overlap else []
        current = [" ".join(tail)] if tail else []
        current_len = len(tail)

    for sentence in sentences:
        words = sentence.split()
        # split oversize sentences
        while len(words) > size:
            if current_len:
                flush()
            part, words = words[:size], words[size - overlap :]
            chunks.append(RawChunk(len(chunks), " ".join(part), len(part)))
            current, current_len = [], 0
        if not words:
            continue
        if current_len + len(words) > size and current_len > overlap:
            flush()
        current.append(" ".join(words))
        current_len += len(words)
    if current_len > overlap or (current and not chunks):
        flush()
    return chunks
