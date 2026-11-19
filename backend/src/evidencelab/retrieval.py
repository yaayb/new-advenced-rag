"""Retrievers: BM25 (from scratch), dense vectors (memory / Qdrant), and RRF fusion."""

import hashlib
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .text import tokenize


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    score: float
    rank: int


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder over unigrams + bigrams (L2-normalised).

    Runs offline with no model download. It is a lexical-semantic baseline, not a neural model;
    implement the `Embedder` protocol to plug in a neural encoder.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _slot(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            for f in feats:
                idx, sign = self._slot(f)
                out[i, idx] += sign
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._df: Counter[str] = Counter()

    def rebuild(self, docs: dict[str, str]) -> None:
        self._tf, self._len, self._df = {}, {}, Counter()
        for cid, text in docs.items():
            toks = tokenize(text)
            self._tf[cid] = Counter(toks)
            self._len[cid] = len(toks)
            self._df.update(set(toks))

    def add(self, cid: str, text: str) -> None:
        self.remove(cid)
        toks = tokenize(text)
        self._tf[cid] = Counter(toks)
        self._len[cid] = len(toks)
        self._df.update(set(toks))

    def remove(self, cid: str) -> None:
        tf = self._tf.pop(cid, None)
        if tf is None:
            return
        self._len.pop(cid, None)
        for t in tf:
            self._df[t] -= 1
            if self._df[t] <= 0:
                del self._df[t]

    def search(self, query: str, k: int) -> list[Candidate]:
        n = len(self._tf)
        if n == 0:
            return []
        avg = sum(self._len.values()) / n or 1.0
        q = tokenize(query)
        scores: dict[str, float] = {}
        for cid, tf in self._tf.items():
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * self._len[cid] / avg)
                s += idf * f * (self.k1 + 1) / denom
            if s > 0:
                scores[cid] = s
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Candidate(cid, sc, r + 1) for r, (cid, sc) in enumerate(ranked)]


class VectorStore(Protocol):
    def reset(self) -> None: ...
    def upsert(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search(self, vector: np.ndarray, k: int) -> list[Candidate]: ...


class MemoryVectorStore:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None

    def reset(self) -> None:
        self._ids, self._mat = [], None

    def upsert(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        self._ids.extend(ids)
        self._mat = vectors if self._mat is None else np.vstack([self._mat, vectors])

    def delete(self, ids: list[str]) -> None:
        if self._mat is None or not ids:
            return
        drop = set(ids)
        keep = [i for i, cid in enumerate(self._ids) if cid not in drop]
        self._ids = [self._ids[i] for i in keep]
        self._mat = self._mat[keep] if keep else None

    def search(self, vector: np.ndarray, k: int) -> list[Candidate]:
        if self._mat is None:
            return []
        sims = self._mat @ vector
        order = np.argsort(-sims, kind="stable")[:k]
        return [Candidate(self._ids[i], float(sims[i]), r + 1) for r, i in enumerate(order)]


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, dim: int) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url)
        self._collection, self._dim = collection, dim

    @staticmethod
    def _pid(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def reset(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection, vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE)
        )

    def upsert(self, ids: list[str], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        if not ids:
            return
        pts = [
            PointStruct(id=self._pid(i), vector=v.tolist(), payload={"chunk_id": i})
            for i, v in zip(ids, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=pts)

    def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        if ids:
            selector = PointIdsList(points=[self._pid(i) for i in ids])
            self._client.delete(self._collection, points_selector=selector)

    def search(self, vector: np.ndarray, k: int) -> list[Candidate]:
        res = self._client.query_points(self._collection, query=vector.tolist(), limit=k)
        return [
            Candidate(p.payload["chunk_id"], float(p.score), r + 1)
            for r, p in enumerate(res.points)
        ]


def rrf_fuse(lists: list[list[Candidate]], k: int, rrf_k: int = 60) -> list[Candidate]:
    scores: dict[str, float] = {}
    for lst in lists:
        for c in lst:
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + c.rank)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [Candidate(cid, sc, r + 1) for r, (cid, sc) in enumerate(ranked)]
