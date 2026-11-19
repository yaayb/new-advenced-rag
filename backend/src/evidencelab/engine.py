"""RAG engine: ingestion + staged query pipeline with per-stage timing."""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import Settings
from .db import Database
from .llm import ExtractiveGenerator, Generator, OpenAICompatGenerator
from .retrieval import (
    BM25Index,
    Candidate,
    Embedder,
    HashingEmbedder,
    MemoryVectorStore,
    QdrantVectorStore,
    VectorStore,
    rrf_fuse,
)
from .text import chunk_text, count_tokens

Mode = Literal["bm25", "dense", "hybrid"]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "doc"


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    token_count: int
    bm25_rank: int | None = None
    dense_rank: int | None = None


@dataclass
class QueryResult:
    question: str
    mode: str
    answer: str
    chunks: list[RetrievedChunk]
    stage_ms: dict[str, float] = field(default_factory=dict)
    context_tokens: int = 0
    generator: str = ""


class Engine:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        settings: Settings,
    ) -> None:
        self.db, self.embedder, self.store = db, embedder, store
        self.generator, self.settings = generator, settings
        self.bm25 = BM25Index()
        self._chunks: dict[str, dict[str, Any]] = {}
        self.reindex()

    @classmethod
    def from_settings(cls, s: Settings) -> "Engine":
        embedder = HashingEmbedder(s.embedding_dim)
        store: VectorStore = (
            QdrantVectorStore(s.qdrant_url, s.qdrant_collection, s.embedding_dim)
            if s.vector_backend == "qdrant"
            else MemoryVectorStore()
        )
        gen: Generator = (
            OpenAICompatGenerator(s.llm_base_url, s.llm_model, s.llm_api_key)
            if s.llm_backend == "openai"
            else ExtractiveGenerator()
        )
        return cls(Database(s.db_path), embedder, store, gen, s)

    def reindex(self) -> None:
        """Rebuild BM25 and vector indexes from the SQLite source of truth."""
        rows = self.db.all_chunks()
        self._chunks = {r["chunk_id"]: r for r in rows}
        self.bm25.rebuild({r["chunk_id"]: r["text"] for r in rows})
        self.store.reset()
        if rows:
            self.store.upsert(
                [r["chunk_id"] for r in rows], self.embedder.embed([r["text"] for r in rows])
            )

    def ingest(self, title: str, text: str, doc_id: str | None = None) -> tuple[str, int]:
        doc_id = doc_id or slugify(title)
        raw = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not raw:
            raise ValueError("le document ne contient aucun texte")
        self.db.add_document(
            doc_id, title, [(f"{doc_id}#{c.index}", c.index, c.text, c.token_count) for c in raw]
        )
        self.reindex()
        return doc_id, len(raw)

    def delete(self, doc_id: str) -> bool:
        ok = self.db.delete_document(doc_id)
        if ok:
            self.reindex()
        return ok

    def retrieve(self, question: str, mode: Mode, k: int) -> tuple[list[Candidate], dict[str, Any]]:
        timings: dict[str, float] = {}
        pool = max(k * 4, 20)
        bm: list[Candidate] = []
        dn: list[Candidate] = []
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            bm = self.bm25.search(question, pool)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
        if mode in ("dense", "hybrid"):
            t = time.perf_counter()
            vec = self.embedder.embed([question])[0]
            timings["embed_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dn = self.store.search(vec, pool)
            timings["vector_search_ms"] = (time.perf_counter() - t) * 1000
        if mode == "hybrid":
            t = time.perf_counter()
            final = rrf_fuse([bm, dn], k)
            timings["fusion_ms"] = (time.perf_counter() - t) * 1000
        else:
            final = (bm or dn)[:k]
        ranks = {"bm25": {c.chunk_id: c.rank for c in bm}, "dense": {c.chunk_id: c.rank for c in dn}}
        return final, {"timings": timings, "ranks": ranks}

    def query(
        self, question: str, mode: Mode = "hybrid", k: int = 5, generate: bool = True
    ) -> QueryResult:
        t0 = time.perf_counter()
        final, info = self.retrieve(question, mode, k)
        chunks = []
        for c in final:
            row = self._chunks[c.chunk_id]
            chunks.append(
                RetrievedChunk(
                    c.chunk_id, row["doc_id"], row["text"], c.score, c.rank, row["token_count"],
                    info["ranks"]["bm25"].get(c.chunk_id), info["ranks"]["dense"].get(c.chunk_id),
                )
            )
        stage = dict(info["timings"])
        answer = ""
        if generate:
            t = time.perf_counter()
            answer = self.generator.generate(question, [c.text for c in chunks])
            stage["generate_ms"] = (time.perf_counter() - t) * 1000
        stage["total_ms"] = (time.perf_counter() - t0) * 1000
        return QueryResult(
            question, mode, answer, chunks,
            {k_: round(v, 3) for k_, v in stage.items()},
            sum(c.token_count for c in chunks), self.generator.name,
        )

    @staticmethod
    def count_tokens(text: str) -> int:
        return count_tokens(text)
