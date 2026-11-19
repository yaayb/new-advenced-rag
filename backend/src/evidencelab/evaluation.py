"""Evaluation harness: ingest benchmark corpus, run ground-truth QA, aggregate metrics."""

import json
import statistics
from pathlib import Path
from typing import Any

from . import metrics as m
from .engine import Engine, Mode


def load_dataset(data_dir: str) -> list[dict[str, Any]]:
    path = Path(data_dir) / "qa.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["questions"])


def ingest_corpus(engine: Engine, data_dir: str) -> int:
    n = 0
    for f in sorted((Path(data_dir) / "corpus").glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].strip() if text.strip() else f.stem
        engine.ingest(title, text, doc_id=f.stem)
        n += 1
    return n


def run_eval(engine: Engine, questions: list[dict[str, Any]], mode: Mode, k: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        res = engine.query(q["question"], mode=mode, k=k)
        ranked = m.dedupe_keep_order([c.doc_id for c in res.chunks])
        rel = set(q["relevant_docs"])
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "relevant_docs": sorted(rel),
                "retrieved_docs": ranked,
                "hit": m.hit_at_k(ranked, rel, k),
                "recall": m.recall_at_k(ranked, rel, k),
                "mrr": m.reciprocal_rank(ranked, rel),
                "ndcg": m.ndcg_at_k(ranked, rel, k),
                "answer_coverage": m.keyword_coverage(res.answer, q.get("answer_keywords", [])),
                "context_tokens": res.context_tokens,
                "latency_ms": res.stage_ms["total_ms"],
                "answer": res.answer,
            }
        )
    mean = lambda key: round(statistics.fmean(r[key] for r in rows), 4) if rows else 0.0  # noqa: E731
    lat = sorted(r["latency_ms"] for r in rows)
    summary = {
        "mode": mode, "k": k, "n_questions": len(rows),
        "hit_rate": mean("hit"), "recall": mean("recall"), "mrr": mean("mrr"),
        "ndcg": mean("ndcg"), "answer_coverage": mean("answer_coverage"),
        "avg_context_tokens": mean("context_tokens"),
        "latency_ms_mean": round(statistics.fmean(lat), 3) if lat else 0.0,
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3) if lat else 0.0,
        "generator": engine.generator.name,
    }
    return {"summary": summary, "details": rows}
