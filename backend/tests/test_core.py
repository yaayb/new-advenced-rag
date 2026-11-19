import pytest
from fastapi.testclient import TestClient

from evidencelab import api
from evidencelab.config import Settings
from evidencelab.engine import Engine
from evidencelab.evaluation import ingest_corpus, load_dataset, run_eval
from evidencelab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from evidencelab.retrieval import BM25Index, Candidate, rrf_fuse
from evidencelab.text import chunk_text

DATA = str(__import__("pathlib").Path(__file__).parents[2] / "eval_data")


@pytest.fixture
def engine() -> Engine:
    return Engine.from_settings(Settings(db_path=":memory:", eval_data_dir=DATA))


def test_chunker_overlap_and_bounds() -> None:
    text = " ".join(f"Sentence number {i} has some words." for i in range(40))
    chunks = chunk_text(text, 30, 5)
    assert len(chunks) > 3
    assert all(c.token_count <= 40 for c in chunks)
    with pytest.raises(ValueError):
        chunk_text("x", 10, 10)


def test_bm25_ranks_exact_match_first() -> None:
    idx = BM25Index()
    idx.rebuild({"a": "cats purr loudly", "b": "dogs bark at night", "c": "stock markets fell"})
    res = idx.search("dogs bark", 3)
    assert res[0].chunk_id == "b" and res[0].rank == 1


def test_rrf_prefers_items_in_both_lists() -> None:
    a = [Candidate("x", 1, 1), Candidate("y", 1, 2)]
    b = [Candidate("z", 1, 1), Candidate("y", 1, 2)]
    assert rrf_fuse([a, b], 3)[0].chunk_id == "y"


def test_metrics() -> None:
    assert recall_at_k(["a", "b"], {"b", "c"}, 2) == 0.5
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a"], {"a"}, 3) == 1.0


def test_query_reports_stage_latency(engine: Engine) -> None:
    ingest_corpus(engine, DATA)
    res = engine.query("When was the lighthouse completed?", "hybrid", 3)
    assert res.chunks[0].doc_id == "lighthouse"
    assert {"bm25_ms", "vector_search_ms", "fusion_ms", "generate_ms", "total_ms"} <= set(
        res.stage_ms
    )
    assert res.context_tokens == sum(c.token_count for c in res.chunks)


def test_eval_runs_all_modes(engine: Engine) -> None:
    ingest_corpus(engine, DATA)
    qs = load_dataset(DATA)
    assert len(qs) == 16
    for mode in ("bm25", "dense", "hybrid"):
        out = run_eval(engine, qs, mode, 3)  # type: ignore[arg-type]
        assert out["summary"]["n_questions"] == 16
        assert 0.0 <= out["summary"]["mrr"] <= 1.0


def test_api_flow(monkeypatch: pytest.MonkeyPatch, engine: Engine) -> None:
    monkeypatch.setattr(api, "_engine", engine)
    monkeypatch.setattr(api, "get_settings", lambda: Settings(eval_data_dir=DATA))
    c = TestClient(api.create_app())
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.post("/api/eval/run", json={}).status_code == 409
    assert c.post("/api/eval/load-benchmark").json()["documents"] == 8
    runs = c.post("/api/eval/run", json={"k": 3}).json()
    assert [r["mode"] for r in runs] == ["bm25", "dense", "hybrid"]
    assert c.get(f"/api/eval/runs/{runs[0]['run_id']}").json()["details"]
    r = c.post("/api/documents", json={"title": "Test Doc", "text": "Pandas eat bamboo."})
    assert r.status_code == 201
    assert c.delete("/api/documents/test-doc").status_code == 204
    assert c.delete("/api/documents/test-doc").status_code == 404
    assert c.post("/api/query", json={"question": ""}).status_code == 422


def test_french_accents_are_tokenised_and_retrieved() -> None:
    e = Engine.from_settings(Settings(db_path=":memory:"))
    e.ingest("Cuisine", "La pâtisserie française utilise du beurre. Le café est servi chaud.")
    e.ingest("Sport", "Le football se joue à onze joueurs sur un terrain.")
    res = e.query("Quel est le café servi ?", "bm25", 1)
    assert res.chunks[0].doc_id == "cuisine"
    assert "café" in res.answer
