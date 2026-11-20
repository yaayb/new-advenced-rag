import httpx
import numpy as np
import pytest

from evidencelab import retrieval
from evidencelab.config import Settings
from evidencelab.engine import Engine
from evidencelab.llm import OpenAICompatGenerator
from evidencelab.retrieval import OpenAICompatEmbedder


class FakeResp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._p, self.status_code = payload, status

    def json(self) -> dict:
        return self._p

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]


def fake_embeddings(url: str, json: dict, **_: object) -> FakeResp:
    assert url.endswith("/embeddings")
    data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(json["input"])]
    return FakeResp({"data": data[::-1]})  # out of order on purpose


def test_openai_embedder_orders_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", fake_embeddings)
    e = OpenAICompatEmbedder("http://x/v1", "m", "k", batch_size=2)
    assert e.dim == 3
    out = e.embed(["a", "bbb", "cc"])
    assert out.shape == (3, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert out[1][0] > out[0][0]  # "bbb" (len 3) keeps its own row


def test_generator_parses_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def post(url: str, json: dict, headers: dict, **_: object) -> FakeResp:
        seen.update(url=url, auth=headers["Authorization"], model=json["model"])
        return FakeResp({"choices": [{"message": {"content": " Réponse [1] "}}]})

    monkeypatch.setattr(httpx, "post", post)
    g = OpenAICompatGenerator("https://api.example/v1/", "llama", "KEY")
    assert g.generate("q", ["ctx"]) == "Réponse [1]"
    assert seen == {"url": "https://api.example/v1/chat/completions", "auth": "Bearer KEY", "model": "llama"}


def test_llm_outage_falls_back_to_extractive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp({}, 429))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    e = Engine.from_settings(Settings(db_path=":memory:"))
    e.generator = OpenAICompatGenerator("http://x/v1", "m", "k")
    e.ingest("Zephyr", "Le chef de projet est Amira Haddad.")
    res = e.query("Qui est le chef de projet ?", "bm25", 1)
    assert "Amira" in res.answer and "repli" in res.generator and res.warning


def test_embedding_api_down_at_startup_uses_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp({}, 503))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    e = Engine.from_settings(Settings(db_path=":memory:", embedding_backend="openai"))
    assert isinstance(e.embedder, retrieval.HashingEmbedder)


def test_hybrid_degrades_to_bm25_when_dense_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    e = Engine.from_settings(Settings(db_path=":memory:"))
    e.ingest("Zephyr", "Le chef de projet est Amira Haddad.")

    def boom(_: list[str]) -> np.ndarray:
        raise RuntimeError("down")

    monkeypatch.setattr(e.embedder, "embed", boom)
    res = e.query("chef de projet", "hybrid", 1)
    assert res.chunks and res.warning
    with pytest.raises(RuntimeError):
        e.query("chef de projet", "dense", 1)
