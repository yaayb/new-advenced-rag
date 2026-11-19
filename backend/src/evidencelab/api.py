import hmac
from dataclasses import asdict
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .engine import Engine
from .evaluation import ingest_corpus, load_dataset, run_eval
from .extract import extract_text

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine.from_settings(get_settings())
    return _engine


class IngestRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=50)
    generate: bool = True


class EvalRequest(BaseModel):
    modes: list[Literal["bm25", "dense", "hybrid"]] = ["bm25", "dense", "hybrid"]
    k: int = Field(default=5, ge=1, le=50)


def create_app() -> FastAPI:
    app = FastAPI(title="EvidenceLab", version=__version__)

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):  # type: ignore[no-untyped-def]
        key = get_settings().api_key
        path = request.url.path
        if (
            key
            and request.method != "OPTIONS"
            and path.startswith("/api")
            and path != "/api/health"
            and not hmac.compare_digest(
                request.headers.get("x-api-key", "").encode(), key.encode()
            )
        ):
            return JSONResponse({"detail": "clé API manquante ou invalide"}, status_code=401)
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins.split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/auth/check")
    def auth_check() -> dict:
        return {"ok": True}

    @app.get("/api/health")
    def health(engine: Engine = Depends(get_engine)) -> dict:
        return {
            "status": "ok", "version": __version__,
            "auth_required": bool(get_settings().api_key),
            "vector_backend": get_settings().vector_backend,
            "generator": engine.generator.name,
            "documents": len(engine.db.list_documents()),
        }

    @app.get("/api/documents")
    def documents(engine: Engine = Depends(get_engine)) -> list[dict]:
        return engine.db.list_documents()

    @app.post("/api/documents", status_code=201)
    def add_document(req: IngestRequest, engine: Engine = Depends(get_engine)) -> dict:
        try:
            doc_id, n = engine.ingest(req.title, req.text)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"doc_id": doc_id, "chunks": n}

    @app.post("/api/documents/upload", status_code=201)
    async def upload(file: UploadFile = File(...), engine: Engine = Depends(get_engine)) -> dict:
        raw = await file.read()
        name = file.filename or "upload.txt"
        try:
            text = extract_text(name, raw)
            doc_id, n = engine.ingest(name.rsplit(".", 1)[0], text)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"doc_id": doc_id, "chunks": n}

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, engine: Engine = Depends(get_engine)) -> None:
        if not engine.delete(doc_id):
            raise HTTPException(404, "document introuvable")

    @app.post("/api/query")
    def query(req: QueryRequest, engine: Engine = Depends(get_engine)) -> dict:
        try:
            return asdict(engine.query(req.question, req.mode, req.k, req.generate))
        except Exception as e:  # generator/vector backend outage -> clear 502
            raise HTTPException(502, f"échec du pipeline : {e}") from e

    @app.post("/api/eval/load-benchmark")
    def load_benchmark(engine: Engine = Depends(get_engine)) -> dict:
        s = get_settings()
        try:
            n = ingest_corpus(engine, s.eval_data_dir)
            qs = len(load_dataset(s.eval_data_dir))
        except FileNotFoundError as e:
            raise HTTPException(404, f"données de référence introuvables : {e}") from e
        return {"documents": n, "questions": qs}

    @app.get("/api/eval/questions")
    def questions() -> list[dict]:
        try:
            return load_dataset(get_settings().eval_data_dir)
        except FileNotFoundError as e:
            raise HTTPException(404, "qa.json introuvable") from e

    @app.post("/api/eval/run")
    def eval_run(req: EvalRequest, engine: Engine = Depends(get_engine)) -> list[dict]:
        try:
            qs = load_dataset(get_settings().eval_data_dir)
        except FileNotFoundError as e:
            raise HTTPException(404, "qa.json introuvable") from e
        if not engine.db.list_documents():
            raise HTTPException(409, "aucun document indexé ; chargez d'abord le corpus de référence")
        out = []
        for mode in req.modes:
            res = run_eval(engine, qs, mode, req.k)
            run_id = engine.db.save_run(mode, req.k, res["summary"], res["details"])
            out.append({"run_id": run_id, **res["summary"]})
        return out

    @app.get("/api/eval/runs")
    def runs(engine: Engine = Depends(get_engine)) -> list[dict]:
        return engine.db.list_runs()

    @app.get("/api/eval/runs/{run_id}")
    def run_detail(run_id: int, engine: Engine = Depends(get_engine)) -> dict:
        r = engine.db.get_run(run_id)
        if r is None:
            raise HTTPException(404, "run not found")
        return r

    return app


app = create_app()
