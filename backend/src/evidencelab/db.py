"""SQLite persistence for documents, chunks and evaluation runs."""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  idx INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS eval_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  mode TEXT NOT NULL, k INTEGER NOT NULL, summary TEXT NOT NULL, details TEXT NOT NULL);
"""


class Database:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def add_document(self, doc_id: str, title: str, chunks: list[tuple[str, int, str, int]]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._conn.execute("INSERT INTO documents(doc_id, title) VALUES (?, ?)", (doc_id, title))
            self._conn.executemany(
                "INSERT INTO chunks(chunk_id, doc_id, idx, text, token_count) VALUES (?,?,?,?,?)",
                [(cid, doc_id, i, t, n) for cid, i, t, n in chunks],
            )

    def chunk_ids(self, doc_id: str) -> list[str]:
        rows = self._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
        return [r["chunk_id"] for r in rows.fetchall()]

    def delete_document(self, doc_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cur.rowcount > 0

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT d.doc_id, d.title, COUNT(c.chunk_id) AS chunks, COALESCE(SUM(c.token_count),0)"
            " AS tokens FROM documents d LEFT JOIN chunks c USING(doc_id) GROUP BY d.doc_id"
            " ORDER BY d.doc_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def all_chunks(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, idx, text, token_count FROM chunks ORDER BY doc_id, idx"
        ).fetchall()
        return [dict(r) for r in rows]

    def save_run(self, mode: str, k: int, summary: dict[str, Any], details: list[Any]) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO eval_runs(mode, k, summary, details) VALUES (?,?,?,?)",
                (mode, k, json.dumps(summary), json.dumps(details)),
            )
            return int(cur.lastrowid or 0)

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, created_at, mode, k, summary FROM eval_runs ORDER BY run_id DESC"
        ).fetchall()
        return [{**dict(r), "summary": json.loads(r["summary"])} for r in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        r = self._conn.execute("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()
        if r is None:
            return None
        return {**dict(r), "summary": json.loads(r["summary"]), "details": json.loads(r["details"])}
