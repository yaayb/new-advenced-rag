"""Answer generators behind a Protocol."""

import re
from typing import Protocol

import httpx

from .text import tokenize


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[str]) -> str: ...


class ExtractiveGenerator:
    """Model-free generator: returns the context sentences that best overlap the question."""

    name = "extractive"

    def generate(self, question: str, contexts: list[str]) -> str:
        q = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        pos = 0
        for ctx in contexts:
            for sent in re.split(r"(?<=[.!?])\s+", ctx):
                toks = set(tokenize(sent))
                if toks and q:
                    scored.append((len(q & toks) / len(q), -pos, sent.strip()))
                pos += 1
        scored.sort(reverse=True)
        best = [s for sc, _, s in scored[:2] if sc > 0]
        return " ".join(best) if best else "Aucun élément de preuve trouvé dans le contexte récupéré."


class OpenAICompatGenerator:
    """Calls any OpenAI-compatible /chat/completions server (Ollama, vLLM, LM Studio, ...)."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 120.0) -> None:
        self.name = f"openai:{model}"
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model, self._key, self._timeout = model, api_key, timeout

    def generate(self, question: str, contexts: list[str]) -> str:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        messages = [
            {
                "role": "system",
                "content": "Réponds en français, uniquement à partir du contexte numéroté. Cite les sources comme [1]. "
                "Si le contexte est insuffisant, dis-le.",
            },
            {"role": "user", "content": f"Contexte :\n{ctx}\n\nQuestion : {question}"},
        ]
        resp = httpx.post(
            self._url,
            json={"model": self._model, "messages": messages, "temperature": 0},
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()
