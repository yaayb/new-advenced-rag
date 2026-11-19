from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVIDENCELAB_", env_file=".env", extra="ignore")

    db_path: str = "data/evidencelab.db"
    # "memory" keeps vectors in-process; "qdrant" uses a Qdrant server.
    vector_backend: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "evidencelab_chunks"
    embedding_dim: int = 512
    chunk_size: int = 120  # tokens (whitespace/word based)
    chunk_overlap: int = 20
    # "extractive" needs no model; "openai" talks to any OpenAI-compatible server (Ollama, vLLM...)
    llm_backend: str = "extractive"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"
    llm_api_key: str = "not-needed"
    eval_data_dir: str = "eval_data"
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    # When set, every /api route except /api/health requires header "X-API-Key: <value>".
    api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
