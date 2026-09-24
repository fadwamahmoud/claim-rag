"""Application settings, loaded from environment variables / a .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -------------------------------------------------------------
    # Any provider supported by langchain's `init_chat_model`
    # (groq, openai, anthropic, ollama). Groq and Ollama are free options.
    llm_provider: Literal["groq", "openai", "anthropic", "ollama"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"

    # --- RAG -------------------------------------------------------------
    # "fastembed" runs a small ONNX model locally (downloaded on first use).
    # "hashing" is a dependency-free lexical embedder for offline tests.
    embedding_backend: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    policy_docs_dir: Path = DATA_DIR
    chroma_persist_dir: Path | None = None  # None = in-memory index
    retrieval_k: int = 3

    # --- Tools -----------------------------------------------------------
    claims_db_path: Path = DATA_DIR / "mock_claims.json"

    # --- API -------------------------------------------------------------
    max_message_chars: int = 2000
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
