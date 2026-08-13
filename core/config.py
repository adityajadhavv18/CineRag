"""Single source of truth for configuration (contract §5, §6).

Nothing else in the codebase reads os.environ or hardcodes a URL, key, or model
name. Import `settings` from here instead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = the directory containing this file's parent (core/ -> repo root).
ROOT_DIR = Path(__file__).resolve().parent.parent


class MissingConfigError(RuntimeError):
    """Raised when a script needs a credential that isn't set."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated vars in the shell environment
        case_sensitive=False,
    )

    # ─── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ─── TMDB ─────────────────────────────────────────────────────────────────
    # Two credential types exist. The v4 read-access token (a JWT, sent as a
    # Bearer header) is preferred when present; the v3 key (a query param) is the
    # fallback. Either alone is sufficient — see tmdb_auth() below.
    tmdb_api_key: str = ""
    tmdb_api_read_access_token: str = ""

    @property
    def has_tmdb_credential(self) -> bool:
        return bool(self.tmdb_api_read_access_token or self.tmdb_api_key)

    # ─── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "movies"

    # ─── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ─── LangSmith ────────────────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "cinerag"
    langchain_tracing_v2: bool = False

    # ─── Ingestion ────────────────────────────────────────────────────────────
    catalog_size: int = 5000

    # ─── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = False

    # ─── Derived paths (not env-driven; kept here so nothing hardcodes them) ──
    @property
    def data_dir(self) -> Path:
        return ROOT_DIR / "data"

    @property
    def raw_dir(self) -> Path:
        """Stage-1 cache: one raw TMDB JSON response per movie (contract §3.1a)."""
        return self.data_dir / "raw"

    @property
    def movies_jsonl(self) -> Path:
        """Stage-2 output: the canonical snapshot both stores are built from."""
        return self.data_dir / "movies.jsonl"

    # Credentials default to "" rather than being required fields, so that every
    # script still imports cleanly without a fully-populated .env. Scripts call
    # require() for exactly the keys they need, which turns "missing key" into a
    # precise error at the point of use instead of an import-time wall of red.
    def require(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise MissingConfigError(
                f"Missing required config: {', '.join(n.upper() for n in missing)}. "
                f"Set them in {ROOT_DIR / '.env'} (see .env.example)."
            )


settings = Settings()


def configure_langsmith() -> bool:
    """Export the env vars the LangSmith SDK reads (contract §6, Day 3).

    LangSmith is configured entirely through the environment — LangGraph and the
    wrapped OpenAI client pick these up on their own, so there is no client to
    pass around. We export both spellings of the tracing flag: LANGCHAIN_TRACING_V2
    is the older name the contract lists, LANGSMITH_TRACING is the current one,
    and different library versions read different ones.

    Returns True if tracing is actually on. Called once from agent/graph.py.
    """
    import os

    if not (settings.langchain_tracing_v2 and settings.langsmith_api_key):
        # Explicitly off, so a stale shell env var can't silently enable tracing
        # and start shipping queries somewhere the user didn't ask for.
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return True
