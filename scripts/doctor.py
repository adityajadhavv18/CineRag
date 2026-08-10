"""Environment doctor: what's configured, what's reachable.

Run it any time something feels off:

    uv run python -m scripts.doctor
"""

from __future__ import annotations

import sys

import httpx

from core.config import ROOT_DIR, settings
from core.logger import get_logger

log = get_logger("doctor")

OK, WARN, BAD = "\033[32m✓\033[0m", "\033[33m•\033[0m", "\033[31m✗\033[0m"


def _mask(value: str) -> str:
    if not value:
        return "(not set)"
    return f"{value[:4]}…{value[-4:]} ({len(value)} chars)" if len(value) > 12 else "set"


def check_config() -> None:
    print("\n\033[1mConfiguration\033[0m")
    print(f"  .env file            {ROOT_DIR / '.env'}"
          f"{'' if (ROOT_DIR / '.env').exists() else '   ← MISSING (cp .env.example .env)'}")
    print(f"  chat model           {settings.chat_model}")
    print(f"  embedding model      {settings.embedding_model}")
    print(f"  catalog size         {settings.catalog_size}")
    print(f"  qdrant               {settings.qdrant_url}  collection={settings.qdrant_collection}")
    print(f"  neo4j                {settings.neo4j_uri}  user={settings.neo4j_user}")
    print(f"  langsmith tracing    {settings.langchain_tracing_v2}  project={settings.langsmith_project}")

    print("\n\033[1mCredentials\033[0m")
    for label, value, needed_by in [
        ("TMDB_API_KEY", settings.tmdb_api_key, "Day 1 (ingestion)"),
        ("OPENAI_API_KEY", settings.openai_api_key, "Day 2 (embeddings)"),
        ("NEO4J_PASSWORD", settings.neo4j_password, "Day 1 (docker compose)"),
        ("LANGSMITH_API_KEY", settings.langsmith_api_key, "Day 3 (tracing)"),
    ]:
        mark = OK if value else WARN
        print(f"  {mark} {label:<20} {_mask(value):<28} needed by {needed_by}")


def check_stores() -> bool:
    print("\n\033[1mStores\033[0m")
    all_up = True

    # Qdrant exposes a REST API on 6333; / returns version info.
    try:
        r = httpx.get(settings.qdrant_url, timeout=3.0)
        r.raise_for_status()
        version = r.json().get("version", "?")
        print(f"  {OK} Qdrant reachable at {settings.qdrant_url} (version {version})")
    except Exception as exc:  # noqa: BLE001 — doctor reports, never raises
        all_up = False
        print(f"  {BAD} Qdrant unreachable at {settings.qdrant_url}: {type(exc).__name__}")

    # Neo4j's Bolt port isn't HTTP, so we probe the HTTP endpoint (7474) instead.
    # Day 2 swaps this for a real Bolt handshake once the driver is a dependency.
    neo4j_http = settings.neo4j_uri.replace("bolt://", "http://").replace(":7687", ":7474")
    try:
        r = httpx.get(neo4j_http, timeout=3.0)
        r.raise_for_status()
        version = r.json().get("neo4j_version", "?")
        print(f"  {OK} Neo4j reachable at {neo4j_http} (version {version})")
    except Exception as exc:  # noqa: BLE001
        all_up = False
        print(f"  {BAD} Neo4j unreachable at {neo4j_http}: {type(exc).__name__}")

    if not all_up:
        print("\n  Start them with:  docker compose up -d")
    return all_up


def check_data() -> None:
    print("\n\033[1mData\033[0m")
    raw_count = len(list(settings.raw_dir.glob("*.json"))) if settings.raw_dir.exists() else 0
    print(f"  {OK if raw_count else WARN} raw cache        {raw_count} movies in {settings.raw_dir}")

    if settings.movies_jsonl.exists():
        with settings.movies_jsonl.open() as fh:
            lines = sum(1 for _ in fh)
        print(f"  {OK if lines else WARN} movies.jsonl     {lines} records")
    else:
        print(f"  {WARN} movies.jsonl     (not built yet — run ingest.tmdb_pull then ingest.transform)")


def main() -> int:
    print("\n\033[1m═══ CineRAG doctor ═══\033[0m")
    check_config()
    stores_up = check_stores()
    check_data()
    print()
    # A structured log line, so the doctor itself demonstrates guideline C.
    log.info("doctor_complete", stores_up=stores_up, catalog_size=settings.catalog_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
