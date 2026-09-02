"""FastAPI application (contract §4, Day 7).

    uv run uvicorn server.main:app --reload
    uv run python -m server.main            # same thing, no reload

Lifespan does the expensive work ONCE at startup rather than on the first
unlucky request: opening store connections, compiling the LangGraph, and warming
the catalogue vocabulary. Without it the first caller pays several seconds of
setup and every subsequent one wonders why it was slow.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import configure_langsmith, settings
from core.logger import get_logger
from server.routes import router
from server.routes_catalog import router as catalog_router

log = get_logger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracing = configure_langsmith()

    warmed: dict[str, bool] = {}

    # Compile the graph now. Compilation validates the wiring, so a broken edge
    # fails at boot rather than on a user's request.
    try:
        from agent.graph import get_agent

        get_agent()
        warmed["graph"] = True
    except Exception as exc:  # noqa: BLE001
        log.error("graph_compile_failed", error=type(exc).__name__)
        warmed["graph"] = False

    # Open store connections and warm the genre vocabulary (an lru_cache that
    # would otherwise be populated by the first request).
    for name, warm in (
        ("qdrant", lambda: __import__("stores.qdrant_client", fromlist=["x"]).collection_stats()),
        ("neo4j", lambda: __import__("stores.neo4j_client", fromlist=["x"]).graph_stats()),
        ("catalogue", lambda: __import__("agent.catalog", fromlist=["x"]).genres()),
    ):
        try:
            warm()
            warmed[name] = True
        except Exception as exc:  # noqa: BLE001
            # A store being down at boot must not stop the server: the agent
            # degrades per-request, and /health reports what is actually up.
            log.warning("warmup_failed", component=name, error=type(exc).__name__)
            warmed[name] = False

    log.info("server_ready", tracing=tracing, warmed=warmed,
             collection=settings.qdrant_collection)
    yield

    try:
        from stores.neo4j_client import get_driver

        get_driver().close()
    except Exception:  # noqa: BLE001
        pass
    log.info("server_shutdown")


app = FastAPI(
    title="CineRAG",
    version="1.0.0",
    description="Intent-driven movie recommendations over a vector store + knowledge graph.",
    lifespan=lifespan,
)

# The Netflix-clone frontend is a separate origin (contract §9). Wide-open in dev;
# a real deployment should list its actual origins here instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
# The browse surface (contract §9). Separate router because it shares nothing
# with the agent but the prefix: no LLM, no intent, just reads off the stores.
app.include_router(catalog_router)


@app.get("/")
def root() -> dict:
    return {
        "service": "CineRAG",
        "docs": "/docs",
        "chat": "POST /api/v1/chat",
        "browse": "GET /api/v1/browse",
    }


def main() -> None:
    import uvicorn

    uvicorn.run("server.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
