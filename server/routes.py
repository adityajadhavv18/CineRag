"""POST /api/v1/chat and GET /api/v1/health (contract §4, Day 7)."""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent.graph import run as run_agent
from core.logger import get_logger
from server.schemas import ChatRequest, ChatResponse, HealthResponse, Source
from server.streaming import sse_stream

log = get_logger("api")
router = APIRouter(prefix="/api/v1")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one turn through the agent.

    Deliberately `def`, not `async def`. Our nodes are synchronous — the Qdrant,
    Neo4j and OpenAI clients all block — so declaring this `async` would run them
    directly on the event loop and stall every other request for the duration.
    FastAPI runs a plain `def` endpoint in a threadpool, which is exactly right
    for blocking work.
    """
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()

    # Bind the id to this request's logging context, so every node's structured
    # log line for this turn carries it (guideline C). Without it, concurrent
    # requests interleave in the log and no line can be traced to a caller.
    structlog.contextvars.bind_contextvars(request_id=request_id)

    try:
        log.info("request_start", chars=len(request.message),
                 history_turns=len(request.history))

        state = run_agent(
            request.message,
            [turn.model_dump() for turn in request.history],
        )

        errors = state.get("retrieval_errors") or []
        sources = [
            Source(
                tmdb_id=c["tmdb_id"],
                title=c["title"],
                year=c.get("year"),
                overview=_overview_for(state, c["tmdb_id"]),
                poster_path=c.get("poster_path"),
                backdrop_path=c.get("backdrop_path"),
            )
            for c in (state.get("citations") or [])
        ]

        response = ChatResponse(
            response=state.get("response") or "",
            intent=state.get("intent"),
            lead_engine=state.get("lead_engine"),
            sources=sources,
            franchise=state.get("franchise") or [],
            clarification=state.get("clarification"),
            degraded=bool(errors),
            trace=state.get("trace") or [],
        )

        log.info(
            "request_complete",
            intent=response.intent,
            lead_engine=response.lead_engine,
            sources=len(response.sources),
            degraded=response.degraded,
            ms=round((time.perf_counter() - started) * 1000),
        )
        return response

    except Exception as exc:  # noqa: BLE001
        # Last line of defence. Every node already degrades internally, so
        # reaching here means something genuinely unexpected — log it with the
        # request id and return a generic message rather than leaking a traceback.
        log.exception("request_failed", error=type(exc).__name__, request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong handling that request (ref {request_id}).",
        ) from exc
    finally:
        structlog.contextvars.clear_contextvars()


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """The same turn as POST /chat, delivered as Server-Sent Events.

    Both endpoints exist on purpose. This one is for the UI, where the wait is
    the problem; `/chat` stays the single-blob API that the eval harness, the
    tests and scripts/chat.py want — none of them benefit from reading an answer
    a token at a time, and none of them had to change for this.

    No `response_model`: the body is a stream of frames, not one document. The
    event shapes are documented in server/streaming.py.
    """
    request_id = uuid.uuid4().hex[:8]
    log.info("stream_start", request_id=request_id, chars=len(request.message),
             history_turns=len(request.history))

    return StreamingResponse(
        sse_stream(
            request.message,
            [turn.model_dump() for turn in request.history],
            request_id,
        ),
        media_type="text/event-stream",
        headers={
            # An SSE body must reach the client in pieces, so anything that
            # buffers it defeats the whole endpoint — and does so SILENTLY, with
            # the answer simply arriving all at once. These two ask the usual
            # culprits not to. The third is that no compression middleware may be
            # added to this app without excluding this route (see server/main.py).
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _overview_for(state: dict, tmdb_id: int) -> str | None:
    """Pull the stored plot text for a cited film, so cards can show a blurb.

    Read from the reranked candidates rather than re-querying a store: the text
    is already in hand, and a second lookup per card would be pure waste.
    """
    for row in state.get("reranked") or []:
        if row.get("tmdb_id") == tmdb_id:
            return row.get("overview") or None
    return None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus per-store reachability.

    Reports `degraded` rather than failing when a store is down: the agent still
    answers vector-led queries without Neo4j, so a hard failure here would take a
    partly-working service out of a load balancer for no reason.
    """
    stores: dict[str, bool] = {}
    catalogue_size = None

    try:
        from stores import qdrant_client

        stats = qdrant_client.collection_stats()
        stores["qdrant"] = True
        catalogue_size = stats.get("points")
    except Exception:  # noqa: BLE001
        stores["qdrant"] = False

    try:
        from stores import neo4j_client

        neo4j_client.graph_stats()
        stores["neo4j"] = True
    except Exception:  # noqa: BLE001
        stores["neo4j"] = False

    return HealthResponse(
        status="ok" if all(stores.values()) else "degraded",
        stores=stores,
        catalogue_size=catalogue_size,
    )
