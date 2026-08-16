"""vector_retrieve — Qdrant dense semantic search (contract §2.3).

Vera the librarian: she has read everything and works on feel. Ask for a mood and
she hands you the nearest books, always — which is exactly her strength and
exactly her danger. She never says "no such book"; there is always a nearest one.

Two inputs, doing different jobs:
  refined_query   the text that gets embedded — this is the "vibe"
  filters         HARD payload constraints that EXCLUDE (contract §3.4)

`entities` deliberately never reaches this node. Soft signals must not exclude.
"""

from __future__ import annotations

from core.logger import get_logger
from agent.state import AgentState
from stores import qdrant_client

log = get_logger("vector_retrieve")

TOP_K = 20


def vector_retrieve(state: AgentState) -> dict:
    query = state.get("refined_query") or state["query"]
    filters = state.get("filters") or {}

    year_range = filters.get("year_range")
    query_filter = qdrant_client.build_filter(
        genres=filters.get("genres"),
        # Qdrant wants a tuple; the intent schema produces a 2-element list.
        year_range=tuple(year_range) if year_range and len(year_range) == 2 else None,
        min_rating=filters.get("min_rating"),
        people=filters.get("people"),
    )

    try:
        hits = qdrant_client.search(query, limit=TOP_K, query_filter=query_filter)
    except Exception as exc:  # noqa: BLE001 — a dead store must not 500 the request
        log.error("vector_retrieve_failed", error=type(exc).__name__, detail=str(exc)[:200])
        return {
            "retrieval_errors": [{"store": "qdrant", "error": type(exc).__name__}],
            "trace": ["vector_retrieve(FAILED)"],
        }

    # rank is 1-based and recorded HERE, at the only place that knows this list's
    # ordering. Day 5's RRF fuses on rank, not on score, because a cosine
    # similarity and a graph row have no common scale.
    results = [
        {**hit, "rank": i, "source": "vector"}
        for i, hit in enumerate(hits, start=1)
    ]

    log.info(
        "vector_retrieved",
        count=len(results),
        filtered=query_filter is not None,
        top=[r["title"] for r in results[:3]],
        query=query[:60],
    )
    return {"vector_results": results, "trace": [f"vector_retrieve({len(results)})"]}
