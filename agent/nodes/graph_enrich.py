"""graph_enrich — link signals for movies we ALREADY found (contract §2.2).

Gopal annotating Vera's picks. He is handed five books and asked "who wrote
these, any of them in a series?" — he answers about those five and adds none.

The one rule that makes this node what it is: it NEVER returns a new tmdb_id.
Everything it produces is keyed by an id that some retriever already surfaced.
Break that rule and it silently becomes a second retriever, and the fusion on
Day 5 starts rewarding movies that only one store ever actually chose.

Skipped when lead_engine == "graph": those rows arrived from Cypher already
carrying genres, directors, cast and collection, so there is nothing to add.
"""

from __future__ import annotations

from core.logger import get_logger
from agent.state import AgentState
from stores import neo4j_client

log = get_logger("graph_enrich")


def graph_enrich(state: AgentState) -> dict:
    lead = state.get("lead_engine", "vector")

    if lead == "graph":
        log.info("enrich_skipped", reason="graph-led rows already carry their links")
        return {"trace": ["graph_enrich(skipped)"]}

    # Enrich everything we hold, from either store. For lead="both" that is the
    # merged set, per contract §2.2.
    candidates = list(state.get("vector_results") or []) + list(state.get("graph_results") or [])
    ids = list({c["tmdb_id"] for c in candidates if c.get("tmdb_id")})

    if not ids:
        log.info("enrich_skipped", reason="no candidates to enrich")
        return {"trace": ["graph_enrich(0)"]}

    try:
        enrichment = neo4j_client.enrich_by_ids(ids)
    except Exception as exc:  # noqa: BLE001
        # Enrichment is additive, so losing it degrades quality without breaking
        # the request — we still have the candidates themselves.
        log.error("graph_enrich_failed", error=type(exc).__name__, detail=str(exc)[:200])
        return {
            "retrieval_errors": [{"store": "neo4j", "error": type(exc).__name__, "stage": "enrich"}],
            "trace": ["graph_enrich(FAILED)"],
        }

    linked = sum(1 for v in enrichment.values() if v.get("collection_id"))
    log.info("graph_enriched", requested=len(ids), returned=len(enrichment), in_collection=linked)

    return {"enrichment": enrichment, "trace": [f"graph_enrich({len(enrichment)})"]}
