"""final_response — Day 4 STUB (real generation is Day 5).

Prints what each store returned rather than writing prose. That is deliberate:
until Day 5's rerank fuses the two lists and the LLM is constrained to cite only
retrieved films, any generated answer would be ungrounded — and an ungrounded
answer that reads well is the exact failure this whole architecture exists to
prevent (contract §5).

It also reports store FAILURES separately from empty results, because both look
like "no movies" and mean opposite things.
"""

from __future__ import annotations

from core.logger import get_logger
from agent.state import AgentState

log = get_logger("final_response")


def _format(rows: list[dict], enrichment: dict, limit: int = 8) -> str:
    if not rows:
        return "    (none)"
    lines = []
    for r in rows[:limit]:
        extra = ""
        if r.get("score") is not None:
            extra = f"score={r['score']:.3f}"
        elif r.get("match_reason"):
            extra = r["match_reason"]
        links = enrichment.get(r.get("tmdb_id"), {})
        directors = links.get("directors") or r.get("directors") or []
        coll = links.get("collection_name") or r.get("collection_name")
        tail = f"  dir={directors[:2]}" if directors else ""
        tail += f"  franchise={coll}" if coll else ""
        lines.append(f"    {r['rank']:>2}. {r['title'][:40]:<40} {str(r.get('year') or '????'):<6}{extra}{tail}")
    return "\n".join(lines)


def final_response(state: AgentState) -> dict:
    vector_rows = state.get("vector_results") or []
    graph_rows = state.get("graph_results") or []
    enrichment = state.get("enrichment") or {}
    errors = state.get("retrieval_errors") or []

    # The overlap Day 5's fusion will reward. Computed here only to make it
    # visible today — rerank_node owns this logic tomorrow.
    overlap = {r["tmdb_id"] for r in vector_rows} & {r["tmdb_id"] for r in graph_rows}
    overlap_titles = [r["title"] for r in vector_rows if r["tmdb_id"] in overlap]

    parts = [
        f"[Day 4 stub] intent={state.get('intent')}  lead_engine={state.get('lead_engine')}",
        f"  refined: {state.get('refined_query')}",
        "",
        f"  VECTOR ({len(vector_rows)}):",
        _format(vector_rows, enrichment),
        "",
        f"  GRAPH ({len(graph_rows)}):",
        _format(graph_rows, enrichment),
    ]

    if overlap:
        parts += ["", f"  IN BOTH LISTS ({len(overlap)}): {overlap_titles}"]

    if errors:
        # Never let an outage read as an empty catalogue.
        parts += ["", f"  ⚠ STORE FAILURES: {errors}"]
    elif not vector_rows and not graph_rows:
        parts += ["", "  Nothing matched — and both stores answered. This is a real"
                      " 'no such films', not an outage."]

    log.info(
        "final_response_stub",
        vector=len(vector_rows),
        graph=len(graph_rows),
        overlap=len(overlap),
        enriched=len(enrichment),
        errors=len(errors),
    )
    return {"response": "\n".join(parts), "citations": [], "trace": ["final_response(stub)"]}
