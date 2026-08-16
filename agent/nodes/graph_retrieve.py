"""graph_retrieve — Cypher over people, titles, genres (contract §2.3).

Gopal the librarian: he has read nothing, but the card catalogue is exact. He
returns titles you did not have. When he has nothing, he says nothing — an empty
result from him is a real answer, not a failure to understand.

Contrast with graph_enrich, which never returns a new title.

Two strategies, in order:
  1. EXACT-FACT SHORTCUT — named titles resolve straight to Movie nodes. This is
     the "who directed Whiplash" path: no search, just a lookup.
  2. CONSTRAINT SEARCH — people / genres / year / rating as hard requirements.

Both may run; results are merged and deduped by tmdb_id, shortcut hits first
because a directly-named film outranks one found by constraint.
"""

from __future__ import annotations

from core.logger import get_logger
from agent.state import AgentState
from stores import neo4j_client

log = get_logger("graph_retrieve")

TOP_K = 20


def graph_retrieve(state: AgentState) -> dict:
    filters = state.get("filters") or {}
    entities = state.get("entities") or {}

    titles = entities.get("titles") or []
    people = filters.get("people") or []
    genres = filters.get("genres") or []
    year_range = filters.get("year_range")
    min_rating = filters.get("min_rating")

    rows: list[dict] = []
    seen: set[int] = set()
    reasons: dict[int, str] = {}

    try:
        # 1. Exact-fact shortcut.
        for row in neo4j_client.movies_by_titles(titles):
            if row["tmdb_id"] not in seen:
                seen.add(row["tmdb_id"])
                reasons[row["tmdb_id"]] = f"named title: {row.get('matched_title')}"
                rows.append(row)

        # 2. Constraint search — only worth running if there is a constraint.
        # Without one this would return "the 20 most popular films", which is not
        # an answer to anything the user asked.
        if people or genres or year_range or min_rating:
            for row in neo4j_client.search_movies(
                people=people,
                genres=genres,
                year_range=year_range,
                min_rating=min_rating,
                limit=TOP_K,
            ):
                if row["tmdb_id"] not in seen:
                    seen.add(row["tmdb_id"])
                    bits = []
                    if people:
                        bits.append("+".join(people))
                    if genres:
                        bits.append("/".join(genres))
                    reasons[row["tmdb_id"]] = "constraints: " + " ".join(bits or ["year/rating"])
                    rows.append(row)
    except Exception as exc:  # noqa: BLE001
        log.error("graph_retrieve_failed", error=type(exc).__name__, detail=str(exc)[:200])
        return {
            "retrieval_errors": [{"store": "neo4j", "error": type(exc).__name__}],
            "trace": ["graph_retrieve(FAILED)"],
        }

    results = [
        {**row, "rank": i, "source": "graph", "match_reason": reasons.get(row["tmdb_id"])}
        for i, row in enumerate(rows[:TOP_K], start=1)
    ]

    log.info(
        "graph_retrieved",
        count=len(results),
        titles=titles,
        people=people,
        genres=genres,
        top=[r["title"] for r in results[:3]],
    )
    return {"graph_results": results, "trace": [f"graph_retrieve({len(results)})"]}
