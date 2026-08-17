"""franchise_node — the sequel timeline for any cited film (contract §2.3).

The knowledge-graph showcase, and the cheapest feature in the build: the work was
done at ingestion when every film got a PART_OF edge to its Collection. Asking
for "the whole series, in order" is then one hop out and one hop back.

A vector store cannot produce this at any price. "Ordered members of the same
franchise" is not a similarity — Qdrant would return films that FEEL like The
Fellowship of the Ring, which is a different question with a different answer.

Runs after final_response, over the films the answer actually CITED — not over
everything retrieved. A timeline for a film the user was never shown is noise.
"""

from __future__ import annotations

from core.logger import get_logger
from agent.state import AgentState
from stores import neo4j_client

log = get_logger("franchise_node")

# One timeline is a nice touch; four is a wall of text.
MAX_TIMELINES = 2
MIN_FILMS_IN_SERIES = 2


def franchise_node(state: AgentState) -> dict:
    citations = state.get("citations") or []
    if not citations:
        return {"trace": ["franchise(skipped)"]}

    timelines: list[dict] = []
    seen_collections: set[int] = set()

    try:
        for citation in citations:
            if len(timelines) >= MAX_TIMELINES:
                break
            rows = neo4j_client.franchise_timeline(citation["tmdb_id"])
            if len(rows) < MIN_FILMS_IN_SERIES:
                continue  # standalone film, or the only one of its series we hold

            collection_id = rows[0].get("collection_id")
            if collection_id in seen_collections:
                continue  # two cited films from one series — show it once
            seen_collections.add(collection_id)

            timelines.append(
                {
                    "collection_id": collection_id,
                    "collection_name": rows[0].get("collection_name"),
                    "seed_tmdb_id": citation["tmdb_id"],
                    "films": [
                        {
                            "tmdb_id": r["tmdb_id"],
                            "title": r["title"],
                            "year": r["year"],
                            "is_seed": r.get("is_seed", False),
                            "poster_path": r.get("poster_path"),
                        }
                        for r in rows
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001 — a bonus feature must never break the answer
        log.error("franchise_failed", error=type(exc).__name__)
        return {"trace": ["franchise(FAILED)"]}

    if not timelines:
        log.info("franchise_none", cited=len(citations))
        return {"trace": ["franchise(0)"]}

    # Append to the response rather than regenerating it: the answer is already
    # written and validated, and re-prompting risks losing its grounding.
    blocks = []
    for t in timelines:
        entries = [
            f"    {'▸' if f['is_seed'] else ' '} {f['year'] or '????'}  {f['title']}"
            for f in t["films"]
        ]
        blocks.append(f"\n**{t['collection_name']}** — the full series:\n" + "\n".join(entries))

    log.info(
        "franchise_timelines",
        count=len(timelines),
        collections=[t["collection_name"] for t in timelines],
        films=sum(len(t["films"]) for t in timelines),
    )

    return {
        "franchise": timelines,
        "response": (state.get("response") or "") + "\n" + "\n".join(blocks),
        "trace": [f"franchise({len(timelines)})"],
    }
