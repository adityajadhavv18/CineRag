"""Real catalogue vocabulary, for grounding the intent node (contract §5).

The stores match payload values EXACTLY: a filter for "crime" finds nothing when
the catalogue says "Crime", and it fails silently — zero results look identical
to "no such movies". So the LLM must be told the real vocabulary, and whatever
it returns must be normalised against it before it reaches a store.

This is the same principle Day 6's clarification node runs on: options come from
the catalogue, never from the model's imagination.
"""

from __future__ import annotations

from functools import lru_cache

from core.logger import get_logger

log = get_logger("catalog")

# TMDB's genre vocabulary is fixed and small. Kept as a fallback so the agent
# still routes sensibly when Neo4j is unavailable (contract §5: degrade cleanly).
FALLBACK_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "TV Movie", "Thriller", "War", "Western",
]


@lru_cache(maxsize=1)
def genres() -> list[str]:
    """The genre names actually present in the graph, in canonical casing."""
    try:
        from stores import neo4j_client

        rows = neo4j_client._query("MATCH (g:Genre) RETURN g.name AS name ORDER BY name")
        found = [r["name"] for r in rows]
        if found:
            return found
        log.warning("catalog_genres_empty", falling_back=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("catalog_genres_unavailable", error=type(exc).__name__, falling_back=True)
    return FALLBACK_GENRES


@lru_cache(maxsize=1)
def _genre_lookup() -> dict[str, str]:
    """lowercased -> canonical, plus a few aliases users and LLMs actually type."""
    lookup = {g.lower(): g for g in genres()}
    aliases = {
        "sci-fi": "Science Fiction",
        "scifi": "Science Fiction",
        "sci fi": "Science Fiction",
        "romcom": "Romance",
        "rom-com": "Romance",
        "docu": "Documentary",
        "documentaries": "Documentary",
        "biography": "History",
        "biopic": "History",
    }
    for alias, canonical in aliases.items():
        if canonical in genres():
            lookup[alias] = canonical
    return lookup


def probe(genres: list[str] | None = None, limit: int = 5) -> dict:
    """Look at the shelf before asking the question (contract §2.3, Day 6).

    Returns narrowing options that are guaranteed to have films behind them,
    because every one is counted straight out of the graph. An LLM inventing
    plausible-sounding options ("80s martial arts", "Hong Kong action") can offer
    a category matching ZERO of our 4,966 films — the user picks it, retrieval
    returns nothing, and the clarification made the conversation worse.

    Counts are returned alongside each option so the caller can drop thin ones.
    """
    from stores import neo4j_client

    genres = genres or []
    result: dict = {"genres": genres, "pairings": [], "decades": [], "people": [], "total": 0}

    try:
        result["total"] = neo4j_client._query(
            """
            MATCH (m:Movie)
            WHERE $genres = [] OR EXISTS {
                MATCH (m)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $genres
            }
            RETURN count(m) AS total
            """,
            genres=genres,
        )[0]["total"]

        if genres:
            # Which other genres these films are ALSO tagged with — the natural
            # "action, but what kind of action?" axis.
            result["pairings"] = neo4j_client._query(
                """
                MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $genres
                MATCH (m)-[:HAS_GENRE]->(other:Genre) WHERE NOT other.name IN $genres
                RETURN other.name AS option, count(DISTINCT m) AS films
                ORDER BY films DESC LIMIT $limit
                """,
                genres=genres,
                limit=limit,
            )
            result["decades"] = neo4j_client._query(
                """
                MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
                WHERE g.name IN $genres AND m.year IS NOT NULL
                WITH (m.year / 10) * 10 AS decade, count(*) AS films
                WHERE films >= 20
                RETURN decade AS option, films ORDER BY decade DESC LIMIT $limit
                """,
                genres=genres,
                limit=limit,
            )
            result["people"] = neo4j_client._query(
                """
                MATCH (p:Person)-[:ACTED_IN]->(m:Movie)-[:HAS_GENRE]->(g:Genre)
                WHERE g.name IN $genres
                RETURN p.name AS option, count(DISTINCT m) AS films
                ORDER BY films DESC LIMIT $limit
                """,
                genres=genres,
                limit=limit,
            )
        else:
            # Nothing to narrow from at all ("recommend some movies") — offer the
            # catalogue's biggest genres as a starting point.
            result["pairings"] = neo4j_client._query(
                """
                MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
                RETURN g.name AS option, count(m) AS films
                ORDER BY films DESC LIMIT $limit
                """,
                limit=limit,
            )
            result["decades"] = neo4j_client._query(
                """
                MATCH (m:Movie) WHERE m.year IS NOT NULL
                WITH (m.year / 10) * 10 AS decade, count(*) AS films
                WHERE films >= 100
                RETURN decade AS option, films ORDER BY decade DESC LIMIT $limit
                """,
                limit=limit,
            )
    except Exception as exc:  # noqa: BLE001 — a failed probe must still let us ask
        log.warning("probe_failed", error=type(exc).__name__)

    log.info(
        "catalogue_probed",
        seed_genres=genres,
        pairings=len(result["pairings"]),
        decades=len(result["decades"]),
        people=len(result["people"]),
    )
    return result


def normalize_genres(values: list[str] | None) -> list[str]:
    """Map whatever the LLM produced onto real catalogue values, dropping unknowns.

    Dropping is deliberate. An unrecognised genre passed through to a store
    becomes a filter nothing can satisfy, turning a slightly-wrong query into
    zero results — worse than ignoring the term and still retrieving.
    """
    if not values:
        return []
    lookup = _genre_lookup()
    out: list[str] = []
    for value in values:
        canonical = lookup.get(value.strip().lower())
        if canonical and canonical not in out:
            out.append(canonical)
        elif not canonical:
            log.info("genre_dropped", value=value, reason="not in catalogue")
    return out
