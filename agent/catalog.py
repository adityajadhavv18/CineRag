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
