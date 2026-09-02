"""Thin search wrapper over Qdrant (contract §4).

Deliberately thin: embed the query, search, optionally filter on payload, return
plain dicts. Retrieval *policy* lives in the agent nodes from Day 4.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from core.config import settings
from core.llm import embed_query
from core.logger import get_logger

log = get_logger("qdrant_client")


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def build_filter(
    genres: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
    min_rating: float | None = None,
    people: list[str] | None = None,
) -> models.Filter | None:
    """Translate contract §3.4 `filters` into a Qdrant payload filter.

    Everything here is a HARD constraint — `must`. Soft signals (entities) never
    reach this function; they influence scoring, not eligibility (contract §3.4).
    """
    must: list[models.Condition] = []

    if genres:
        # MatchAny on a list payload field = "has at least one of these genres".
        must.append(models.FieldCondition(key="genres", match=models.MatchAny(any=genres)))

    if year_range:
        lo, hi = year_range
        must.append(models.FieldCondition(key="year", range=models.Range(gte=lo, lte=hi)))

    if min_rating is not None:
        must.append(models.FieldCondition(key="rating", range=models.Range(gte=min_rating)))

    if people:
        # A person may be credited as either cast or director, so each name is an
        # OR across both fields, and the names themselves are ANDed together
        # ("with Denzel AND directed by Fuqua" must satisfy both).
        for name in people:
            must.append(
                models.Filter(
                    should=[
                        models.FieldCondition(key="cast_names", match=models.MatchValue(value=name)),
                        models.FieldCondition(key="director", match=models.MatchValue(value=name)),
                    ]
                )
            )

    return models.Filter(must=must) if must else None


def search(
    query: str,
    limit: int = 10,
    query_filter: models.Filter | None = None,
) -> list[dict]:
    """Dense semantic search. Returns payloads plus the cosine score."""
    vector = embed_query(query)
    hits = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points

    results = [{**(h.payload or {}), "score": h.score} for h in hits]
    log.info("vector_search", query=query, returned=len(results),
             filtered=query_filter is not None)
    return results


def similar_to(tmdb_id: int, limit: int = 12) -> list[dict]:
    """Nearest neighbours of a film's OWN stored vector — the "More Like This" row.

    Note there is no `embed_query` call and no text: passing a point id as the
    query makes Qdrant search from the vector it already holds. That costs no
    OpenAI call and, more importantly, compares like with like — searching by the
    film's overview text would embed a *description of* the film rather than
    reusing the embedding the whole catalogue was indexed with (contract §3.2).

    Qdrant does not exclude the query point itself, so it is filtered out here;
    otherwise every row would open with the film you are already looking at.
    """
    hits = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=tmdb_id,
        limit=limit,
        query_filter=models.Filter(must_not=[models.HasIdCondition(has_id=[tmdb_id])]),
        with_payload=True,
    ).points

    results = [{**(h.payload or {}), "score": h.score} for h in hits]
    log.info("vector_similar", tmdb_id=tmdb_id, returned=len(results))
    return results


def get_by_ids(tmdb_ids: list[int]) -> list[dict]:
    """Fetch payloads for known movies without searching — used by rerank/enrich."""
    records = get_client().retrieve(
        collection_name=settings.qdrant_collection, ids=tmdb_ids, with_payload=True
    )
    return [dict(r.payload or {}) for r in records]


def collection_stats() -> dict:
    info = get_client().get_collection(settings.qdrant_collection)
    return {"points": info.points_count, "status": str(info.status)}
