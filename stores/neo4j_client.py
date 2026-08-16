"""Thin Cypher wrapper over Neo4j (contract §4).

Deliberately thin: it opens one driver, runs named queries, and returns plain
dicts. All the retrieval *policy* (which query to run for which intent) lives in
the agent nodes from Day 4, not here.
"""

from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase

from core.config import settings
from core.logger import get_logger
from core.text import normalize_name

log = get_logger("neo4j_client")

# Fields every movie-returning query yields, so callers get a consistent shape.
MOVIE_FIELDS = """
    m.tmdb_id AS tmdb_id, m.title AS title, m.year AS year,
    m.rating AS rating, m.popularity AS popularity,
    m.poster_path AS poster_path, m.backdrop_path AS backdrop_path
"""


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    settings.require("neo4j_password")
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    driver.verify_connectivity()
    return driver


def _query(cypher: str, **params) -> list[dict]:
    result = get_driver().execute_query(cypher, **params)
    return [dict(r) for r in result.records]


def movies_by_director(name: str, limit: int = 20) -> list[dict]:
    """Exact filmography. The graph's home turf: no ranking, no guessing."""
    return _query(
        f"""
        MATCH (p:Person)-[:DIRECTED]->(m:Movie)
        WHERE toLower(p.name) = toLower($name)
        RETURN {MOVIE_FIELDS}, p.name AS director, p.person_id AS person_id
        ORDER BY m.year
        LIMIT $limit
        """,
        name=name,
        limit=limit,
    )


def movies_by_actor(name: str, limit: int = 20) -> list[dict]:
    """Filmography with the character played — the character sits on the edge."""
    return _query(
        f"""
        MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
        WHERE toLower(p.name) = toLower($name)
        RETURN {MOVIE_FIELDS}, r.character AS character,
               p.name AS actor, p.person_id AS person_id
        ORDER BY m.year
        LIMIT $limit
        """,
        name=name,
        limit=limit,
    )


def franchise_timeline(tmdb_id: int) -> list[dict]:
    """Every film in this film's collection, in release order, INCLUDING itself.

    The KG showcase (contract §3.3, Day 6): one hop out to the Collection node
    and back down gives the whole sequel chain, ordered, for free.

    Note the two-step MATCH. Writing this as a single path —
        (m)-[:PART_OF]->(c)<-[:PART_OF]-(sibling)
    — silently drops the seed movie, because Cypher's relationship-uniqueness
    rule forbids traversing the same PART_OF edge twice in one path, so
    `sibling` can never bind to `m`. That quietly omits the first film of every
    franchise. Binding the Collection first and then matching all its members
    keeps the seed in the timeline, which is what a timeline means.
    """
    return _query(
        f"""
        MATCH (:Movie {{tmdb_id: $tmdb_id}})-[:PART_OF]->(c:Collection)
        MATCH (m:Movie)-[:PART_OF]->(c)
        RETURN {MOVIE_FIELDS},
               c.name AS collection_name, c.id AS collection_id,
               m.tmdb_id = $tmdb_id AS is_seed
        ORDER BY m.year
        """,
        tmdb_id=tmdb_id,
    )


"""Rows returned by the retrieval queries carry their graph facts inline.

graph_retrieve's rows are self-describing (genres, directors, cast, collection)
so that a graph-led query needs no separate enrichment pass — which is exactly
what contract §2.2 says: for lead_engine="graph", graph_retrieve runs alone.
"""
RETRIEVAL_FIELDS = """
    m.tmdb_id AS tmdb_id, m.title AS title, m.year AS year,
    m.rating AS rating, m.popularity AS popularity,
    m.poster_path AS poster_path, m.backdrop_path AS backdrop_path,
    [(m)-[:HAS_GENRE]->(g:Genre) | g.name]                AS genres,
    [(d:Person)-[:DIRECTED]->(m)  | d.name]               AS directors,
    [(a:Person)-[:ACTED_IN]->(m)  | a.name][..10]         AS cast_names,
    head([(m)-[:PART_OF]->(c:Collection) | c.id])         AS collection_id,
    head([(m)-[:PART_OF]->(c:Collection) | c.name])       AS collection_name
"""


def movies_by_titles(titles: list[str], limit_per_title: int = 3) -> list[dict]:
    """Exact-fact shortcut: resolve named titles to Movie nodes.

    Tries an exact case-insensitive match first, then falls back to a prefix
    match — users type "Alien" meaning the franchise, and "Lord of the Rings"
    for "The Lord of the Rings: The Fellowship of the Ring".
    """
    if not titles:
        return []
    return _query(
        f"""
        UNWIND $titles AS wanted
        MATCH (m:Movie)
        WHERE toLower(m.title) = toLower(wanted)
           OR toLower(m.title) STARTS WITH toLower(wanted)
        WITH m, wanted,
             CASE WHEN toLower(m.title) = toLower(wanted) THEN 0 ELSE 1 END AS exactness
        ORDER BY exactness, m.popularity DESC
        WITH wanted, collect(m)[..$limit_per_title] AS matches
        UNWIND matches AS m
        RETURN {RETRIEVAL_FIELDS}, wanted AS matched_title
        """,
        titles=titles,
        limit_per_title=limit_per_title,
    )


def search_movies(
    people: list[str] | None = None,
    genres: list[str] | None = None,
    year_range: list[int] | None = None,
    min_rating: float | None = None,
    limit: int = 20,
) -> list[dict]:
    """Constraint search: every supplied constraint is a HARD requirement (§3.4).

    Note `ALL(... WHERE EXISTS { ... })`: multiple people are ANDed, so "with
    Denzel Washington and directed by Antoine Fuqua" needs both on the same film.
    A person matches as either cast or director — the user rarely distinguishes,
    and asking for "movies with Clint Eastwood" should not miss the ones he only
    directed.

    Ordering is popularity-descending. That is a deliberate placeholder: the graph
    has no notion of relevance to a *vibe*, so it offers its most prominent
    matches and lets Day 5's fusion decide what actually ranks.
    """
    # Fold the requested names the same way ingestion folded the stored ones, so
    # "Bong Joon-ho" finds the node stored as "Bong Joon Ho".
    people = [normalize_name(p) for p in (people or []) if p]
    genres = genres or []
    lo, hi = (year_range or [None, None])[:2] if year_range else (None, None)

    return _query(
        f"""
        MATCH (m:Movie)
        WHERE ($people = [] OR ALL(wanted IN $people WHERE EXISTS {{
                  MATCH (p:Person)-[:ACTED_IN|DIRECTED]->(m)
                  WHERE p.name_normalized = wanted
              }}))
          AND ($genres = [] OR ALL(wanted IN $genres WHERE EXISTS {{
                  MATCH (m)-[:HAS_GENRE]->(g:Genre) WHERE g.name = wanted
              }}))
          AND ($lo IS NULL OR m.year >= $lo)
          AND ($hi IS NULL OR m.year <= $hi)
          AND ($min_rating IS NULL OR m.rating >= $min_rating)
        RETURN {RETRIEVAL_FIELDS}
        ORDER BY m.popularity DESC
        LIMIT $limit
        """,
        people=people,
        genres=genres,
        lo=lo,
        hi=hi,
        min_rating=min_rating,
        limit=limit,
    )


def people_who_share_a_name(limit: int = 10) -> list[dict]:
    """Diagnostic for contract §3.3: names owned by more than one real person.

    If Person had been keyed on name, every row here would be a corrupted node.
    """
    return _query(
        """
        MATCH (p:Person)
        WITH p.name AS name, collect(p.person_id) AS ids
        WHERE size(ids) > 1
        RETURN name, ids, size(ids) AS person_count
        ORDER BY person_count DESC, name
        LIMIT $limit
        """,
        limit=limit,
    )


def enrich_by_ids(tmdb_ids: list[int]) -> dict[int, dict]:
    """Link signals for movies we already found — the Day-4 `graph_enrich` core.

    Returns signals only, never new titles (contract §2.2).
    """
    rows = _query(
        """
        UNWIND $ids AS id
        MATCH (m:Movie {tmdb_id: id})
        OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
        OPTIONAL MATCH (d:Person)-[:DIRECTED]->(m)
        OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
        OPTIONAL MATCH (m)-[:PART_OF]->(c:Collection)
        RETURN m.tmdb_id AS tmdb_id,
               collect(DISTINCT g.name) AS genres,
               collect(DISTINCT d.name) AS directors,
               collect(DISTINCT a.name)[..10] AS cast,
               c.id AS collection_id, c.name AS collection_name
        """,
        ids=tmdb_ids,
    )
    return {r["tmdb_id"]: r for r in rows}


def graph_stats() -> dict:
    return _query(
        """
        CALL (){ MATCH (m:Movie)  RETURN count(m) AS movies }
        CALL (){ MATCH (p:Person) RETURN count(p) AS people }
        CALL (){ MATCH (k:Keyword) RETURN count(k) AS keywords }
        CALL (){ MATCH (c:Collection) RETURN count(c) AS collections }
        RETURN movies, people, keywords, collections
        """
    )[0]
