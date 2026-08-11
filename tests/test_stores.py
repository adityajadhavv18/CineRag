"""Day 2 invariants: the embed template, and each store's contract.

Store tests skip cleanly when the stores aren't running, so the suite is still
useful on a laptop with Docker off.
"""

from __future__ import annotations

import json

import pytest

from core.config import settings
from ingest.build_qdrant import build_payload, compose_embed_text

# ── fixtures ──────────────────────────────────────────────────────────────────

MOVIES = []
if settings.movies_jsonl.exists():
    with settings.movies_jsonl.open(encoding="utf-8") as fh:
        MOVIES = [json.loads(line) for line in fh if line.strip()][:300]

needs_movies = pytest.mark.skipif(not MOVIES, reason="movies.jsonl not built")


def _stores_up() -> bool:
    try:
        from stores import neo4j_client, qdrant_client

        qdrant_client.collection_stats()
        neo4j_client.graph_stats()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_stores = pytest.mark.skipif(not _stores_up(), reason="stores not reachable")


# ── the frozen embed template (contract §3.2) ────────────────────────────────


@needs_movies
def test_embed_template_shape_is_frozen():
    """Pin the template's SHAPE, so a reorder or relabel fails loudly.

    Changing this template invalidates every stored vector and requires a full
    re-embed — it is a versioned decision (§3.2). This test is the tripwire.
    """
    movie = next(m for m in MOVIES if m.get("tagline") and m.get("genres") and m.get("keywords"))
    lines = compose_embed_text(movie).split("\n")

    assert len(lines) == 5, "template must be exactly 5 lines"
    assert lines[0] == f"{movie['title']} ({movie['year']})"
    assert lines[1] == movie["tagline"]
    assert lines[2] == movie["overview"]
    assert lines[3] == "Genres: " + ", ".join(movie["genres"])
    assert lines[4] == "Keywords: " + ", ".join(movie["keywords"])


@needs_movies
def test_embed_template_never_emits_the_word_none():
    """Missing fields collapse to empty, not the literal string 'None'.

    Embedding "None" would add a token thousands of sparse movies share, quietly
    pulling unrelated films toward each other.
    """
    for movie in MOVIES:
        text = compose_embed_text(movie)
        assert "None" not in text, f"{movie['title']}: template leaked a None"


@needs_movies
def test_payload_is_not_the_embed_text():
    """Payload and vector are different things (§3.2): payload carries fields the
    embedding deliberately excludes, because they are for filtering/returning."""
    movie = next(m for m in MOVIES if m.get("cast") and m.get("director"))
    payload = build_payload(movie)
    assert payload["cast_names"] and payload["director"]
    assert "cast_names" not in compose_embed_text(movie)
    for key in ("tmdb_id", "poster_path", "backdrop_path", "rating"):
        assert key in payload


# ── Neo4j (contract §3.3) ────────────────────────────────────────────────────


@needs_stores
def test_franchise_timeline_includes_the_seed_movie():
    """A timeline missing its own first film is not a timeline.

    Regression guard: the single-path form
        (m)-[:PART_OF]->(c)<-[:PART_OF]-(sibling)
    silently drops the seed, because Cypher forbids reusing the same edge.
    """
    from stores import neo4j_client

    seed = neo4j_client._query(
        """
        MATCH (m:Movie)-[:PART_OF]->(c:Collection)<-[:PART_OF]-(:Movie)
        RETURN m.tmdb_id AS tmdb_id LIMIT 1
        """
    )
    if not seed:
        pytest.skip("no multi-film collections in the graph")
    tmdb_id = seed[0]["tmdb_id"]

    timeline = neo4j_client.franchise_timeline(tmdb_id)
    assert any(r["is_seed"] for r in timeline), "seed movie missing from its own timeline"
    years = [r["year"] for r in timeline if r["year"] is not None]
    assert years == sorted(years), "timeline must be in release order"


@needs_stores
def test_a_person_id_maps_to_exactly_one_node():
    """The uniqueness constraint from §3.3, verified in the live graph."""
    from stores import neo4j_client

    dupes = neo4j_client._query(
        """
        MATCH (p:Person)
        WITH p.person_id AS pid, count(*) AS n
        WHERE n > 1
        RETURN pid, n LIMIT 5
        """
    )
    assert dupes == [], f"person_id duplicated across nodes: {dupes}"


@needs_stores
def test_shared_names_remain_distinct_people():
    """The payoff of ID-keying: names owned by several people stay several nodes."""
    from stores import neo4j_client

    shared = neo4j_client.people_who_share_a_name(limit=5)
    for row in shared:
        assert len(set(row["ids"])) == row["person_count"] > 1


# ── Qdrant (contract §3.2) ───────────────────────────────────────────────────


@needs_stores
def test_collection_is_fully_populated():
    from stores import qdrant_client

    stats = qdrant_client.collection_stats()
    assert stats["points"] > 0
    if MOVIES:
        with settings.movies_jsonl.open(encoding="utf-8") as fh:
            total = sum(1 for line in fh if line.strip())
        assert stats["points"] == total, "every movie must be indexed"


@needs_stores
def test_payload_filter_is_a_hard_constraint():
    """filters.* must EXCLUDE, not merely down-rank (contract §3.4)."""
    from stores import qdrant_client

    f = qdrant_client.build_filter(genres=["Horror"], min_rating=7.0)
    rows = qdrant_client.search("a quiet family drama", limit=10, query_filter=f)
    assert rows, "filtered search returned nothing at all"
    for r in rows:
        assert "Horror" in r["genres"], f"{r['title']} slipped past the genre filter"
        assert r["rating"] >= 7.0, f"{r['title']} slipped past the rating filter"
