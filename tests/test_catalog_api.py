"""The browse surface (contract §9): /browse, /movie, /similar, /person.

Integration-style, like test_stores.py: these run against the real stores and
skip cleanly when Docker is off, because the interesting failures here are
Cypher and payload-shape failures that a mock would happily hide.

The FastAPI TestClient is instantiated WITHOUT its context manager on purpose —
that skips the lifespan, which compiles the LangGraph and warms an OpenAI client
none of these endpoints touch. Browse being independent of the agent is the
design; these tests hold it to that.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


def _stores_up() -> bool:
    try:
        from stores import neo4j_client, qdrant_client

        qdrant_client.collection_stats()
        neo4j_client.graph_stats()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_stores = pytest.mark.skipif(not _stores_up(), reason="stores not reachable")


@pytest.fixture(scope="module")
def a_film() -> dict:
    """A real film from the catalogue, so tests never hard-code a tmdb_id."""
    response = client.get("/api/v1/browse", params={"rows": 1, "seed": 7})
    rows = response.json()["rows"]
    if not rows or not rows[0]["films"]:
        pytest.skip("catalogue is empty")
    return rows[0]["films"][0]


# ── /browse ───────────────────────────────────────────────────────────────────


@needs_stores
def test_browse_returns_rows_of_films():
    body = client.get("/api/v1/browse", params={"rows": 4, "seed": 1}).json()

    assert len(body["rows"]) <= 4
    assert body["rows"], "a populated catalogue must yield at least one row"
    for row in body["rows"]:
        assert row["films"], f"row {row['title']!r} is empty and should not have been sent"
        assert row["genre"] in row["title"]
        # The card must be renderable on its own — that is the whole point of
        # carrying genres and overview in the row query (contract §9).
        first = row["films"][0]
        assert first["title"] and isinstance(first["tmdb_id"], int)


@needs_stores
def test_browse_is_reproducible_for_a_given_seed():
    """The shelf rotates by day, but a seed must pin it — otherwise a user
    scrolling and re-fetching would see the page reshuffle under them."""
    first = client.get("/api/v1/browse", params={"seed": 42}).json()
    second = client.get("/api/v1/browse", params={"seed": 42}).json()

    assert [r["genre"] for r in first["rows"]] == [r["genre"] for r in second["rows"]]


@needs_stores
def test_every_hero_has_a_backdrop():
    """A poster in a 16:9 hero slot letterboxes; the picker must skip films that
    only have one."""
    body = client.get("/api/v1/browse", params={"seed": 3}).json()

    assert body["heroes"], "a populated catalogue must yield at least one hero"
    assert all(h["backdrop_path"] for h in body["heroes"])


@needs_stores
def test_carousel_is_six_distinct_films_spread_across_rows():
    """One film per row before a second from any row — otherwise the page opens
    with six films of a single genre, which is a poor advert for a recommender.
    Distinct because the same title legitimately appears in two genre rows."""
    body = client.get("/api/v1/browse", params={"rows": 6, "seed": 3}).json()

    ids = [h["tmdb_id"] for h in body["heroes"]]
    assert len(ids) == len(set(ids)), "a film must not appear twice in the carousel"
    assert len(ids) <= 6

    # Each hero comes from a different row, drawn from that row's popular head.
    #
    # A film can sit in several genre rows at once, so "which row did this hero
    # come from" has no single answer from out here. What can be checked is that
    # some assignment of heroes to DISTINCT rows exists — which is exactly the
    # guarantee the picker makes, and is false the moment it drains one row.
    pools = [{f["tmdb_id"] for f in row["films"][:10]} for row in body["rows"]]

    def assignable(remaining: list[int], used: frozenset[int]) -> bool:
        if not remaining:
            return True
        hero, rest = remaining[0], remaining[1:]
        return any(
            assignable(rest, used | {i})
            for i, pool in enumerate(pools)
            if i not in used and hero in pool
        )

    assert assignable(ids, frozenset()), "heroes cannot be spread one per row"


@needs_stores
def test_heroes_are_the_best_rated_of_the_popular_head():
    """The banner is the most prominent thing on the page. Straight popularity
    ordering fills it with unreleased titles carrying a handful of votes, so
    rating decides the order within each row's popular head."""
    body = client.get("/api/v1/browse", params={"rows": 6, "seed": 3}).json()

    for hero in body["heroes"]:
        row = next(r for r in body["rows"] if hero["tmdb_id"] in {f["tmdb_id"] for f in r["films"]})
        pool = [f for f in row["films"][:10] if f["backdrop_path"]]
        best = max(f["rating"] or 0 for f in pool)
        # Not necessarily THE best — an earlier row may have taken it first — but
        # never a film the pool's top-rated beats by a wide margin.
        assert (hero["rating"] or 0) >= best - 1.5


def test_browse_rejects_an_absurd_row_count():
    assert client.get("/api/v1/browse", params={"rows": 500}).status_code == 422


# ── /movie/{id} ───────────────────────────────────────────────────────────────


@needs_stores
def test_movie_detail_carries_credits_the_agent_never_returns(a_film):
    """The reason this endpoint exists: cast, characters and directors are not
    in any chat response, and the modal cannot be built without them."""
    body = client.get(f"/api/v1/movie/{a_film['tmdb_id']}").json()

    assert body["tmdb_id"] == a_film["tmdb_id"]
    assert body["title"] == a_film["title"]
    assert body["genres"], "a film in a genre row must report that genre"
    assert body["cast"], "credits are the point of this endpoint"
    assert all(isinstance(c["person_id"], int) for c in body["cast"])


@needs_stores
def test_cast_is_ordered_by_billing(a_film):
    """Top-billed first. Without the `billing` property on ACTED_IN the graph
    returns cast in arbitrary order, which is why ingest records the index."""
    from stores import neo4j_client

    detail = neo4j_client.movie_detail(a_film["tmdb_id"])
    billings = [c.get("billing") for c in detail["cast"]]

    if any(b is None for b in billings):
        pytest.skip("graph predates the billing property — re-run ingest.build_neo4j")
    assert billings == sorted(billings)


@needs_stores
def test_unknown_film_is_404_not_500():
    """404 and 503 mean different things to the client: 'no such film' versus
    'ask again later'. Collapsing them would make the UI retry forever."""
    assert client.get("/api/v1/movie/999999999").status_code == 404


# ── /movie/{id}/similar ───────────────────────────────────────────────────────


@needs_stores
def test_similar_returns_neighbours_that_exclude_the_seed(a_film):
    body = client.get(f"/api/v1/movie/{a_film['tmdb_id']}/similar", params={"limit": 8}).json()

    assert body["films"], "every indexed film has neighbours"
    assert len(body["films"]) <= 8
    ids = [f["tmdb_id"] for f in body["films"]]
    assert a_film["tmdb_id"] not in ids, "a film is not 'more like' itself"


@needs_stores
def test_franchise_timeline_includes_its_seed():
    """Harry Potter is a known 7-film collection (contract Day 6). The timeline
    must contain the film it was seeded from — the bug §11 documents."""
    from stores import neo4j_client

    rows = neo4j_client._query(
        "MATCH (m:Movie)-[:PART_OF]->(c:Collection) RETURN m.tmdb_id AS id LIMIT 1"
    )
    if not rows:
        pytest.skip("no collections in the graph")

    body = client.get(f"/api/v1/movie/{rows[0]['id']}/similar").json()

    assert body["franchise"] is not None
    seeds = [f for f in body["franchise"]["films"] if f["is_seed"]]
    assert len(seeds) == 1
    assert seeds[0]["tmdb_id"] == rows[0]["id"]


# ── /person/{id} ──────────────────────────────────────────────────────────────


@needs_stores
def test_person_filmography_is_addressed_by_id(a_film):
    detail = client.get(f"/api/v1/movie/{a_film['tmdb_id']}").json()
    actor = detail["cast"][0]

    body = client.get(f"/api/v1/person/{actor['person_id']}").json()

    assert body["person_id"] == actor["person_id"]
    assert body["name"] == actor["name"]
    assert a_film["tmdb_id"] in [m["tmdb_id"] for m in body["acted"]]


@needs_stores
def test_shared_names_stay_separate_people():
    """Contract §11 #1: 48 people here share a name with someone else. Two ids
    with the same name must return different filmographies — a name-keyed
    lookup would merge them and quietly show one person the other's films."""
    from stores import neo4j_client

    clashes = neo4j_client.people_who_share_a_name(limit=1)
    if not clashes:
        pytest.skip("no name collisions in this catalogue")

    first, second = clashes[0]["ids"][:2]
    one = client.get(f"/api/v1/person/{first}").json()
    two = client.get(f"/api/v1/person/{second}").json()

    def credits(profile: dict) -> set[int]:
        # Across BOTH roles: a name clash is just as likely between two directors
        # as between two actors, and comparing only `acted` would call two empty
        # lists a match.
        return {m["tmdb_id"] for m in profile["acted"] + profile["directed"]}

    assert one["name"] == two["name"]
    assert credits(one) and credits(two), "a Person node with no credits is a bad fixture"
    assert credits(one) != credits(two)


@needs_stores
def test_unknown_person_is_404():
    assert client.get("/api/v1/person/999999999").status_code == 404
