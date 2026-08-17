"""Day 6 invariants: grounded clarification, and the franchise timeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import catalog
from agent.nodes.clarification import MIN_FILMS_PER_OPTION, MAX_QUESTIONS, clarification_node
from agent.nodes.franchise import franchise_node


def _stores_up() -> bool:
    try:
        from stores import neo4j_client

        neo4j_client.graph_stats()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_stores = pytest.mark.skipif(not _stores_up(), reason="stores not reachable")


# ── clarification must offer only options the catalogue can satisfy ──────────


@needs_stores
def test_every_offered_option_exists_in_the_catalogue():
    """THE Day 6 rule (contract §2.3).

    An invented option — "80s martial arts", "Hong Kong action" — sounds
    plausible and may match zero films. The user picks it, retrieval returns
    nothing, and asking made the conversation worse than not asking.
    """
    probe = catalog.probe(["Action"])
    real_genres = set(catalog.genres())

    for option in probe["pairings"]:
        assert option["option"] in real_genres, f"{option['option']} is not a catalogue genre"
        assert option["films"] > 0


@needs_stores
def test_offered_options_are_backed_by_enough_films():
    """Thin options are dropped — "3 films" is not a useful direction."""
    out = clarification_node({"query": "recommend some action movies",
                              "filters": {"genres": ["Action"]}, "entities": {}})
    # Every count rendered into the response must clear the floor.
    import re

    for count in re.findall(r"\((\d+)\)", out["response"]):
        assert int(count) >= MIN_FILMS_PER_OPTION


@needs_stores
def test_clarification_asks_at_most_three_questions():
    out = clarification_node({"query": "recommend some movies", "filters": {}, "entities": {}})
    numbered = [line for line in out["response"].splitlines() if line[:2] in
                ("1.", "2.", "3.", "4.", "5.")]
    assert 0 < len(numbered) <= MAX_QUESTIONS


@needs_stores
def test_clarification_never_retrieves():
    """It is a question, not an answer — no films may be cited."""
    out = clarification_node({"query": "recommend some movies", "filters": {}, "entities": {}})
    assert out["citations"] == []


def test_clarification_degrades_when_the_probe_is_empty():
    """Neo4j down, or a genre with nothing behind it (contract §5).

    Falls back to an honest open question rather than inventing options.
    """
    empty = {"genres": [], "pairings": [], "decades": [], "people": [], "total": 0}
    with patch("agent.catalog.probe", return_value=empty):
        out = clarification_node({"query": "recommend something", "filters": {}, "entities": {}})
    assert out["response"]
    assert out["citations"] == []
    assert "ungrounded" in out["trace"][0]


# ── franchise timeline ───────────────────────────────────────────────────────


@needs_stores
def test_timeline_includes_the_seed_and_is_ordered():
    """Regression from Day 2: the single-path Cypher dropped the seed film."""
    from stores import neo4j_client

    seed = neo4j_client._query(
        """
        MATCH (m:Movie)-[:PART_OF]->(:Collection)<-[:PART_OF]-(:Movie)
        RETURN m.tmdb_id AS tmdb_id, m.title AS title LIMIT 1
        """
    )
    if not seed:
        pytest.skip("no multi-film collections")

    out = franchise_node({"citations": [{"tmdb_id": seed[0]["tmdb_id"], "title": seed[0]["title"]}],
                          "response": "base"})
    timeline = out["franchise"][0]["films"]

    assert any(f["is_seed"] for f in timeline), "seed missing from its own timeline"
    years = [f["year"] for f in timeline if f["year"] is not None]
    assert years == sorted(years), "timeline not in release order"


@needs_stores
def test_franchise_appends_rather_than_replacing_the_answer():
    """The answer is already written and citation-validated; regenerating it
    would risk losing that grounding."""
    from stores import neo4j_client

    seed = neo4j_client._query(
        """
        MATCH (m:Movie)-[:PART_OF]->(:Collection)<-[:PART_OF]-(:Movie)
        RETURN m.tmdb_id AS tmdb_id LIMIT 1
        """
    )
    if not seed:
        pytest.skip("no multi-film collections")

    out = franchise_node({"citations": [{"tmdb_id": seed[0]["tmdb_id"]}],
                          "response": "ORIGINAL ANSWER"})
    assert out["response"].startswith("ORIGINAL ANSWER")


def test_franchise_skips_when_nothing_was_cited():
    out = franchise_node({"citations": [], "response": "x"})
    assert "franchise" not in out
    assert out["trace"] == ["franchise(skipped)"]


@needs_stores
def test_standalone_film_produces_no_timeline():
    """Inception is in no collection — it must not get an empty 'series' block."""
    out = franchise_node({"citations": [{"tmdb_id": 27205, "title": "Inception"}], "response": "x"})
    assert "franchise" not in out


def test_franchise_failure_does_not_break_the_answer():
    """A bonus feature must degrade, never take the response down (contract §5)."""
    with patch("stores.neo4j_client.franchise_timeline", side_effect=ConnectionError("down")):
        out = franchise_node({"citations": [{"tmdb_id": 671}], "response": "ANSWER"})
    assert out["trace"] == ["franchise(FAILED)"]
    assert "response" not in out  # original answer left untouched
