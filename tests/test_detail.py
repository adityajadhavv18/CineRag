"""Film-detail answers: the mode switch, the plot budget, and plot grounding.

Background: "tell me about Inception" used to return 48 characters ("directed by
Christopher Nolan"), because the overview text lived in movies.jsonl and in
NEITHER store — we embedded the plot into the vector and discarded the words.
"""

from __future__ import annotations

import pytest

from agent.nodes.final_response import build_context, wants_detail


def _state(query, intent="factual_lookup", titles=("Inception",), refined=None):
    return {
        "query": query,
        "refined_query": refined or query,
        "intent": intent,
        "entities": {"titles": list(titles), "people": [], "genres": []},
    }


# ── the mode switch ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    ["tell me about Inception", "what is Inception about", "describe Inception",
     "tell me more about The Matrix"],
)
def test_open_requests_get_the_detail_profile(query):
    assert wants_detail(_state(query)) is True


@pytest.mark.parametrize(
    "query",
    ["who directed Inception", "what year was Inception released",
     "how long is Inception", "what rating does Inception have"],
)
def test_pointed_questions_stay_terse(query):
    """A specific attribute was asked for — answer it, don't write an essay."""
    assert wants_detail(_state(query)) is False


def test_detail_requires_a_named_title():
    """"tell me about something good" is a clarification, not a film profile."""
    assert wants_detail(_state("tell me about something good", titles=())) is False


def test_only_factual_lookup_can_be_detail():
    assert wants_detail(_state("tell me about Inception", intent="recommend")) is False


def test_mode_is_decided_from_the_users_own_words():
    """Regression: refined_query is model-written and once rewrote "tell me about
    Inception" into "who directed Inception", flipping this check and producing a
    one-line answer to an open question. The original phrasing is the ground truth
    for the SHAPE of the request."""
    state = _state("tell me about Inception", refined="who directed Inception")
    assert wants_detail(state) is True


# ── the plot token budget ────────────────────────────────────────────────────

ROW = {
    "tmdb_id": 1,
    "title": "A Film",
    "year": 2000,
    "overview": "A very long plot summary that costs tokens.",
    "tagline": "A tagline.",
    "genres": ["Drama"],
}


def test_plot_is_omitted_by_default():
    """15 recommendation candidates x ~70 tokens is ~1,000 tokens of text the
    answer will barely use."""
    context = build_context([ROW], {})
    assert "plot:" not in context
    assert "A Film" in context


def test_plot_is_included_for_detail_queries():
    context = build_context([ROW], {}, include_plot=True)
    assert "plot: A very long plot summary" in context
    assert "tagline: A tagline." in context


def test_plot_is_labelled_so_the_prompt_can_point_at_it():
    """"paraphrase the plot: text" is checkable guidance; "use the given
    information" is not."""
    context = build_context([ROW], {}, include_plot=True)
    assert "plot:" in context


def test_missing_plot_does_not_emit_an_empty_label():
    context = build_context([{**ROW, "overview": "", "tagline": ""}], {}, include_plot=True)
    assert "plot:" not in context
    assert "tagline:" not in context


# ── the data that makes grounding possible ───────────────────────────────────


def _stores_up() -> bool:
    try:
        from stores import neo4j_client, qdrant_client

        qdrant_client.collection_stats()
        neo4j_client.graph_stats()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_stores = pytest.mark.skipif(not _stores_up(), reason="stores not reachable")


@needs_stores
def test_both_stores_carry_the_plot_text():
    """Either store may answer alone, so both must be able to describe a film —
    the same reasoning that puts poster_path in both (contract §3.2, §3.3)."""
    from stores import neo4j_client, qdrant_client

    payload = qdrant_client.get_by_ids([27205])[0]
    assert payload.get("overview"), "Qdrant payload lost the overview"

    row = neo4j_client.movies_by_titles(["Inception"])[0]
    assert row.get("overview"), "Neo4j Movie node lost the overview"


@pytest.mark.llm
@needs_stores
def test_description_does_not_invent_plot_details():
    """The grounding hole this feature opened.

    validate_citations checks WHICH films are named, never WHAT is claimed about
    them. Inception's stored overview mentions Cobb, corporate espionage and the
    idea of inception — it says nothing about the spinning top, the totem, limbo,
    or Ariadne. A model writing from memory reaches for those almost immediately.
    """
    from agent.graph import run

    response = run("tell me about Inception")["response"].lower()
    for invented in ("spinning top", "totem", "limbo", "ariadne"):
        assert invented not in response, f"plot detail {invented!r} is not in the stored overview"


@pytest.mark.llm
@needs_stores
def test_a_detail_answer_always_carries_a_source():
    """Without a citation the API returns no `sources`, so the frontend has no
    poster to render (contract §9)."""
    from agent.graph import run

    citations = run("tell me about The Matrix")["citations"]
    assert citations, "detail answer produced no source"
    assert citations[0]["poster_path"]
