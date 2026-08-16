"""Day 3 invariants: state reducers, catalogue grounding, and routing.

The reducer and catalogue tests are pure and free. The routing tests cost a few
cents in API calls and are marked `llm` so they can be skipped:

    uv run pytest -q -m "not llm"
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from agent import catalog
from agent.state import AgentState, initial_state

# ── reducers: the Day 3 concept, proven rather than asserted ─────────────────


class _ParallelState(TypedDict, total=False):
    with_reducer: Annotated[list[str], operator.add]
    without_reducer: list[str]


def _build_parallel_graph(key: str) -> StateGraph:
    """Two nodes that run concurrently and both write `key`.

    Both edges leave START, so a and b execute in the SAME superstep — exactly
    the situation lead_engine="both" creates on Day 4.
    """

    def node_a(_state):
        return {key: ["from_a"]}

    def node_b(_state):
        return {key: ["from_b"]}

    builder = StateGraph(_ParallelState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    return builder


def test_reducer_keeps_both_concurrent_writes():
    """`Annotated[list, operator.add]` combines parallel writes.

    This is why vector_results and graph_results are annotated in agent/state.py.
    """
    result = _build_parallel_graph("with_reducer").compile().invoke({})
    assert sorted(result["with_reducer"]) == ["from_a", "from_b"]


def test_without_a_reducer_concurrent_writes_are_rejected():
    """What the annotation actually buys.

    LangGraph does NOT silently pick a winner — an un-reduced key written twice
    in one superstep raises InvalidUpdateError. So the reducer is not guarding
    against quiet data loss; it is what makes parallel fan-out possible at all.
    Without it, Day 4's `lead_engine=both` would hard-fail on every request.
    """
    from langgraph.errors import InvalidUpdateError

    with pytest.raises(InvalidUpdateError, match="without_reducer"):
        _build_parallel_graph("without_reducer").compile().invoke({})


def test_accumulating_fields_are_reduced_in_the_real_state():
    """Guard the actual AgentState, not just the toy one above.

    `trace` and `retrieval_errors` are the ones that genuinely conflict when
    lead_engine="both" — every node writes trace, so two retrievers in one
    superstep collide on it. vector_results/graph_results have one writer each
    today and are annotated for correctness and future writers.
    """
    hints = AgentState.__annotations__
    for field in ("vector_results", "graph_results", "trace", "retrieval_errors"):
        assert "Annotated" in str(hints[field]), f"{field} lost its reducer"


def test_both_retrievers_collide_on_trace_specifically():
    """The concrete conflict lead_engine="both" creates.

    Two nodes writing DIFFERENT result keys never conflict; they conflict on the
    key they share. Pinning this so the reducer rationale stays honest.
    """
    from langgraph.errors import InvalidUpdateError

    class _S(TypedDict, total=False):
        vector_results: list
        graph_results: list
        trace: list  # deliberately un-reduced

    def v(_s):
        return {"vector_results": [1], "trace": ["v"]}

    def g(_s):
        return {"graph_results": [2], "trace": ["g"]}

    b = StateGraph(_S)
    b.add_node("v", v)
    b.add_node("g", g)
    b.add_edge(START, "v")
    b.add_edge(START, "g")
    b.add_edge("v", END)
    b.add_edge("g", END)

    with pytest.raises(InvalidUpdateError, match="trace"):
        b.compile().invoke({})


def test_initial_state_seeds_the_reduced_lists():
    """Reduced fields start as lists.

    Belt-and-braces rather than load-bearing: LangGraph initialises an annotated
    channel on its own, so invoking without these keys also works. Seeding them
    keeps `initial_state()` an honest description of the shape a run starts in.
    """
    state = initial_state("hello")
    assert state["vector_results"] == []
    assert state["graph_results"] == []
    assert state["trace"] == []


# ── catalogue grounding (the silent-zero-results bug) ────────────────────────


def test_genres_normalize_to_catalogue_casing():
    """Stores match payload values EXACTLY: "crime" finds nothing, "Crime" works."""
    assert catalog.normalize_genres(["crime", "DRAMA", "Horror"]) == ["Crime", "Drama", "Horror"]


def test_colloquial_genres_are_mapped():
    assert catalog.normalize_genres(["sci-fi"]) == ["Science Fiction"]
    assert catalog.normalize_genres(["rom-com"]) == ["Romance"]


def test_unknown_genres_are_dropped_not_passed_through():
    """An unmatchable filter returns zero results — worse than ignoring the term."""
    assert catalog.normalize_genres(["Cyberpunk", "Crime"]) == ["Crime"]
    assert catalog.normalize_genres([]) == []
    assert catalog.normalize_genres(None) == []


def test_every_catalogue_genre_survives_normalization():
    assert catalog.normalize_genres(catalog.genres()) == catalog.genres()


# ── routing (costs API calls) ────────────────────────────────────────────────


@pytest.mark.llm
@pytest.mark.parametrize(
    "query,intent",
    [
        ("hi", "general"),
        ("what's the weather in Paris", "off_topic"),
        ("who directed Whiplash", "factual_lookup"),
        ("something tense and claustrophobic", "recommend"),
    ],
)
def test_intent_routing(query, intent):
    from agent.nodes.intent import classify

    assert classify(query).intent == intent


@pytest.mark.llm
@pytest.mark.parametrize(
    "query,lead",
    [
        ("something tense and claustrophobic", "vector"),
        ("films directed by Bong Joon-ho", "graph"),
        ("gritty crime dramas starring Denzel Washington", "both"),
    ],
)
def test_lead_engine_selection(query, lead):
    from agent.nodes.intent import classify

    assert classify(query).lead_engine == lead


@pytest.mark.llm
def test_lead_engine_is_invariant_to_phrasing():
    """Synonymous phrasings must route identically.

    Regression: `both`'s prompt examples all used the word "starring", so the
    model pattern-matched on the word — "gritty crime dramas STARRING Denzel"
    routed `both` while "...WITH Denzel" routed `vector`, dropping the graph from
    a query that names a person. Fixed by deriving lead_engine mechanically from
    the extracted fields instead of from the wording.
    """
    from agent.nodes.intent import classify

    phrasings = [
        "gritty crime dramas with Denzel Washington",
        "gritty crime dramas starring Denzel Washington",
        "gritty crime dramas featuring Denzel Washington",
    ]
    leads = {classify(q).lead_engine for q in phrasings}
    assert leads == {"both"}, f"phrasing changed the route: {leads}"


@pytest.mark.llm
def test_hard_constraint_vs_soft_signal():
    """Contract §3.4 — the distinction that breaks queries in opposite directions."""
    from agent.nodes.intent import classify

    hard = classify("movies with Denzel Washington")
    assert hard.filters.people, "'with X' must be a hard constraint"
    assert not hard.entities.people

    soft = classify("something like a Tarantino film")
    assert soft.entities.people, "'like X' must be a soft signal"
    assert not soft.filters.people, "'like X' must not exclude"


@pytest.mark.llm
def test_extracted_genres_are_always_catalogue_values():
    """The end-to-end version of the silent-zero-results bug."""
    from agent.nodes.intent import classify

    result = classify("horror movies rated above 8 from the 90s")
    for genre in result.filters.genres:
        assert genre in catalog.genres(), f"{genre!r} is not a catalogue value"


def test_off_topic_costs_nothing():
    """off_topic_node must not call an LLM — it is a fixed refusal by design."""
    from agent.nodes.off_topic import off_topic_node

    out = off_topic_node({"query": "what's the weather"})
    assert out["response"]
    assert out["citations"] == []
