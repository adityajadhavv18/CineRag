"""The shared state that flows through the graph (contract §4, Day 3).

Every node is a function `state -> partial state`. LangGraph merges whatever a
node returns into this dict and hands it to the next node. Nodes never call each
other; they only read and write here. That indirection is what makes routing
declarative instead of a nest of if-statements.

READ THE REDUCER NOTE BELOW — it is the one non-obvious thing in this file.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Intent = Literal[
    "recommend", "factual_lookup", "follow_up", "clarification", "general", "off_topic"
]
LeadEngine = Literal["vector", "graph", "both"]


class AgentState(TypedDict, total=False):
    """Total=False: nodes fill in their own slice, nobody must supply everything."""

    # ── input ────────────────────────────────────────────────────────────────
    query: str
    # Prior turns, passed in by the caller. The server holds no memory
    # (contract §1: stateless v1) — history arrives with the request.
    history: list[dict[str, str]]

    # ── intent_node output (contract §3.4) ───────────────────────────────────
    intent: Intent
    lead_engine: LeadEngine
    refined_query: str
    filters: dict[str, Any]
    entities: dict[str, Any]

    # ── retrieval (Day 4) ────────────────────────────────────────────────────
    #
    # THE REDUCER. When lead_engine == "both", vector_retrieve and graph_retrieve
    # run CONCURRENTLY and both write to state in the same superstep.
    #
    # LangGraph does not quietly pick a winner: an un-annotated key written twice
    # in one step raises InvalidUpdateError ("Can receive only one value per
    # step"). So the reducer is not protecting against silent data loss — it is
    # what makes the parallel fan-out legal in the first place. Without it,
    # `lead_engine=both` hard-fails on every request.
    #
    # `Annotated[list, operator.add]` tells LangGraph how to COMBINE two writes to
    # this key: with `+`. That is why these fields are annotated and the scalar
    # fields above, which only ever have one writer, are not.
    # Proven in tests/test_agent.py::test_without_a_reducer_concurrent_writes_are_rejected
    vector_results: Annotated[list[dict], operator.add]
    graph_results: Annotated[list[dict], operator.add]

    # Written by a single node (graph_enrich), so no reducer is needed.
    enrichment: dict[int, dict]

    # ── rerank + response (Day 5) ────────────────────────────────────────────
    reranked: list[dict]
    citations: list[dict]
    response: str
    franchise: list[dict]

    # ── observability (guideline C) ──────────────────────────────────────────
    # Appended to by every node, so one run carries its own decision trail.
    trace: Annotated[list[str], operator.add]


def initial_state(query: str, history: list[dict[str, str]] | None = None) -> AgentState:
    return {
        "query": query,
        "history": history or [],
        "vector_results": [],
        "graph_results": [],
        "trace": [],
    }
