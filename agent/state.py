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
    # THE REDUCER. An un-annotated key written twice in ONE superstep raises
    # InvalidUpdateError ("Can receive only one value per step") — LangGraph does
    # not quietly pick a winner. So a reducer is what makes a parallel fan-out
    # legal, not a guard against silent data loss.
    #
    # The rule is per KEY, not per node. What matters is whether two nodes can
    # write THE SAME key in the same step:
    #
    #   vector_results   one writer (vector_retrieve)  -> no conflict possible
    #   graph_results    one writer (graph_retrieve)   -> no conflict possible
    #   trace            EVERY node writes it          -> conflict when both
    #                                                     retrievers run
    #
    # So when lead_engine == "both", the key that actually needs its reducer is
    # `trace` (and `retrieval_errors`, if both stores fail at once). These two are
    # annotated anyway: it is the correct type for an accumulating list, and it
    # keeps them safe if a later day adds a second writer — Day 6's clarification
    # re-retrieval is the likely candidate.
    # Proven in tests/test_agent.py::test_without_a_reducer_concurrent_writes_are_rejected
    vector_results: Annotated[list[dict], operator.add]
    graph_results: Annotated[list[dict], operator.add]

    # Which stores FAILED, as opposed to returning nothing. Both cases leave the
    # lists above empty, but they mean opposite things: "no film matches" is a
    # legitimate answer we should state honestly (§5), while "Neo4j is down" is an
    # outage that must not be dressed up as an empty catalogue. Also reduced,
    # because two retrievers can fail in the same superstep.
    retrieval_errors: Annotated[list[dict], operator.add]

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
        "retrieval_errors": [],
        "trace": [],
    }
