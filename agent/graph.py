"""The compiled LangGraph agent (contract §2.2, §4).

Day 4 shape — real retrieval, fanned out by lead_engine:

    START -> intent -> ┬-> general    -> END
                       ├-> off_topic  -> END
                       │
                       ├-> vector_retrieve ─┐
                       └-> graph_retrieve  ─┴-> graph_enrich -> final_response -> END

Which of the two retrievers start is decided by `lead_engine`; when it is "both"
they start together in one superstep and graph_enrich waits for both.

The router below is a plain function returning the NAME of the next node. That
is the whole mechanism behind a conditional edge: LangGraph calls it with the
current state, takes the string back, and jumps there. Nothing magic — but note
that the routing VALUE comes from state (`state["intent"]`), which means the
intent node's output, not any hardcoded control flow, decides the path.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from core.config import configure_langsmith, settings
from core.logger import get_logger
from agent.state import AgentState
from agent.nodes.final_response import final_response
from agent.nodes.general import general_node
from agent.nodes.graph_enrich import graph_enrich
from agent.nodes.graph_retrieve import graph_retrieve
from agent.nodes.intent import intent_node
from agent.nodes.off_topic import off_topic_node
from agent.nodes.vector_retrieve import vector_retrieve

log = get_logger("graph")


def route_by_intent(state: AgentState) -> list[str]:
    """Conditional edge: read state, return the next node's NAME.

    The three retrieval intents share one destination for now; on Day 4 they
    fan out to the retrievers according to lead_engine.
    """
    intent = state.get("intent")
    if intent is None:
        # Unreachable by design: intent_node always writes this key, because
        # classify() falls back rather than raising. If it ever fires, the graph
        # was invoked with a hand-built state or a node was skipped — surface it
        # loudly instead of substituting a plausible-looking default, which would
        # log a value no classifier ever produced.
        log.warning("intent_missing", query=str(state.get("query"))[:80])

    # Only the two no-retrieval intents get their own door. Everything else —
    # recommend, factual_lookup, follow_up, and (until Day 6) clarification —
    # goes through retrieval, so adding a retrieval intent needs no change here.
    if intent in ("general", "off_topic"):
        log.info("routed", intent=intent, destination=[intent])
        return [intent]

    # Past this point we are retrieving, and lead_engine picks WHICH retrievers.
    # Returning a LIST is how LangGraph fans out: every name in it starts in the
    # same superstep. That is the concurrency the reducers in state.py exist for.
    lead = state.get("lead_engine", "vector")
    destinations = {
        "vector": ["vector_retrieve"],
        "graph": ["graph_retrieve"],
        "both": ["vector_retrieve", "graph_retrieve"],
    }.get(lead, ["vector_retrieve"])

    log.info("routed", intent=intent, lead_engine=lead, destinations=destinations,
             parallel=len(destinations) > 1)
    return destinations


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("general", general_node)
    builder.add_node("off_topic", off_topic_node)
    builder.add_node("vector_retrieve", vector_retrieve)
    builder.add_node("graph_retrieve", graph_retrieve)
    builder.add_node("graph_enrich", graph_enrich)
    builder.add_node("final_response", final_response)

    builder.add_edge(START, "intent")
    builder.add_conditional_edges(
        "intent",
        route_by_intent,
        # The set of nodes this edge may jump to. Declaring it keeps the drawn
        # diagram accurate and makes a typo'd destination fail loudly.
        ["general", "off_topic", "vector_retrieve", "graph_retrieve"],
    )

    # Both retrievers converge on enrich. When both ran, LangGraph waits for BOTH
    # before starting graph_enrich — a node runs once its scheduled parents are
    # done, not once per parent. That join is why enrichment sees the merged set.
    builder.add_edge("vector_retrieve", "graph_enrich")
    builder.add_edge("graph_retrieve", "graph_enrich")
    builder.add_edge("graph_enrich", "final_response")

    builder.add_edge("general", END)
    builder.add_edge("off_topic", END)
    builder.add_edge("final_response", END)

    return builder


@lru_cache(maxsize=1)
def get_agent():
    """Compile once, reuse. Compilation validates the graph's wiring."""
    tracing = configure_langsmith()
    agent = build_graph().compile()
    log.info("graph_compiled", tracing=tracing,
             project=settings.langsmith_project if tracing else None)
    return agent


def run(query: str, history: list[dict[str, str]] | None = None) -> AgentState:
    from agent.state import initial_state

    return get_agent().invoke(initial_state(query, history))
