"""The compiled LangGraph agent (contract §2.2, §4).

Day 3 shape — the cheap paths are real, retrieval is a placeholder:

    START -> intent_node -> ┬-> general_node   -> END
                            ├-> off_topic_node -> END
                            └-> retrieval_placeholder -> END   (Day 4 replaces this)

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
from agent.nodes.general import general_node
from agent.nodes.intent import intent_node
from agent.nodes.off_topic import off_topic_node

log = get_logger("graph")


def retrieval_placeholder(state: AgentState) -> dict:
    """Stand-in for the Day 4 retrieval fan-out.

    Deliberately honest: it states what WOULD have run rather than pretending to
    answer, so a half-built graph can never look like a working recommender.
    """
    lead = state.get("lead_engine", "vector")
    engines = {
        "vector": "vector_retrieve + graph_enrich",
        "graph": "graph_retrieve",
        "both": "vector_retrieve + graph_retrieve (parallel), then graph_enrich",
    }[lead]

    log.info("retrieval_placeholder", intent=state.get("intent"), lead_engine=lead,
             would_run=engines)

    return {
        "response": (
            f"[Day 4 stub] intent={state.get('intent')} lead_engine={lead}\n"
            f"  would run: {engines}\n"
            f"  refined:   {state.get('refined_query')}\n"
            f"  filters:   {state.get('filters')}\n"
            f"  entities:  {state.get('entities')}"
        ),
        "citations": [],
        "trace": [f"retrieval_placeholder({lead})"],
    }


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: read state, return the next node's NAME.

    The three retrieval intents share one destination for now; on Day 4 they
    fan out to the retrievers according to lead_engine.
    """
    intent = state.get("intent", "recommend")
    destination = {
        "general": "general",
        "off_topic": "off_topic",
        # clarification gets its own node on Day 6; until then it retrieves.
    }.get(intent, "retrieval")

    log.info("routed", intent=intent, destination=destination)
    return destination


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("general", general_node)
    builder.add_node("off_topic", off_topic_node)
    builder.add_node("retrieval", retrieval_placeholder)

    builder.add_edge(START, "intent")
    builder.add_conditional_edges(
        "intent",
        route_by_intent,
        # The map is the set of nodes this edge may jump to. Declaring it keeps
        # the drawn diagram accurate and makes a typo'd destination fail loudly.
        {"general": "general", "off_topic": "off_topic", "retrieval": "retrieval"},
    )
    builder.add_edge("general", END)
    builder.add_edge("off_topic", END)
    builder.add_edge("retrieval", END)

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
