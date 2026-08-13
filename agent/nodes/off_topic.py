"""off_topic_node — politely decline non-movie queries (contract §2.3).

No retrieval, and no LLM call either. A fixed response is the right call here:
the whole point is to refuse, so there is nothing to generate, and spending a
round-trip plus tokens to rephrase "I only do movies" six different ways buys
nothing. It also makes the refusal impossible to talk the model out of.
"""

from __future__ import annotations

from core.logger import get_logger
from agent.state import AgentState

log = get_logger("off_topic_node")

RESPONSE = (
    "I only handle movies, I'm afraid — recommendations, facts about films and the "
    "people who made them, and franchise timelines. Ask me for something to watch "
    "and I'm all yours."
)


def off_topic_node(state: AgentState) -> dict:
    log.info("off_topic_declined", query=state["query"][:80])
    return {"response": RESPONSE, "citations": [], "trace": ["off_topic"]}
