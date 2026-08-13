"""general_node — greetings, small talk, "what can you do" (contract §2.3).

No retrieval. The capability description is hardcoded rather than generated
because it is a claim about THIS system: an LLM improvising here would happily
promise streaming availability and personalised watchlists, which we do not have
(contract §1 puts both explicitly out of scope).
"""

from __future__ import annotations

from core.config import settings
from core.llm import chat
from core.logger import get_logger
from agent.state import AgentState

log = get_logger("general_node")

CAPABILITIES = """\
I recommend films from a catalogue of about 5,000 real movies. I can:

  • find films by mood or theme — "something tense and claustrophobic"
  • answer factual questions — "who directed Whiplash", "what else is Cillian Murphy in"
  • combine the two — "gritty crime dramas starring Denzel Washington"
  • walk you through a franchise in order — every sequel, by release year

I only ever recommend films that are actually in my catalogue, and I will tell \
you when I find nothing rather than invent a title.\
"""

SYSTEM_PROMPT = f"""\
You are a friendly movie recommendation assistant. The user has sent a greeting, \
small talk, or a question about you — not a request for films.

Reply in one or two short sentences. Be warm and brief; do not pad.

If they ask what you can do, describe exactly these capabilities and nothing more:
{CAPABILITIES}

Never claim abilities beyond that list — in particular you have no access to \
streaming availability, showtimes, prices, or the user's personal history.
Never mention a specific film title: no retrieval has run, so any title you name \
would be pulled from memory rather than the catalogue.\
"""


def general_node(state: AgentState) -> dict:
    try:
        response = chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["query"]},
            ],
            temperature=0.4,
        )
    except Exception as exc:  # noqa: BLE001 — contract §5: degrade, don't crash
        log.error("general_node_failed", error=type(exc).__name__)
        response = "Hi! I recommend films from a catalogue of about 5,000 movies. What are you in the mood for?"

    log.info("general_response", chars=len(response), model=settings.chat_model)
    return {"response": response, "citations": [], "trace": ["general"]}
