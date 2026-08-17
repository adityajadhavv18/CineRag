"""clarification_node — ask a narrowing question the catalogue can answer.

On-topic but too vague to retrieve well: "recommend some action movies" matches
1,156 films, so any five we picked would be arbitrary.

THE RULE: every option offered is counted out of the graph first. An LLM asked to
suggest narrowing options for "action" will happily propose "80s martial arts" or
"Hong Kong action" — plausible-sounding categories that may match zero of our
films. The user picks one, retrieval returns nothing, and the clarification has
made the conversation worse than not asking. So we look at the shelf, then ask.

That guarantee is why the questions are TEMPLATED rather than written by an LLM.
A prompt saying "only use these options" is guidance; building the sentence from
the query result is a guarantee. The cost is slightly stiffer prose, and no
tokens or latency — the same trade-off off_topic_node makes.
"""

from __future__ import annotations

from core.logger import get_logger
from agent import catalog
from agent.state import AgentState

log = get_logger("clarification_node")

# Below this, an option is too thin to be worth offering — "3 films" is not a
# useful direction to send someone in.
MIN_FILMS_PER_OPTION = 15
MAX_QUESTIONS = 3


def _fmt(options: list[dict], label_fn=str) -> str:
    return "  ".join(f"• {label_fn(o['option'])} ({o['films']})" for o in options)


def clarification_node(state: AgentState) -> dict:
    filters = state.get("filters") or {}
    entities = state.get("entities") or {}
    seed_genres = filters.get("genres") or entities.get("genres") or []

    probe = catalog.probe(seed_genres)

    pairings = [o for o in probe["pairings"] if o["films"] >= MIN_FILMS_PER_OPTION]
    decades = [o for o in probe["decades"] if o["films"] >= MIN_FILMS_PER_OPTION]
    people = [o for o in probe["people"] if o["films"] >= MIN_FILMS_PER_OPTION]

    subject = f"{' / '.join(seed_genres).lower()} films" if seed_genres else "films"
    total = probe.get("total") or 0

    questions: list[str] = []
    if pairings:
        stem = (
            f"What flavour of {subject}? These pair most often in my catalogue:"
            if seed_genres
            else "What sort of thing are you in the mood for? My biggest categories:"
        )
        questions.append(f"{stem}\n   {_fmt(pairings[:4])}")
    if decades:
        questions.append(
            "Any particular era?\n   " + _fmt(decades[:4], lambda d: f"{d}s")
        )
    if people and seed_genres:
        questions.append(
            "Anyone you'd like to see in it? Most prolific here:\n   " + _fmt(people[:4])
        )

    if not questions:
        # Probe came back empty — Neo4j down, or a genre with almost nothing
        # behind it. Degrade to an ungrounded but honest ask rather than
        # inventing options (contract §5).
        log.warning("clarification_ungrounded", seed_genres=seed_genres,
                    reason="probe returned no usable options")
        response = (
            "Happy to help — could you give me a bit more to go on? A mood, an era, "
            "a favourite film to steer by, or an actor or director you like."
        )
        return {"response": response, "citations": [], "trace": ["clarification(ungrounded)"]}

    lead = (
        f"I've got {total:,} {subject} — let me narrow it down."
        if total
        else "Let me narrow that down."
    )
    body = "\n\n".join(f"{i}. {q}" for i, q in enumerate(questions[:MAX_QUESTIONS], start=1))
    response = f"{lead}\n\n{body}\n\nAnswer any of these and I'll take it from there."

    log.info(
        "clarification_asked",
        seed_genres=seed_genres,
        total_films=total,
        questions=len(questions[:MAX_QUESTIONS]),
        # Proof the options were grounded, visible in the trace.
        options_offered=[o["option"] for o in pairings[:4]],
    )
    return {"response": response, "citations": [], "trace": [f"clarification({len(questions)}q)"]}
