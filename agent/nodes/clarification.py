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

TWO RENDERINGS OF ONE QUESTION SET. `response` is the prose, unchanged: it is
what the CLI prints, what the eval reads, and what goes back into history as the
agent's own words. `clarification` is the same questions before they were
flattened — options still separate, still carrying their counts — so a UI can
offer them as choices instead of asking someone to type "1990s" back at us.
Nothing new is computed for it; the prose was always built from this.

Each question carries the PHRASING for its own answer (`phrase` + `slot`), so
composing the picks back into a query stays here, next to the counts that make
the options real, rather than being reinvented by every client that renders them.
The ungrounded fallback deliberately emits no payload: there are no counted
options, so there must be no options UI.
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


def _options(options: list[dict], label_fn=str) -> list[dict]:
    """The same options `_fmt` renders, kept apart instead of joined.

    The count travels with each one: it is what proves the option was counted
    out of the graph rather than imagined, and it is genuinely useful to see
    before choosing ("Thriller 212" against "Music 18").
    """
    return [{"label": label_fn(o["option"]), "films": o["films"]} for o in options]


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
    asked: list[dict] = []

    def ask(prompt: str, note: str | None, options: list[dict], *, slot: str, phrase: str,
            id: str, label_fn=str) -> None:
        """Record one question in both renderings, from one set of options."""
        stem = f"{prompt} {note}:" if note else prompt
        questions.append(f"{stem}\n   {_fmt(options, label_fn)}")
        asked.append({
            "id": id,
            "prompt": prompt,
            "note": note,
            # Where this answer belongs in the sentence the picks compose into,
            # and how it reads there. "Thriller" goes in front of the subject as
            # an adjective; an era or a name follows it as a clause.
            "slot": slot,
            "phrase": phrase,
            "options": _options(options, label_fn),
        })

    if pairings:
        ask(
            f"What flavour of {subject}?" if seed_genres
            else "What sort of thing are you in the mood for?",
            "These pair most often in my catalogue" if seed_genres
            else "My biggest categories",
            pairings[:4],
            slot="before",
            phrase="{value}",
            id="flavour",
        )
    if decades:
        ask(
            "Any particular era?",
            None,
            decades[:4],
            slot="after",
            phrase="from the {value}",
            id="era",
            label_fn=lambda d: f"{d}s",
        )
    if people and seed_genres:
        ask(
            "Anyone you'd like to see in it?",
            "Most prolific here",
            people[:4],
            slot="after",
            phrase="starring {value}",
            id="cast",
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

    # `subject` is the noun the picks are composed around ("crime films"), so a
    # client can build "Thriller crime films from the 1990s" without knowing
    # what was asked or in which order.
    clarification = {
        "lead": lead,
        "subject": subject,
        "questions": asked[:MAX_QUESTIONS],
    }

    log.info(
        "clarification_asked",
        seed_genres=seed_genres,
        total_films=total,
        questions=len(questions[:MAX_QUESTIONS]),
        # Proof the options were grounded, visible in the trace.
        options_offered=[o["option"] for o in pairings[:4]],
    )
    return {
        "response": response,
        "citations": [],
        "clarification": clarification,
        "trace": [f"clarification({len(questions)}q)"],
    }
