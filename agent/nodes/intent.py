"""intent_node — classify, route, and extract (contract §3.4).

The brain of routing. One LLM call produces everything downstream needs:

  intent         which branch of the graph runs at all
  lead_engine    which store(s) lead the retrieval  (contract §2.2)
  refined_query  a self-contained restatement, so retrieval never needs history
  filters        HARD constraints — exclude anything that fails them
  entities       SOFT signals — traverse and score, never exclude

We use OpenAI structured outputs, so the schema is guaranteed by the API rather
than parsed hopefully out of prose. A malformed classification here would
mis-route every request downstream.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.config import settings
from core.llm import get_client
from core.logger import get_logger
from agent import catalog
from agent.state import AgentState

log = get_logger("intent_node")


class Filters(BaseModel):
    """HARD constraints. A movie failing any of these is excluded outright."""

    genres: list[str] = Field(default_factory=list)
    year_range: list[int] | None = None  # [start, end], inclusive
    min_rating: float | None = None
    people: list[str] = Field(default_factory=list)


class Entities(BaseModel):
    """SOFT signals. Mentioned things — used to traverse and score, never to exclude."""

    people: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)


class IntentResult(BaseModel):
    intent: Literal[
        "recommend", "factual_lookup", "follow_up", "clarification", "general", "off_topic"
    ]
    lead_engine: Literal["vector", "graph", "both"]
    refined_query: str
    filters: Filters
    entities: Entities


SYSTEM_PROMPT = """\
You classify user messages for a movie recommendation agent and decide how to \
retrieve for them. Return ONLY the structured object.

## intent — what kind of message is this?

- `recommend`       wants movie suggestions. "something tense", "good sci-fi like Arrival"
- `factual_lookup`  asks a specific answerable fact. "who directed Whiplash", "what year was Alien"
- `follow_up`       refines or references the immediately previous turn. "only the 90s ones",
                    "what about her other films", "more like the second one"
- `clarification`   on-topic but too vague to retrieve well. "recommend some movies",
                    "I want something good"
- `general`         greetings, small talk, or questions about the assistant itself.
                    "hi", "what can you do", "who are you", "thanks"
- `off_topic`       not about movies at all. "what's the weather", "write me a python script"

## lead_engine — which store should lead?

- `vector`  MOOD / VIBE / THEME queries with no named person or title. The answer depends on
            what films are ABOUT or how they FEEL.
            "something tense and claustrophobic", "a cozy heist with a twist"
- `graph`   FACT queries about named people, titles, or franchises, where the answer is a
            relationship that is either true or false.
            "movies directed by Bong Joon-ho", "what else is Cillian Murphy in",
            "the Alien films in order"
- `both`    BLENDED — a vibe requirement AND a named person/title/franchise constraint.
            "gritty crime dramas starring Denzel Washington",
            "mind-bending sci-fi like Inception with a female lead"

DECIDE THIS MECHANICALLY, from what you extracted — not from the wording. Work out the two
questions below and read the answer off the table. Synonymous phrasings ("with X", "starring X",
"featuring X", "that has X in it") MUST all produce the same lead_engine.

  A. Did you put any person or title in filters.people / entities.people / entities.titles?
  B. Does the request also describe a mood, theme, tone, subject or style
     ("gritty", "cozy", "mind-bending", "about grief", "feel-good")?

     A = no,  B = yes  ->  vector
     A = yes, B = no   ->  graph
     A = yes, B = yes  ->  both
     A = no,  B = no   ->  vector

A pure attribute filter with no names and no vibe ("horror rated above 8 from the 90s") is
`vector`: payload filters do the work and there is nothing for the graph to traverse.

For `general` and `off_topic`, no retrieval happens; use `vector` as an inert default.

## filters vs entities — the distinction that matters most

Put a person in `filters.people` when they MUST be in the movie — the user is constraining:
    "movies WITH Denzel Washington"      "DIRECTED BY Kurosawa"      "STARRING Toni Collette"

Put a person in `entities.people` when they are a REFERENCE POINT — the user is comparing:
    "something LIKE a Tarantino film"    "IN THE SPIRIT OF Kubrick"   "gives me Wes Anderson VIBES"

Getting this backwards breaks the query in opposite directions: "like a Tarantino film" as a
hard filter returns only the handful he directed, when the user wanted the style; "with Denzel
Washington" as a soft signal returns crime dramas he is not in.

EVERY named person in the message MUST appear in exactly one of `filters.people` or
`entities.people` — never both, never neither. A dropped name means the retriever has nothing
to look up. Use the person's full name as written.
    "films directed by Bong Joon-ho"     -> filters.people  = ["Bong Joon-ho"]
    "something like a Tarantino film"    -> entities.people = ["Quentin Tarantino"]
    "give me Wes Anderson vibes"         -> entities.people = ["Wes Anderson"]

EVERY movie title named in the message MUST appear in `entities.titles`, exactly as written.
This includes the subject of a factual question and any film used as a comparison — without it
the retriever has no movie to look up.
    "who directed Whiplash"              -> entities.titles = ["Whiplash"]
    "something like Inception"           -> entities.titles = ["Inception"]
    "the Alien films in order"           -> entities.titles = ["Alien"]

The same rule applies to genres: an explicit demand ("only horror", "rated above 8") is a
filter; a mentioned comparison ("like Hereditary") is an entity.

## genres — use the catalogue's exact vocabulary

Genre values MUST come from this list, spelled and capitalised exactly:
{genre_list}

Map colloquial terms onto it ("sci-fi" -> "Science Fiction", "rom-com" -> "Romance"). If a
requested genre has no equivalent in the list, leave it out of `filters.genres` and let the
`refined_query` carry the meaning instead — an unmatchable filter returns nothing at all.

## refined_query

Rewrite the user's message as a single self-contained sentence describing what they want,
resolving any pronouns or references using the conversation history. Downstream retrieval sees
ONLY this string, never the history. For `general`/`off_topic`, echo the message back.
"""


def _fallback(query: str) -> IntentResult:
    """Degrade gracefully (contract §5) rather than 500 on an LLM failure.

    `recommend` + `vector` is the safest guess: it retrieves something plausible
    instead of dead-ending, and a vibe search over a movie catalogue is a
    reasonable response to almost any movie-shaped message.
    """
    return IntentResult(
        intent="recommend",
        lead_engine="vector",
        refined_query=query,
        filters=Filters(),
        entities=Entities(),
    )


def build_system_prompt() -> str:
    """Inject the catalogue's real genre vocabulary into the prompt.

    Grounding the model in actual values is cheaper and more reliable than
    correcting it afterwards — though normalize_genres() still runs as a
    backstop, because a prompt is guidance and a lookup is a guarantee.
    """
    return SYSTEM_PROMPT.format(genre_list=", ".join(catalog.genres()))


def classify(query: str, history: list[dict[str, str]] | None = None) -> IntentResult:
    messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]
    for turn in (history or [])[-6:]:  # a few turns is plenty to resolve a pronoun
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": query})

    try:
        completion = get_client().chat.completions.parse(
            model=settings.chat_model,
            messages=messages,
            response_format=IntentResult,
            temperature=0,  # classification should be deterministic
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("structured output returned no parsed object")

        # Backstop: the stores match payload values exactly, so a genre the model
        # invented or mis-cased would produce a filter nothing satisfies — zero
        # results that look like "no such movies" rather than a bad filter.
        parsed.filters.genres = catalog.normalize_genres(parsed.filters.genres)
        parsed.entities.genres = catalog.normalize_genres(parsed.entities.genres)
        return parsed
    except Exception as exc:  # noqa: BLE001 — routing must never hard-fail
        log.error("intent_classification_failed", error=type(exc).__name__, detail=str(exc)[:200])
        return _fallback(query)


def intent_node(state: AgentState) -> dict:
    result = classify(state["query"], state.get("history"))

    log.info(
        "intent_classified",
        query=state["query"][:80],
        intent=result.intent,
        lead_engine=result.lead_engine,
        filter_people=result.filters.people,
        entity_people=result.entities.people,
        entity_titles=result.entities.titles,
        filter_genres=result.filters.genres,
        year_range=result.filters.year_range,
        min_rating=result.filters.min_rating,
    )

    return {
        "intent": result.intent,
        "lead_engine": result.lead_engine,
        "refined_query": result.refined_query,
        "filters": result.filters.model_dump(),
        "entities": result.entities.model_dump(),
        "trace": [f"intent={result.intent} lead={result.lead_engine}"],
    }
