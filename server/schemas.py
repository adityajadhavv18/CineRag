"""HTTP request/response models (contract §4, §9).

This is a real trust boundary — the first one in the build. Everything upstream
was our own code calling our own code; here an unknown caller sends whatever it
likes. So this is where Pydantic earns its keep (validation at the edges, plain
dicts in the middle — the same reasoning that kept AgentState a TypedDict).

The response shape is designed for the Netflix-clone frontend in §9: every cited
film carries poster/backdrop PATHS (never URLs, never bytes) plus the overview,
so a card can render title + image + blurb without a second call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# A single message is a sentence, not a document. Bounding it stops a caller
# turning our OpenAI bill into their denial-of-service budget.
MAX_MESSAGE_CHARS = 1000
MAX_HISTORY_TURNS = 20


class Turn(BaseModel):
    """One prior message. The server holds NO memory (contract §1: stateless v1),
    so the client sends the conversation back with every request."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_CHARS * 4)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("message")
    @classmethod
    def not_only_whitespace(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be empty or only whitespace")
        return cleaned


class Source(BaseModel):
    """A film the answer actually cited — never merely retrieved (contract §5).

    `poster_path` / `backdrop_path` are TMDB paths, not URLs. The frontend builds
    `https://image.tmdb.org/t/p/w500{poster_path}` and loads from TMDB's CDN; this
    backend never stores or serves image bytes (contract §5, §9).
    """

    tmdb_id: int
    title: str
    year: int | None = None
    overview: str | None = None
    poster_path: str | None = None
    backdrop_path: str | None = None


class FranchiseFilm(BaseModel):
    tmdb_id: int
    title: str
    year: int | None = None
    is_seed: bool = False
    poster_path: str | None = None


class FranchiseTimeline(BaseModel):
    collection_id: int | None = None
    collection_name: str | None = None
    films: list[FranchiseFilm] = Field(default_factory=list)


class ClarifyOption(BaseModel):
    """One narrowing choice. `films` is how many films sit behind it — counted
    out of the graph, never estimated, which is the whole point of contract
    §2.3: an option nothing can satisfy makes asking worse than not asking."""

    label: str
    films: int


class ClarifyQuestion(BaseModel):
    prompt: str
    # Where the counts came from ("These pair most often in my catalogue"), when
    # that is worth saying. Not every question has one.
    note: str | None = None
    options: list[ClarifyOption] = Field(default_factory=list)

    # How a chosen option reads once composed back into a query. `phrase` holds
    # a single `{value}` placeholder; `slot` says whether the filled phrase goes
    # before the subject ("Thriller crime films") or after it ("from the 1990s").
    # Server-side because the wording belongs with the counts, not in whichever
    # client happens to be rendering them.
    slot: Literal["before", "after"] = "after"
    phrase: str = "{value}"
    id: str


class Clarification(BaseModel):
    """`response` as data.

    The prose in `response` says the same thing and remains authoritative — it
    is what goes into history as the agent's own words. This is for a client
    that would rather offer the options than ask someone to type one back.
    """

    lead: str
    # The noun the answers are composed around: "crime films", or just "films".
    subject: str
    questions: list[ClarifyQuestion] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    intent: str | None = None
    lead_engine: str | None = None
    sources: list[Source] = Field(default_factory=list)
    franchise: list[FranchiseTimeline] = Field(default_factory=list)

    # Present only when the agent asked grounded narrowing questions. Null on
    # every other intent, and null on clarification's ungrounded fallback, where
    # the honest open question in `response` is all there is to show.
    clarification: Clarification | None = None

    # True when a store was unreachable, so the client can show a "some data was
    # unavailable" notice instead of treating a thin answer as a complete one.
    # An empty `sources` list means different things depending on this flag —
    # exactly the distinction retrieval_errors exists to preserve.
    degraded: bool = False

    # The node path taken. Useful in a dev UI and when debugging a bad answer;
    # harmless to expose since it contains no internals a caller could exploit.
    trace: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    stores: dict[str, bool]
    catalogue_size: int | None = None


# ── browse surface (contract §9) ──────────────────────────────────────────────
#
# The chat endpoint answers questions; these answer "what is on the shelf". A
# browse page has to render before anyone has asked anything, and a detail modal
# needs credits the agent never carries, so neither can be served by /chat.
# Same rule as above throughout: PATHS, never URLs, never bytes.


class MovieCard(BaseModel):
    """One poster tile. Deliberately small — a row renders 20 of these."""

    tmdb_id: int
    title: str
    year: int | None = None
    rating: float | None = None
    poster_path: str | None = None
    backdrop_path: str | None = None
    # The hover preview shows a blurb and genre tags before anything is clicked.
    # Both come back inside the same store query the row already ran, so carrying
    # them costs nothing here and saves a request per hovered card.
    overview: str | None = None
    genres: list[str] = Field(default_factory=list)
    # Only on a person's acting credits: what they played in that film.
    character: str | None = None


class BrowseRow(BaseModel):
    title: str
    genre: str | None = None
    films: list[MovieCard] = Field(default_factory=list)


class BrowseResponse(BaseModel):
    # A carousel, not a single banner. Every film here is guaranteed to have a
    # backdrop — a 2:3 poster stretched across a 16:9 hero letterboxes, so a film
    # without one cannot be a hero at all.
    heroes: list[MovieCard] = Field(default_factory=list)
    rows: list[BrowseRow] = Field(default_factory=list)
    # Same contract as ChatResponse.degraded: empty `rows` with degraded=False
    # means the catalogue really is empty; with degraded=True it means the graph
    # was unreachable. The UI must not show "no films" for the second case.
    degraded: bool = False


class Credit(BaseModel):
    """A person on a film. `person_id` is the address — see contract §11 #1."""

    person_id: int
    name: str
    character: str | None = None


class MovieDetail(BaseModel):
    tmdb_id: int
    title: str
    year: int | None = None
    overview: str | None = None
    tagline: str | None = None
    rating: float | None = None
    runtime: int | None = None
    release_date: str | None = None
    poster_path: str | None = None
    backdrop_path: str | None = None
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    cast: list[Credit] = Field(default_factory=list)
    directors: list[Credit] = Field(default_factory=list)
    collection_id: int | None = None
    collection_name: str | None = None


class SimilarResponse(BaseModel):
    """Two different kinds of "related", kept separate on purpose.

    `films` are vector neighbours — films that *feel* like this one. `franchise`
    is the graph's exact answer — the films that literally belong with it. The UI
    renders them as separate rows because conflating a vibe with a fact is the
    thing this whole architecture exists to avoid (contract §2.1).
    """

    films: list[MovieCard] = Field(default_factory=list)
    franchise: FranchiseTimeline | None = None
    degraded: bool = False


class PersonResponse(BaseModel):
    person_id: int
    name: str
    acted: list[MovieCard] = Field(default_factory=list)
    directed: list[MovieCard] = Field(default_factory=list)
