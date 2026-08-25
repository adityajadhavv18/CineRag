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


class ChatResponse(BaseModel):
    response: str
    intent: str | None = None
    lead_engine: str | None = None
    sources: list[Source] = Field(default_factory=list)
    franchise: list[FranchiseTimeline] = Field(default_factory=list)

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
