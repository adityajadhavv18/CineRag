"""OpenAI access: chat + embeddings (contract §1, §6).

The only paid dependency. Everything that talks to OpenAI goes through here, so
model names live in config and nothing hardcodes them.

LangSmith tracing is wired on Day 3 — see the hook at the bottom.
"""

from __future__ import annotations

import time
import warnings
from functools import lru_cache

from openai import OpenAI, APIError, RateLimitError

from core.config import settings
from core.logger import get_logger

log = get_logger("llm")

# text-embedding-3-small produces 1536-dimensional vectors. Qdrant needs this at
# collection-creation time and it must match forever after, so it lives here as
# the single definition rather than being hardcoded in the Qdrant builder.
EMBEDDING_DIM = 1536

# OpenAI accepts up to 2048 inputs per embedding request. 128 keeps each request
# comfortably small (~30k tokens) so one transient failure retries cheaply.
EMBED_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """The OpenAI client, wrapped for LangSmith tracing when tracing is enabled.

    We use the plain OpenAI SDK rather than LangChain's chat models — fewer
    abstractions between you and the request. `wrap_openai` is what buys back the
    tracing that LangChain would have given for free: it records each call's
    prompt, response, token counts and latency as a child span of whichever node
    is running. Without it, LangSmith would show the node but not what it asked.
    """
    settings.require("openai_api_key")
    client = OpenAI(api_key=settings.openai_api_key)

    if settings.langchain_tracing_v2 and settings.langsmith_api_key:
        from langsmith.wrappers import wrap_openai

        # Tracing serialises the whole completion to send to LangSmith, including
        # the `parsed` field that structured outputs adds. Pydantic warns that the
        # field's declared type is `None` — cosmetic, and it fires on every single
        # traced call, which drowns our own logs. Silence just this one.
        warnings.filterwarnings(
            "ignore", message=".*serializer warnings.*", category=UserWarning,
            module="pydantic.main",
        )
        client = wrap_openai(client)
        log.info("openai_client_traced", project=settings.langsmith_project)

    return client


def embed_texts(texts: list[str], *, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts, in order. Returns one vector per input.

    Order is guaranteed: OpenAI returns an `index` per embedding, and we sort by
    it rather than trusting response order — a silent misalignment here would
    attach every movie's vector to the wrong movie, which is close to unfindable
    later.
    """
    client = get_client()
    out: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        delay = 2.0
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=settings.embedding_model, input=batch)
                out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
                break
            except RateLimitError:
                log.warning("embed_rate_limited", offset=start, retry_in=delay, attempt=attempt)
                time.sleep(delay)
                delay *= 2
            except APIError as exc:
                log.warning("embed_api_error", offset=start, error=type(exc).__name__,
                            attempt=attempt)
                time.sleep(delay)
                delay *= 2
        else:
            raise RuntimeError(f"embedding failed after retries at offset {start}")

    if len(out) != len(texts):
        raise RuntimeError(f"embedding count mismatch: got {len(out)}, expected {len(texts)}")
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single search query.

    Note what is NOT happening here: the frozen §3.2 template is an *indexing*
    concern only. A user's query is embedded as-is — wrapping it in the template
    would push it away from the movies it should match.
    """
    return embed_texts([text])[0]


def chat(messages: list[dict], **kwargs) -> str:
    """Single-turn chat completion returning the message text. Used from Day 3."""
    client = get_client()
    resp = client.chat.completions.create(
        model=settings.chat_model, messages=messages, **kwargs
    )
    return resp.choices[0].message.content or ""


# Day 3 hook: LangSmith tracing is enabled by setting LANGCHAIN_TRACING_V2=true
# plus LANGSMITH_API_KEY in .env — the LangChain/LangGraph integration picks
# those up from the environment. Nothing to wire here until the graph exists.
