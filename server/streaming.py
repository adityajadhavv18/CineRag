"""Server-Sent Events for POST /api/v1/chat/stream (contract §4, §9, Day 8).

Same agent, same answer — delivered in pieces instead of one blob.

Three things happen here, and only the third is interesting:

  1. FRAMING. SSE is not a protocol, it is a plain HTTP response with no
     Content-Length and a dumb text format: `event:` / `data:` lines, a blank
     line meaning "that message is complete". See `_frame`.

  2. A WORKER THREAD. Stage events come from `graph.stream()`, which yields once
     per node; tokens arrive from *inside* a node, between those yields. One
     iterator cannot produce both, so the graph runs on its own thread and pushes
     both kinds into one queue that the response generator drains.

  3. LIVE CITATION RENUMBERING. This is the part specific to CineRAG — see
     `CitationStreamer`.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from collections.abc import Iterator
from typing import Any

import structlog

from agent.graph import get_agent
from agent.nodes.final_response import CITATION_PATTERN
from agent.state import initial_state
from core.llm import token_sink
from core.logger import get_logger

log = get_logger("stream")

# Bounded so a client that reads slowly applies backpressure to the graph thread
# rather than letting an unread answer accumulate in memory.
QUEUE_MAX = 256

# How long the response generator waits on the queue before giving up. Generous:
# it only expires if the worker thread died without posting its sentinel.
QUEUE_TIMEOUT_S = 120.0


class _Disconnected(BaseException):
    """The client hung up mid-answer; unwind the graph and stop paying OpenAI.

    Inherits BaseException on purpose. The nodes catch `Exception` to degrade
    gracefully (contract §5), and this is not a degradation — it must pass
    straight through `final_response`'s handler rather than being turned into a
    "couldn't write them up" fallback for nobody to read.
    """


# ── stage labels ──────────────────────────────────────────────────────────────
#
# Keyed by the node that just FINISHED, and naming what starts next — a label is
# a promise about the wait ahead, not a report on the wait behind. Deliberately
# free of node names, engines and store names: this is the only part of the
# pipeline a user ever sees, and "graph_enrich" means nothing to anyone.

_STARTING = "Working out what you're in the mood for…"
_SEARCHING = "Rummaging through the shelves…"
_ENRICHING = "Rounding up the usual suspects…"
_RANKING = "Arguing with myself about the order…"
_WRITING = "Right — here goes…"

_AFTER_NODE = {
    "vector_retrieve": _ENRICHING,
    "graph_retrieve": _ENRICHING,
    "graph_enrich": _RANKING,
    "rerank": _WRITING,
}


def _label_after(node: str, patch: dict) -> str | None:
    """What to promise next, now that `node` has finished."""
    if node == "intent":
        # The three no-retrieval intents skip straight to writing; announcing a
        # search that will never happen would be a small lie with a visible tell
        # (the label would sit there for the whole generation).
        chatty = patch.get("intent") in ("general", "off_topic", "clarification")
        return _WRITING if chatty else _SEARCHING
    return _AFTER_NODE.get(node)


# ── live citation renumbering ─────────────────────────────────────────────────

# A marker cut in half by a chunk boundary ("[1" now, "2]" next), or trailing
# whitespace that may or may not turn out to be the end of the answer. Both are
# held back rather than emitted, because neither can be rendered correctly yet.
_HOLD_BACK = re.compile(r"(\[\d*|\s+)$")


class CitationStreamer:
    """Applies `validate_citations`' renumbering incrementally, as tokens arrive.

    The model cites films by position in a 15-row candidate block; the caller
    only ever receives the handful actually cited, renumbered in order of first
    mention (see `agent/nodes/final_response.validate_citations` — that function
    stays the authority, this is the same rule applied one token at a time).

    It can be done incrementally because the rule needs nothing from the future:
    validity is `1 <= n <= len(rows)`, known before generation starts, and the
    new number is just "how many distinct films have been cited so far". So a
    marker's final form is decided the moment its closing bracket arrives, and
    the streamed text comes out byte-identical to the batch endpoint's. Without
    that, the user would watch "[7]" appear and then silently become "[2]".

    Proven against the batch function in tests/test_streaming.py.
    """

    def __init__(self) -> None:
        # None means pass-through: no candidate block exists, so there is nothing
        # to validate against. That is the correct behaviour for the general and
        # off_topic nodes, whose answers cite nothing.
        self.rows: list[dict] | None = None
        self.used: list[int] = []
        # Everything actually put on the wire. Compared against the final state's
        # `response` at the end, because nodes AFTER the generation can still add
        # to the answer — franchise appends a series timeline to text the LLM had
        # already finished writing.
        self.sent = ""
        self._buffer = ""
        self._started = False

    def set_rows(self, rows: list[dict]) -> None:
        """Called when rerank finishes — before final_response begins."""
        self.rows = rows

    @property
    def emitted(self) -> set[int]:
        """Renumbered positions already sent as `source` events."""
        return {i for i in range(1, len(self.used) + 1)}

    def feed(self, delta: str) -> tuple[str, list[dict]]:
        """Take a raw token; return (text to send, newly cited films)."""
        self._buffer += delta

        if self.rows is None:
            return self._release(self._buffer, reset=True), []

        out: list[str] = []
        sources: list[dict] = []
        cut = 0

        for match in CITATION_PATTERN.finditer(self._buffer):
            out.append(self._buffer[cut : match.start()])
            cut = match.end()

            n = int(match.group(1))
            if not 1 <= n <= len(self.rows):
                # Invented a reference. Drop the marker, exactly as the batch
                # path does — the surrounding sentence still reads fine.
                continue
            if n not in self.used:
                self.used.append(n)
                sources.append(_source(self.rows[n - 1], len(self.used)))
            out.append(f"[{self.used.index(n) + 1}]")

        out.append(self._buffer[cut:])
        return self._release("".join(out), reset=True), sources

    def flush(self) -> str:
        """Whatever was held back, once no more tokens are coming."""
        tail = self._buffer.rstrip() if self._buffer else ""
        self._buffer = ""
        if not self._started:
            tail = tail.lstrip()
        self.sent += tail
        return tail

    def _release(self, text: str, *, reset: bool) -> str:
        """Emit what is safe to render; keep the rest for the next token.

        Also reproduces `final_response`'s `.strip()`: leading whitespace is
        dropped before anything has been sent, and trailing whitespace is held
        (it is only trailing if the answer stops here, which we cannot yet know).
        """
        hold = _HOLD_BACK.search(text)
        if hold:
            self._buffer = text[hold.start() :]
            text = text[: hold.start()]
        elif reset:
            self._buffer = ""

        if not self._started:
            text = text.lstrip()
            if text:
                self._started = True
        self.sent += text
        return text


def _source(row: dict, n: int) -> dict:
    """One cited film, shaped like `schemas.Source` plus its marker number."""
    return {
        "n": n,
        "tmdb_id": row["tmdb_id"],
        "title": row["title"],
        "year": row.get("year"),
        "overview": row.get("overview"),
        "poster_path": row.get("poster_path"),
        "backdrop_path": row.get("backdrop_path"),
    }


# ── the worker ────────────────────────────────────────────────────────────────


def _run_graph(
    message: str,
    history: list[dict[str, str]],
    events: "queue.Queue[tuple[str, dict] | None]",
    stop: threading.Event,
    request_id: str,
) -> None:
    """Run one turn, posting events as they happen. Runs on its own thread."""
    # Both of these are ContextVar-backed and do NOT cross the thread boundary,
    # so they are established here rather than in the request handler. Skip the
    # binding and every log line for a streamed request loses its request id.
    structlog.contextvars.bind_contextvars(request_id=request_id)
    streamer = CitationStreamer()

    def on_token(delta: str) -> None:
        if stop.is_set():
            raise _Disconnected
        text, sources = streamer.feed(delta)
        for source in sources:
            events.put(("source", source))
        if text:
            events.put(("token", {"text": text}))

    final: dict[str, Any] = {}

    try:
        events.put(("stage", {"label": _STARTING}))
        announced = _STARTING

        with token_sink(on_token):
            for mode, chunk in get_agent().stream(
                initial_state(message, history), stream_mode=["updates", "values"]
            ):
                if mode == "values":
                    # The merged state after this superstep. Taking it from
                    # LangGraph rather than folding the patches ourselves is what
                    # keeps the reducers in state.py the single definition of how
                    # a parallel fan-out combines.
                    final = chunk
                    continue

                for node, patch in (chunk or {}).items():
                    if node == "rerank":
                        # Ordering is guaranteed: this update is handed over
                        # while the graph is suspended, before final_response
                        # runs, so the streamer always has its block in time.
                        streamer.set_rows(patch.get("reranked") or [])

                    label = _label_after(node, patch or {})
                    # A fan-out finishes twice; announcing the same thing twice
                    # would read as a stutter.
                    if label and label != announced:
                        events.put(("stage", {"label": label}))
                        announced = label

        tail = streamer.flush()
        if tail:
            events.put(("token", {"text": tail}))

        # Anything a later node added to the answer. `franchise` appends its
        # series timeline to text the LLM had already finished, and the two
        # no-LLM nodes (off_topic, clarification) write a canned reply that never
        # passed a token sink at all — both arrive here as one final chunk.
        authoritative = final.get("response") or ""
        if authoritative.startswith(streamer.sent):
            remainder = authoritative[len(streamer.sent) :]
            if remainder:
                events.put(("token", {"text": remainder}))
        elif authoritative != streamer.sent:
            # Not expected: the renumbering is meant to reproduce the batch text
            # exactly. Harmless — the client takes `response` from `done` — but
            # it means the two paths have drifted, so say so loudly.
            log.warning("stream_text_diverged", streamed=len(streamer.sent),
                        final=len(authoritative))

        rows = final.get("reranked") or []
        for citation in final.get("citations") or []:
            # Normally empty: every citation was emitted as its marker streamed.
            # It catches the one case that has no marker — the detail-mode source
            # final_response infers when a profile answer forgot to cite.
            if citation["n"] in streamer.emitted:
                continue
            row = next((r for r in rows if r.get("tmdb_id") == citation["tmdb_id"]), citation)
            events.put(("source", _source({**citation, **row}, citation["n"])))

        errors = final.get("retrieval_errors") or []
        events.put((
            "done",
            {
                # The authoritative text. Identical to what was streamed — it is
                # sent so the client stores the agent's exact words in history,
                # which is what the next turn replays.
                "response": final.get("response") or "",
                "intent": final.get("intent"),
                "lead_engine": final.get("lead_engine"),
                "franchise": final.get("franchise") or [],
                # Only ever set by clarification_node. It rides on `done` rather
                # than as an event of its own because it is known before the
                # first token — there is nothing to stream, and the client needs
                # the whole question set at once to render it.
                "clarification": final.get("clarification"),
                "degraded": bool(errors),
                "trace": final.get("trace") or [],
            },
        ))

    except _Disconnected:
        log.info("stream_abandoned", reason="client_disconnected")
    except Exception as exc:  # noqa: BLE001
        # Headers went out with the first frame, so a 500 is no longer available.
        # The failure has to travel as an event and be rendered by the client.
        log.exception("stream_failed", error=type(exc).__name__)
        events.put((
            "error",
            {"detail": f"Something went wrong handling that request (ref {request_id})."},
        ))
    finally:
        structlog.contextvars.clear_contextvars()
        events.put(None)


# ── framing ───────────────────────────────────────────────────────────────────


def _frame(event: str, data: dict) -> str:
    """One SSE message.

    The blank line is the delimiter — it says "this message is complete, deliver
    it". Without it the client cannot know whether another `data:` line is still
    coming. Newlines inside the payload would break that, which is why the data
    is JSON on a single line.
    """
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def sse_stream(
    message: str, history: list[dict[str, str]], request_id: str
) -> Iterator[str]:
    """The response body: SSE frames, drained from the worker's queue.

    A plain `def` generator, so Starlette iterates it in a threadpool and the
    blocking work never touches the event loop — the same reasoning that keeps
    the batch endpoint synchronous.
    """
    events: queue.Queue[tuple[str, dict] | None] = queue.Queue(maxsize=QUEUE_MAX)
    stop = threading.Event()
    worker = threading.Thread(
        target=_run_graph,
        args=(message, history, events, stop, request_id),
        name=f"chat-{request_id}",
        daemon=True,
    )
    worker.start()

    try:
        # A comment frame (a line starting with ':'). Clients ignore it; it exists
        # to push the headers out now, so the browser opens the stream instead of
        # waiting on the first real event several seconds later.
        yield ": open\n\n"

        while True:
            try:
                item = events.get(timeout=QUEUE_TIMEOUT_S)
            except queue.Empty:
                log.error("stream_stalled", seconds=QUEUE_TIMEOUT_S)
                yield _frame("error", {"detail": "That took too long. Try again?"})
                return
            if item is None:
                return
            yield _frame(*item)
    finally:
        # Reached on client disconnect too (the generator is closed). Tells the
        # worker to unwind at its next token rather than finishing an answer that
        # nobody will read.
        stop.set()
