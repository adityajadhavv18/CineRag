"""The streaming path must produce exactly what the batch path produces.

That is the one property everything else rests on. `CitationStreamer` reimplements
`validate_citations`' renumbering incrementally, and the whole justification for
doing so is that the result is indistinguishable — if it drifts, users watch
citation numbers change under them and markers point at the wrong film.

So these tests do not check the streamer against a hand-written expectation. They
check it against `validate_citations` itself, over the same text, at every chunk
size. `final_response` stays the authority; this only proves the copy agrees.
"""

from __future__ import annotations

import pytest

from agent.nodes.final_response import validate_citations
from server.streaming import CitationStreamer

ROWS = [
    {"tmdb_id": 949, "title": "Heat", "year": 1995, "overview": "A crew of thieves.",
     "poster_path": "/heat.jpg", "backdrop_path": "/heat-bd.jpg"},
    {"tmdb_id": 1998, "title": "Collateral", "year": 2004, "overview": "A cab driver.",
     "poster_path": "/col.jpg", "backdrop_path": None},
    {"tmdb_id": 111, "title": "Scarface", "year": 1983, "overview": None,
     "poster_path": None, "backdrop_path": None},
]

ANSWERS = [
    # The ordinary case: cited out of order, so renumbering actually does work.
    "Heat [2] is the obvious pick. Collateral [3] is leaner, and Scarface [1] is louder.",
    # A citation past the end of the block — the model invented a reference.
    "Try Heat [1], or maybe [27] if you want something else.",
    # The same film cited twice keeps one number.
    "Heat [1] earns it. Later, Heat [1] again.",
    # Bullets, bold and newlines — what RichText actually receives.
    "Two picks:\n\n- **Heat** [3] — the shootout.\n- **Collateral** [2] — one night.",
    # Nothing cited at all.
    "I could not find anything matching that.",
    # Leading and trailing whitespace, which final_response strips.
    "\n\n  Heat [1] it is.  \n\n",
]


def drive(text: str, size: int, rows: list[dict] | None = ROWS) -> tuple[str, list[dict]]:
    """Push `text` through the streamer in `size`-character chunks."""
    streamer = CitationStreamer()
    if rows is not None:
        streamer.set_rows(rows)

    out: list[str] = []
    sources: list[dict] = []
    for start in range(0, len(text), size):
        emitted, new = streamer.feed(text[start : start + size])
        out.append(emitted)
        sources.extend(new)
    out.append(streamer.flush())
    return "".join(out), sources


@pytest.mark.parametrize("answer", ANSWERS)
@pytest.mark.parametrize("size", [1, 2, 3, 5, 13, 500])
def test_streamed_text_matches_the_batch_path(answer: str, size: int) -> None:
    """Whatever the chunk boundaries, the user reads the same sentence."""
    expected, _, _ = validate_citations(answer, ROWS)
    streamed, _ = drive(answer, size)
    assert streamed == expected.strip()


@pytest.mark.parametrize("answer", ANSWERS)
@pytest.mark.parametrize("size", [1, 7, 500])
def test_sources_match_the_batch_citations(answer: str, size: int) -> None:
    """And each [N] points at the same film it would have pointed at."""
    _, expected, _ = validate_citations(answer, ROWS)
    _, sources = drive(answer, size)

    assert [s["n"] for s in sources] == [c["n"] for c in expected]
    assert [s["tmdb_id"] for s in sources] == [c["tmdb_id"] for c in expected]


def test_a_marker_split_across_chunks_is_never_shown_half_written() -> None:
    """The reason `feed` holds bytes back at all.

    Emitting eagerly would put a bare "[1" on screen for one frame — and then a
    stray "2]" once the rest arrived, since the marker would never be recognised.
    """
    streamer = CitationStreamer()
    streamer.set_rows(ROWS)

    first, _ = streamer.feed("Watch Heat [")
    second, _ = streamer.feed("2")
    third, sources = streamer.feed("] tonight.")

    assert "[" not in first and "[" not in second
    assert first + second + third == "Watch Heat [1] tonight."
    assert [s["tmdb_id"] for s in sources] == [1998]


def test_a_source_arrives_with_the_marker_that_needs_it() -> None:
    """A marker must never render before its film is known, or it is a dead
    button for as long as the gap lasts."""
    streamer = CitationStreamer()
    streamer.set_rows(ROWS)

    text, sources = streamer.feed("Heat [2] is great.")
    assert "[1]" in text
    assert len(sources) == 1
    assert sources[0]["title"] == "Collateral"
    assert sources[0]["overview"] == "A cab driver."


def test_without_a_candidate_block_text_passes_through() -> None:
    """The general and off_topic nodes answer with no candidates behind them.

    Their replies are not validated in the batch path either, so the streamer
    must not start stripping things there — a chatty answer that happens to
    contain "[1]" is the user's problem, not ours to silently rewrite.
    """
    streamed, sources = drive("Hello! I only know about films [1].", 4, rows=None)
    assert streamed == "Hello! I only know about films [1]."
    assert sources == []


def test_an_empty_block_strips_every_marker() -> None:
    """Distinct from the case above: rerank ran and returned nothing, so every
    marker is out of range and none of them can be honoured."""
    streamer = CitationStreamer()
    streamer.set_rows([])
    text, sources = streamer.feed("Nothing found [1].")
    assert text == "Nothing found ."
    assert sources == []
