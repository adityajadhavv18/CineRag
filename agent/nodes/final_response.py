"""final_response — grounded, cited answers (contract §3.5, §5, Day 5).

The rule this node exists to enforce: **it may only name films that retrieval
actually returned.** Everything here serves that.

  1. Build a NUMBERED context block from the reranked films. Numbering is what
     makes citation checkable — "[3]" points at a specific row we can verify,
     where a bare title cannot be told apart from one the model remembered.
  2. Prompt the LLM to write from that block and cite [N].
  3. Validate afterwards. A prompt is guidance; the check is the guarantee.
     Any [N] outside the block is stripped, and `citations` ends up holding only
     the films the answer actually used.

Empty retrieval short-circuits before the LLM: with nothing to ground on, the
only honest answer is "I found nothing", and asking a model to write that is an
invitation for it to fill the gap from memory.
"""

from __future__ import annotations

import re

from core.llm import chat
from core.logger import get_logger
from agent.state import AgentState

log = get_logger("final_response")

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You are a movie recommendation assistant. Answer using ONLY the numbered films \
in the CANDIDATES block. That block is the complete set of films you know about \
right now.

Rules, in order of importance:
1. Never name a film that is not in the CANDIDATES block. Not as a comparison, \
not as an aside, not "if you liked X" — if it is not numbered below, it does not \
exist for this answer.
2. Cite every film you mention with its number in square brackets, like [3].
3. Say WHY each film fits what the user asked — reference its plot, mood, cast or \
director from the block, not generic praise.
4. Recommend 3-5 films unless the user clearly wants a single answer. Lead with \
the best fit.
5. Be concise and warm. No preamble like "Certainly!" — start with the substance.
"""

FACTUAL_PROMPT = """\
You are a movie assistant answering a specific factual question. Use ONLY the \
numbered films in the CANDIDATES block.

Rules:
1. Answer the question directly in the first sentence.
2. Cite each film with its number in square brackets, like [2].
3. Never name a film that is not in the block.
4. If the block does not contain what was asked, say so plainly rather than \
guessing.
5. Be brief — this is a question, not a recommendation.
"""


def build_context(rows: list[dict], enrichment: dict) -> str:
    """The numbered block the answer must be built from.

    Include the fields an explanation needs (director, cast, genres, year,
    rating) so the model can say WHY a film fits without inventing detail.
    """
    lines = []
    for i, row in enumerate(rows, start=1):
        links = enrichment.get(row.get("tmdb_id"), {})
        directors = row.get("directors") or links.get("directors") or []
        cast = row.get("cast_names") or links.get("cast") or []
        genres = row.get("genres") or links.get("genres") or []

        bits = [f"[{i}] {row['title']} ({row.get('year') or 'year unknown'})"]
        if genres:
            bits.append(f"genres: {', '.join(genres[:4])}")
        if directors:
            bits.append(f"directed by {', '.join(directors[:2])}")
        if cast:
            bits.append(f"starring {', '.join(cast[:4])}")
        if row.get("rating"):
            bits.append(f"rated {row['rating']:.1f}")
        if links.get("collection_name"):
            bits.append(f"part of {links['collection_name']}")
        lines.append("  " + " | ".join(bits))
    return "\n".join(lines)


def validate_citations(text: str, rows: list[dict]) -> tuple[str, list[dict]]:
    """Strip out-of-range citations; return the films actually cited.

    Two distinct failures are handled:
      - a citation numbered beyond the block ([27] when 15 films were offered):
        the model invented a reference, so the marker is removed from the text.
      - a film in the block the answer never mentioned: it simply does not appear
        in `citations`, so the API never reports a source the user was not shown.
    """
    used: list[int] = []
    invalid: list[int] = []

    for match in CITATION_PATTERN.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= len(rows):
            if n not in used:
                used.append(n)
        elif n not in invalid:
            invalid.append(n)

    cleaned = text
    for n in invalid:
        cleaned = cleaned.replace(f"[{n}]", "")

    citations = [
        {
            "n": n,
            "tmdb_id": rows[n - 1]["tmdb_id"],
            "title": rows[n - 1]["title"],
            "year": rows[n - 1].get("year"),
            "poster_path": rows[n - 1].get("poster_path"),
            "backdrop_path": rows[n - 1].get("backdrop_path"),
        }
        for n in used
    ]
    return cleaned, citations, invalid


def _honest_empty(state: AgentState) -> dict:
    """No candidates. Distinguish an outage from a genuine miss (contract §5)."""
    errors = state.get("retrieval_errors") or []
    if errors:
        stores = ", ".join(sorted({e["store"] for e in errors}))
        response = (
            f"I couldn't reach my {stores} data just now, so I can't answer this "
            "reliably. Please try again in a moment."
        )
        log.warning("response_degraded", errors=errors)
    else:
        response = (
            "I couldn't find anything in my catalogue matching that. My catalogue "
            "is about 5,000 films, so it may simply not be in there — try loosening "
            "the constraints, or ask me for something adjacent."
        )
        log.info("response_empty", reason="no_matches")

    return {"response": response, "citations": [], "trace": ["final_response(empty)"]}


def final_response(state: AgentState) -> dict:
    rows = state.get("reranked") or []
    if not rows:
        return _honest_empty(state)

    enrichment = state.get("enrichment") or {}
    context = build_context(rows, enrichment)
    is_factual = state.get("intent") == "factual_lookup"

    user_message = (
        f"User asked: {state.get('refined_query') or state['query']}\n\n"
        f"CANDIDATES:\n{context}"
    )

    try:
        raw = chat(
            [
                {"role": "system", "content": FACTUAL_PROMPT if is_factual else SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001 — contract §5
        log.error("generation_failed", error=type(exc).__name__)
        titles = ", ".join(f"{r['title']} ({r.get('year')})" for r in rows[:5])
        return {
            "response": f"I found these but couldn't write them up: {titles}",
            "citations": [],
            "trace": ["final_response(generation_failed)"],
        }

    text, citations, invalid = validate_citations(raw, rows)

    log.info(
        "response_generated",
        intent=state.get("intent"),
        candidates=len(rows),
        cited=len(citations),
        invalid_citations=invalid,
        chars=len(text),
    )

    return {"response": text.strip(), "citations": citations, "trace": [f"final_response({len(citations)} cited)"]}
