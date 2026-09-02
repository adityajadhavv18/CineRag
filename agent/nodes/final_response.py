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

# "Tell me about Inception" is not a factual lookup with a missing fact — it is a
# request for a profile. Answering it with one sentence ("directed by Christopher
# Nolan") is technically correct and useless.
#
# The dangerous part is rule 2. A model asked to describe a film it was not given
# a plot for will write a fluent, accurate-sounding synopsis FROM MEMORY — and
# validate_citations cannot catch that, because it checks which films are named,
# never what is claimed about them. Supplying the real overview is what makes the
# description checkable; the rule is what points the model at it.
DETAIL_PROMPT = """\
You are a movie assistant describing a film the user asked about. Use ONLY the \
numbered films in the CANDIDATES block.

Rules:
1. Describe the film the user named: what it is about, who made it, who is in it, \
when it came out, and how it is regarded.
2. The plot summary MUST be a paraphrase of the `plot:` text given for that film. \
Do not add plot events, characters, twists or endings that are not in that text — \
even if you know the film. If no plot text is given, describe the film from the \
other fields and do not invent a synopsis.
3. Cite the film with its number in square brackets the FIRST time you name it, like \
"Inception [1] is a ...". This is required — the citation is what links the answer to \
a real catalogue entry.
4. Never name a film that is not in the block.
5. Two or three short paragraphs, or a short paragraph plus a few bullets. \
Conversational, not a database dump.
"""

# Words that mean the user wants ONE specific fact rather than a profile.
ATTRIBUTE_WORDS = (
    "who direct", "who wrote", "who stars", "who is in", "who acted",
    "what year", "when was", "when did", "how long", "runtime",
    "rating", "how many", "which year", "released",
)


def wants_detail(state: AgentState) -> bool:
    """Open request for a profile, or a pointed question?

    Decided from the extracted state rather than the phrasing where possible: a
    named title with no attribute word in the question is a "tell me about X".
    Same shape as the lead_engine decision table — derive it, don't pattern-match
    on wording.
    """
    if state.get("intent") != "factual_lookup":
        return False
    if not (state.get("entities") or {}).get("titles"):
        return False

    # Read the user's OWN words, not refined_query. refined_query is model-written
    # and can drift: "tell me about Inception" was once rewritten to "who directed
    # Inception", which flipped this check and produced a one-line answer to an
    # open question. The prompt now forbids that narrowing, but the shape of the
    # request is something only the original phrasing can be trusted to carry.
    question = (state.get("query") or "").lower()
    if not question:
        question = (state.get("refined_query") or "").lower()
    return not any(word in question for word in ATTRIBUTE_WORDS)


def build_context(rows: list[dict], enrichment: dict, include_plot: bool = False) -> str:
    """The numbered block the answer must be built from.

    Include the fields an explanation needs (director, cast, genres, year,
    rating) so the model can say WHY a film fits without inventing detail.

    `include_plot` is off by default on purpose. An overview is ~70 tokens, so
    adding it to 15 recommendation candidates costs ~1,000 tokens per request for
    text the answer will barely use. A detail query has one or two candidates and
    needs the plot, so it pays for itself there and nowhere else.
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
        if row.get("runtime"):
            bits.append(f"{row['runtime']} min")
        if links.get("collection_name") or row.get("collection_name"):
            bits.append(f"part of {links.get('collection_name') or row.get('collection_name')}")
        line = "  " + " | ".join(bits)

        if include_plot:
            # Labelled `plot:` and `tagline:` so the prompt can point at them by
            # name — "paraphrase the plot: text" is checkable guidance in a way
            # that "use the given information" is not.
            if row.get("tagline"):
                line += f"\n      tagline: {row['tagline']}"
            if row.get("overview"):
                line += f"\n      plot: {row['overview']}"
        lines.append(line)
    return "\n".join(lines)


def validate_citations(text: str, rows: list[dict]) -> tuple[str, list[dict]]:
    """Strip out-of-range citations; return the films actually cited.

    Two distinct failures are handled:
      - a citation numbered beyond the block ([27] when 15 films were offered):
        the model invented a reference, so the marker is removed from the text.
      - a film in the block the answer never mentioned: it simply does not appear
        in `citations`, so the API never reports a source the user was not shown.

    The surviving markers are then RENUMBERED. The model cites films by their
    position in the candidate block it was shown — up to 15 rows — but a caller
    only ever receives the handful that were actually cited. Left alone, an
    answer citing candidates 2, 5 and 6 ships three sources numbered 1, 2, 3 and
    a body reading "[2] ... [5] ... [6]", so "[5]" points at nothing and the
    third source is never referenced. Renumbering closes that gap: after this,
    "[k]" is always the k-th entry of `citations`, which is what §9 promises the
    frontend and the only thing that makes a marker clickable.
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

    # Candidate index -> position in `citations`, assigned in order of first
    # mention so the numbering follows the reading order of the answer.
    renumbered = {n: i for i, n in enumerate(used, start=1)}

    # One pass, not a loop of str.replace: rewriting in place would let an
    # already-renumbered marker be rewritten again by a later mapping (6 -> 1
    # then 1 -> 4), which silently corrupts the very links this exists to fix.
    def rewrite(match: re.Match[str]) -> str:
        n = int(match.group(1))
        return f"[{renumbered[n]}]" if n in renumbered else ""

    cleaned = CITATION_PATTERN.sub(rewrite, text)

    citations = [
        {
            "n": renumbered[n],
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
    detail = wants_detail(state)
    is_factual = state.get("intent") == "factual_lookup"

    if detail:
        mode, prompt = "detail", DETAIL_PROMPT
    elif is_factual:
        mode, prompt = "factual", FACTUAL_PROMPT
    else:
        mode, prompt = "recommend", SYSTEM_PROMPT

    context = build_context(rows, enrichment, include_plot=detail)

    user_message = (
        f"User asked: {state.get('refined_query') or state['query']}\n\n"
        f"CANDIDATES:\n{context}"
    )

    try:
        raw = chat(
            [
                {"role": "system", "content": prompt},
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

    # Safety net for detail mode. The whole answer is ABOUT one film, so an
    # uncited profile still has an unambiguous source — and without a citation the
    # API returns no `sources`, leaving the frontend with no poster to render and
    # franchise_node with nothing to build a timeline from.
    #
    # This attaches a source the answer genuinely used; it never invents one. The
    # rule it must not break is the reverse direction: reporting a film the answer
    # did NOT discuss.
    if detail and not citations and rows:
        citations = [
            {
                "n": 1,
                "tmdb_id": rows[0]["tmdb_id"],
                "title": rows[0]["title"],
                "year": rows[0].get("year"),
                "poster_path": rows[0].get("poster_path"),
                "backdrop_path": rows[0].get("backdrop_path"),
            }
        ]
        log.info("detail_citation_inferred", title=rows[0]["title"],
                 reason="answer describes this film but emitted no [N]")

    log.info(
        "response_generated",
        intent=state.get("intent"),
        mode=mode,
        plot_supplied=detail,
        candidates=len(rows),
        cited=len(citations),
        invalid_citations=invalid,
        chars=len(text),
    )

    return {"response": text.strip(), "citations": citations, "trace": [f"final_response({len(citations)} cited)"]}
