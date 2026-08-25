"""The labelled eval dataset (contract §4, Day 7).

What makes an eval trustworthy is that the answers are fixed BEFORE the run and
the criteria are explicit. These labels were written from the contract, not from
whatever the agent happens to do today — otherwise the eval just certifies the
current behaviour as correct, including its bugs.

Each case carries only what can be checked objectively:
  intent          which branch must fire
  lead_engine     which store(s) must lead (None = don't care)
  must_include    titles that MUST appear in the cited sources
  must_not_cite   nothing may be cited (empty results, off-topic, clarification)
  expect_question the answer must ask something back
  grounded_plot   plot claims must come from the stored overview

There is no "is this a good recommendation?" field. Taste is not testable; the
things above are.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Case:
    query: str
    intent: str
    lead_engine: str | None = None
    # Titles the ANSWER must cite.
    must_include: list[str] = field(default_factory=list)
    # Titles retrieval must SURFACE, without needing to be recommended. A
    # reference film belongs here, not in must_include: asked for "films like
    # Inception", citing Inception itself would be a worse answer, but failing to
    # retrieve it would mean the reference was never used.
    must_retrieve: list[str] = field(default_factory=list)
    must_not_cite: bool = False
    expect_question: bool = False
    grounded_plot: bool = False
    history: list[dict] = field(default_factory=list)
    note: str = ""


CASES: list[Case] = [
    # ── the cheap paths: no retrieval may run ────────────────────────────────
    Case("hi", "general", must_not_cite=True),
    Case("what can you do?", "general", must_not_cite=True),
    Case("thanks, that's great", "general", must_not_cite=True),
    Case("what's the weather in Paris", "off_topic", must_not_cite=True),
    Case("write me a python script to sort a list", "off_topic", must_not_cite=True),
    Case("who is the president of France", "off_topic", must_not_cite=True),

    # ── fact queries: the graph must lead and be exactly right ───────────────
    Case("movies directed by Bong Joon-ho", "factual_lookup", "graph",
         must_include=["Parasite"],
         note="name variant: TMDB stores 'Bong Joon Ho'"),
    Case("who directed Whiplash", "factual_lookup", "graph", must_include=["Whiplash"]),
    Case("what else is Cillian Murphy in", "factual_lookup", "graph"),
    Case("films directed by Christopher Nolan", "factual_lookup", "graph",
         must_include=["Inception"]),

    # ── vibe queries: vector leads, no named entity ──────────────────────────
    Case("something tense and claustrophobic", "recommend", "vector"),
    Case("a feel-good film for a rainy Sunday", "recommend", "vector"),
    Case("mind-bending science fiction about memory", "recommend", "vector"),

    # ── blended: a vibe AND a named person/title ─────────────────────────────
    Case("gritty crime dramas starring Denzel Washington", "recommend", "both",
         must_include=["Training Day"]),
    Case("gritty crime dramas with Denzel Washington", "recommend", "both",
         note="phrasing invariance: 'with' must route like 'starring'"),
    Case("mind-bending sci-fi like Inception", "recommend", "both",
         must_retrieve=["Inception"],
         note="reference film: must be RETRIEVED, but recommending it back is a worse answer"),
    Case("something like a Tarantino film", "recommend", "both",
         note="soft signal: must NOT exclude to Tarantino's own films"),

    # ── clarification: too vague, must ask rather than guess ─────────────────
    Case("recommend some movies", "clarification",
         must_not_cite=True, expect_question=True),
    Case("recommend some action movies", "clarification",
         must_not_cite=True, expect_question=True),
    Case("I want something good", "clarification",
         must_not_cite=True, expect_question=True),

    # ── follow-up: history must fuse into a self-contained query ─────────────
    Case("only the 90s ones", "follow_up",
         history=[
             {"role": "user", "content": "recommend action movies with Jackie Chan"},
             {"role": "assistant", "content": "Here are some Jackie Chan action films."},
         ],
         note="'ones' is meaningless without history"),
    Case("what about her other films", "follow_up",
         history=[
             {"role": "user", "content": "tell me about Parasite"},
             {"role": "assistant", "content": "Parasite [1] is directed by Bong Joon Ho."},
         ]),

    # ── film detail: profile, with plot grounded in the stored overview ──────
    Case("tell me about Inception", "factual_lookup", "graph",
         must_include=["Inception"], grounded_plot=True,
         note="'Inception' is also an ordinary word — must not route to general"),
    Case("tell me about The Matrix", "factual_lookup", "graph",
         must_include=["The Matrix"], grounded_plot=True),
    Case("what is Parasite about", "factual_lookup", "graph",
         must_include=["Parasite"], grounded_plot=True),

    # ── franchise ────────────────────────────────────────────────────────────
    Case("recommend the Harry Potter films", "recommend",
         must_include=["Harry Potter and the Philosopher's Stone"],
         note="must also emit an ordered timeline"),

    # ── honest emptiness: nothing matches, nothing may be invented ───────────
    Case("horror movies rated above 9.8 released in the 1890s", "recommend",
         must_not_cite=True,
         note="impossible constraints — must admit it, not fabricate"),
    Case("films directed by Zxqvwrt Nonexistentsson", "factual_lookup",
         must_not_cite=True,
         note="a director who does not exist"),
]

# Plot details a model reaching into its own memory would very likely produce,
# and which appear in NONE of the stored overviews. Used by the grounding check.
UNGROUNDED_PLOT_TELLS = {
    "Inception": ["spinning top", "totem", "limbo", "ariadne", "dream within a dream"],
    "The Matrix": ["red pill", "blue pill", "there is no spoon", "bullet time"],
    "Parasite": ["basement bunker", "flooding", "birthday party stabbing"],
}


def as_langsmith_examples() -> list[dict]:
    """Shape the cases for a LangSmith dataset."""
    return [
        {
            "inputs": {"query": c.query, "history": c.history},
            "outputs": {
                "intent": c.intent,
                "lead_engine": c.lead_engine,
                "must_include": c.must_include,
                "must_not_cite": c.must_not_cite,
                "expect_question": c.expect_question,
                "grounded_plot": c.grounded_plot,
            },
            "metadata": {"note": c.note},
        }
        for c in CASES
    ]


if __name__ == "__main__":
    from collections import Counter

    print(f"{len(CASES)} cases")
    for intent, n in Counter(c.intent for c in CASES).most_common():
        print(f"  {intent:<16} {n}")
