"""Day 5 invariants: RRF fusion arithmetic and citation grounding.

Almost all of this is pure — no stores, no API — because fusion is arithmetic and
citation validation is string handling. That is a design property worth noticing:
the two places most likely to produce a wrong ANSWER are both cheaply testable.
"""

from __future__ import annotations

import pytest

from agent.nodes.rerank import (
    GENRE_BONUS_MAX,
    QUALITY_BONUS_MAX,
    RRF_K,
    TOP_K,
    rerank,
)
from agent.nodes.final_response import build_context, validate_citations


def _row(tmdb_id: int, rank: int, title: str = None, **kw) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "title": title or f"Film {tmdb_id}",
        "year": 2000,
        "rank": rank,
        **kw,
    }


# ── RRF arithmetic (contract §3.5) ───────────────────────────────────────────


def test_appearing_in_both_lists_doubles_the_score():
    """The headline claim: the overlap win EMERGES from the sum, no special rule.

    Same rank in both lists must score exactly twice a single appearance.
    """
    state = {
        "vector_results": [_row(1, 1), _row(2, 2)],
        "graph_results": [_row(1, 1)],  # film 1 in both, film 2 in one
        "filters": {},
    }
    ranked = rerank(state)["reranked"]
    by_id = {e["tmdb_id"]: e for e in ranked}

    assert by_id[1]["rrf_score"] == pytest.approx(2 / (RRF_K + 1))
    assert by_id[1]["in_both"] is True
    assert by_id[2]["in_both"] is False
    assert by_id[1]["rrf_score"] > by_id[2]["rrf_score"]


def test_both_lists_beats_a_better_rank_in_one_list():
    """A film ranked 1st by ONE store loses to a film ranked 10th by BOTH.

    This is the behaviour the architecture is sold on — cross-store agreement
    outweighing a single store's confidence.
    """
    state = {
        "vector_results": [_row(1, 1), _row(2, 10)],
        "graph_results": [_row(2, 10)],
        "filters": {},
    }
    ranked = rerank(state)["reranked"]
    assert ranked[0]["tmdb_id"] == 2, "cross-store agreement should win"


def test_dedupe_accumulates_rather_than_keeping_the_best_row():
    """One entry per tmdb_id, holding the SUM (contract §3.5)."""
    state = {
        "vector_results": [_row(7, 3)],
        "graph_results": [_row(7, 5)],
        "filters": {},
    }
    ranked = rerank(state)["reranked"]
    assert len(ranked) == 1
    assert ranked[0]["rrf_score"] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 5))
    assert ranked[0]["ranks"] == {"vector": 3, "graph": 5}


def test_bonuses_stay_small_relative_to_the_rrf_spread():
    """Bonuses must nudge neighbours, not reorder the list (contract §3.5).

    The whole-list spread is rank1 - rank20 in a single list. If the bonuses
    could exceed it, a highly-rated irrelevant film could leapfrog the entire
    fused ranking and the fusion would be decorative.
    """
    spread = 1 / (RRF_K + 1) - 1 / (RRF_K + 20)
    assert GENRE_BONUS_MAX + QUALITY_BONUS_MAX < spread / 2, (
        "combined bonuses approach the full RRF spread — they would reorder, not nudge"
    )


def test_a_bonus_cannot_overturn_cross_store_agreement():
    """Concretely: a perfect single-list film must not beat a both-list film."""
    state = {
        # Film 1: rank 1 in both lists, terrible rating, wrong genre.
        "vector_results": [_row(1, 1, rating=1.0, genres=["Documentary"]), _row(2, 1, rating=10.0, genres=["Horror"])],
        "graph_results": [_row(1, 1, rating=1.0, genres=["Documentary"])],
        "filters": {"genres": ["Horror"]},
    }
    ranked = rerank(state)["reranked"]
    assert ranked[0]["tmdb_id"] == 1, "bonuses overturned the fusion result"


def test_genre_bonus_scales_with_how_many_requested_genres_match():
    state = {
        "vector_results": [_row(1, 1, genres=["Crime", "Drama"]), _row(2, 2, genres=["Crime"])],
        "graph_results": [],
        "filters": {"genres": ["Crime", "Drama"]},
    }
    ranked = rerank(state)["reranked"]
    by_id = {e["tmdb_id"]: e for e in ranked}
    assert by_id[1]["genre_bonus"] == pytest.approx(GENRE_BONUS_MAX)
    assert by_id[2]["genre_bonus"] == pytest.approx(GENRE_BONUS_MAX / 2)


def test_output_is_capped_at_top_k():
    state = {
        "vector_results": [_row(i, i) for i in range(1, 40)],
        "graph_results": [],
        "filters": {},
    }
    assert len(rerank(state)["reranked"]) == TOP_K


def test_empty_input_produces_empty_output_not_an_error():
    assert rerank({"vector_results": [], "graph_results": [], "filters": {}})["reranked"] == []


# ── citation grounding (contract §5) ─────────────────────────────────────────


def test_citations_report_only_films_the_answer_used():
    """A film offered but not mentioned must not appear as a source."""
    rows = [_row(10, 1, title="A"), _row(20, 2, title="B"), _row(30, 3, title="C")]
    _, citations, _ = validate_citations("I liked A [1] and C [3].", rows)
    assert [c["title"] for c in citations] == ["A", "C"]


def test_out_of_range_citations_are_stripped():
    """The model inventing [27] when 3 films were offered must not survive."""
    rows = [_row(10, 1, title="A")]
    text, citations, invalid = validate_citations("Try A [1] and also [27].", rows)
    assert "[27]" not in text
    assert invalid == [27]
    assert len(citations) == 1


def test_repeated_citations_are_reported_once():
    rows = [_row(10, 1, title="A")]
    _, citations, _ = validate_citations("A [1] is great. Really, A [1].", rows)
    assert len(citations) == 1


def test_citations_carry_the_poster_paths_the_frontend_needs():
    """Contract §9: each cited film ships poster/backdrop for the card UI."""
    rows = [_row(10, 1, title="A", poster_path="/p.jpg", backdrop_path="/b.jpg")]
    _, citations, _ = validate_citations("A [1]", rows)
    assert citations[0]["poster_path"] == "/p.jpg"
    assert citations[0]["backdrop_path"] == "/b.jpg"


def test_context_block_is_numbered_from_one():
    """Numbering is what makes a citation checkable at all."""
    rows = [_row(10, 1, title="A"), _row(20, 2, title="B")]
    context = build_context(rows, {})
    assert "[1] A" in context
    assert "[2] B" in context
    assert "[0]" not in context


def test_context_includes_the_facts_an_explanation_needs():
    rows = [_row(10, 1, title="A", genres=["Crime"], directors=["Someone"], rating=8.0)]
    context = build_context(rows, {})
    assert "Crime" in context and "Someone" in context and "8.0" in context
