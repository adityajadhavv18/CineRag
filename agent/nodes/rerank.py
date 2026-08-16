"""rerank — Reciprocal Rank Fusion of the two candidate lists (contract §3.5).

Two judges rank the same competition. You cannot average their SCORES — Qdrant
returns a cosine similarity, Cypher returns unscored rows, and there is no common
unit. You can average their POSITIONS, because "1st" means the same thing to
everyone. That is all RRF is.

    rrf(movie) = Σ over each list containing it:  1 / (K + rank_in_that_list)

A film both stores ranked collects TWO contributions that sum, so "appears in
both lists" needs no special rule — it is arithmetic, not a bonus. That is the
whole reason we chose RRF over the multiplicative boosts §3.5 originally had.
"""

from __future__ import annotations

from core.logger import get_logger
from agent.state import AgentState

log = get_logger("rerank")

# The standard RRF damping constant. Larger K flattens the curve, so the gap
# between 1st and 2nd matters less and being present in BOTH lists matters more.
RRF_K = 60

TOP_K = 15

# ── Bonus calibration (contract §3.5: "small relative to the RRF spread") ─────
#
# Work out the actual numbers before picking magnitudes, or "small" is a guess:
#
#   rank 1  in one list        1/(60+1)  = 0.016393
#   rank 2  in one list        1/(60+2)  = 0.016129   <- adjacent gap 0.00026
#   rank 20 in one list        1/(60+20) = 0.012500   <- whole-list spread 0.0039
#   rank 1  in BOTH lists      2/(60+1)  = 0.032787   <- double: overlap dominates
#
# So a bonus of 0.0003 moves a film about one position, and 0.0006 about two.
# Anything approaching 0.004 would reorder the entire list and make the fusion
# decorative. These are deliberately at the "nudge a neighbour" scale.
GENRE_BONUS_MAX = 0.0006      # ≈ 2 positions when every requested genre matches
QUALITY_BONUS_MAX = 0.0003    # ≈ 1 position for a 10/10 film over a 0/10 one


def _rrf_contribution(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def rerank(state: AgentState) -> dict:
    vector_rows = state.get("vector_results") or []
    graph_rows = state.get("graph_results") or []
    enrichment = state.get("enrichment") or {}
    requested_genres = set((state.get("filters") or {}).get("genres") or [])

    # ── 1. Fuse on rank ──────────────────────────────────────────────────────
    # Accumulate into ONE entry per tmdb_id (contract §3.5: do not keep the
    # higher of two rows — sum their contributions).
    fused: dict[int, dict] = {}
    for rows, source in ((vector_rows, "vector"), (graph_rows, "graph")):
        for row in rows:
            tmdb_id = row.get("tmdb_id")
            if tmdb_id is None:
                continue
            entry = fused.setdefault(
                tmdb_id,
                {**row, "rrf_score": 0.0, "sources": [], "ranks": {}},
            )
            # Keep whichever row is richer, but never lose fields already held.
            entry.update({k: v for k, v in row.items() if v not in (None, [], "")})
            entry["rrf_score"] += _rrf_contribution(row["rank"])
            entry["sources"].append(source)
            entry["ranks"][source] = row["rank"]

    # ── 2. Additive bonuses ──────────────────────────────────────────────────
    for entry in fused.values():
        links = enrichment.get(entry["tmdb_id"], {})
        genres = set(entry.get("genres") or links.get("genres") or [])

        genre_bonus = 0.0
        if requested_genres:
            matched = len(requested_genres & genres) / len(requested_genres)
            genre_bonus = matched * GENRE_BONUS_MAX

        rating = entry.get("rating") or 0.0
        quality_bonus = (rating / 10.0) * QUALITY_BONUS_MAX

        entry["genre_bonus"] = genre_bonus
        entry["quality_bonus"] = quality_bonus
        entry["final_score"] = entry["rrf_score"] + genre_bonus + quality_bonus
        entry["in_both"] = len(set(entry["sources"])) > 1

    ranked = sorted(fused.values(), key=lambda e: e["final_score"], reverse=True)[:TOP_K]

    both_count = sum(1 for e in ranked if e["in_both"])
    log.info(
        "reranked",
        candidates=len(fused),
        returned=len(ranked),
        in_both=both_count,
        # When every candidate is in both lists the overlap signal is a constant,
        # and constants cannot rank anything. Worth seeing in the logs.
        overlap_informative=0 < both_count < len(ranked),
        top=[(e["title"], round(e["final_score"], 5)) for e in ranked[:3]],
    )

    return {"reranked": ranked, "trace": [f"rerank({len(ranked)}/{len(fused)})"]}
