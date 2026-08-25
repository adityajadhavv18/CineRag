"""Run the eval and report pass/fail per criterion (contract §4, Day 7).

    uv run python -m eval.run_eval              # local run, prints a report
    uv run python -m eval.run_eval --upload     # also push results to LangSmith

Every evaluator here is DETERMINISTIC — a set membership test, a string check, a
comparison against a label. None of them ask an LLM whether the answer was good.

That is the point. An LLM-judged eval drifts with the judge and cannot be trusted
to gate a regression, and "was this a good recommendation?" is a matter of taste.
What IS checkable: did the right branch fire, did the right store lead, was every
cited film real, did we invent a plot, did we admit we found nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

from agent.graph import run as run_agent
from core.config import configure_langsmith, settings
from core.logger import get_logger
from eval.dataset import CASES, UNGROUNDED_PLOT_TELLS, Case

log = get_logger("eval")

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


# ── evaluators ───────────────────────────────────────────────────────────────


def check_intent(case: Case, state: dict) -> Check:
    actual = state.get("intent")
    return Check("intent", actual == case.intent, f"got {actual}, want {case.intent}")


def check_lead_engine(case: Case, state: dict) -> Check | None:
    if case.lead_engine is None:
        return None
    actual = state.get("lead_engine")
    return Check("routing", actual == case.lead_engine,
                 f"got {actual}, want {case.lead_engine}")


def check_grounding(case: Case, state: dict) -> Check:
    """THE regression this eval exists for (contract §5).

    Every cited film must be present in the retrieved set. If this ever fails,
    the agent is naming films that were never retrieved — i.e. remembering them.
    """
    retrieved = {r.get("tmdb_id") for r in (state.get("reranked") or [])}
    cited = {c.get("tmdb_id") for c in (state.get("citations") or [])}
    invented = cited - retrieved
    return Check("grounded", not invented, f"cited but never retrieved: {invented}")


def check_must_include(case: Case, state: dict) -> Check | None:
    if not case.must_include:
        return None
    titles = {c["title"] for c in (state.get("citations") or [])}
    missing = [t for t in case.must_include if t not in titles]
    return Check("expected_titles", not missing, f"missing {missing}")


def check_must_retrieve(case: Case, state: dict) -> Check | None:
    """A reference film must reach the candidate set even if it is not recommended."""
    if not case.must_retrieve:
        return None
    titles = {r.get("title") for r in (state.get("reranked") or [])}
    missing = [t for t in case.must_retrieve if t not in titles]
    return Check("retrieved_reference", not missing, f"never retrieved {missing}")


def check_must_not_cite(case: Case, state: dict) -> Check | None:
    if not case.must_not_cite:
        return None
    cited = [c["title"] for c in (state.get("citations") or [])]
    return Check("no_citations", not cited, f"cited {cited}")


def check_asks_a_question(case: Case, state: dict) -> Check | None:
    if not case.expect_question:
        return None
    response = state.get("response") or ""
    return Check("asks_question", "?" in response, "response contains no question mark")


def check_plot_grounding(case: Case, state: dict) -> Check | None:
    """Plot claims must paraphrase the stored overview, not model memory.

    Nothing else catches this: citation validation checks WHICH films are named,
    never WHAT is claimed about them. A fluent, wrong synopsis passes every other
    check in this file.
    """
    if not case.grounded_plot:
        return None
    response = (state.get("response") or "").lower()
    tells = [t for title in case.must_include
             for t in UNGROUNDED_PLOT_TELLS.get(title, [])]
    found = [t for t in tells if t in response]
    return Check("plot_grounded", not found, f"details absent from the stored overview: {found}")


EVALUATORS = [
    check_intent,
    check_lead_engine,
    check_grounding,
    check_must_include,
    check_must_retrieve,
    check_must_not_cite,
    check_asks_a_question,
    check_plot_grounding,
]


# ── runner ───────────────────────────────────────────────────────────────────


def run_case(case: Case) -> tuple[list[Check], dict, float]:
    started = time.perf_counter()
    try:
        state = run_agent(case.query, case.history)
    except Exception as exc:  # noqa: BLE001 — a crash is a failed case, not a crashed eval
        log.error("case_crashed", query=case.query, error=type(exc).__name__)
        return [Check("no_crash", False, f"{type(exc).__name__}: {exc}")], {}, 0.0

    checks = [c for c in (ev(case, state) for ev in EVALUATORS) if c is not None]
    return checks, state, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CineRAG eval")
    parser.add_argument("--upload", action="store_true", help="push results to LangSmith")
    parser.add_argument("--filter", help="only run cases whose query contains this")
    args = parser.parse_args()

    configure_langsmith()
    cases = [c for c in CASES if not args.filter or args.filter.lower() in c.query.lower()]

    print(f"\n{BOLD}CineRAG eval — {len(cases)} cases{RESET}\n")

    results = []
    by_criterion: dict[str, list[bool]] = {}

    for case in cases:
        checks, state, elapsed = run_case(case)
        passed = all(c.passed for c in checks)
        results.append((case, checks, state, passed))

        for check in checks:
            by_criterion.setdefault(check.name, []).append(check.passed)

        mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {mark}  {case.query[:52]:<54} {DIM}{elapsed:.1f}s{RESET}")
        for check in checks:
            if not check.passed:
                print(f"        {RED}{check.name}{RESET}: {check.detail}")
        if not passed and case.note:
            print(f"        {DIM}note: {case.note}{RESET}")

    # ── report ───────────────────────────────────────────────────────────────
    total = len(results)
    passed_total = sum(1 for *_, p in results if p)

    print(f"\n{BOLD}By criterion{RESET}")
    for name, outcomes in sorted(by_criterion.items()):
        ok = sum(outcomes)
        colour = GREEN if ok == len(outcomes) else (YELLOW if ok else RED)
        print(f"  {name:<18} {colour}{ok}/{len(outcomes)}{RESET}")

    colour = GREEN if passed_total == total else RED
    print(f"\n{BOLD}Overall{RESET}  {colour}{passed_total}/{total} cases passed{RESET}\n")

    log.info("eval_complete", cases=total, passed=passed_total,
             criteria={k: f"{sum(v)}/{len(v)}" for k, v in by_criterion.items()})

    if args.upload:
        upload(results)

    # Non-zero exit so this can gate CI.
    return 0 if passed_total == total else 1


def upload(results) -> None:
    """Push the run to LangSmith as a dataset + feedback, for tracking over time."""
    if not (settings.langsmith_api_key and settings.langchain_tracing_v2):
        print(f"{YELLOW}skipping upload: LANGSMITH_API_KEY / LANGCHAIN_TRACING_V2 not set{RESET}")
        return

    from langsmith import Client

    client = Client()
    name = f"{settings.langsmith_project}-eval"

    try:
        dataset = next(
            (d for d in client.list_datasets(dataset_name=name)), None
        ) or client.create_dataset(dataset_name=name, description="CineRAG Day 7 eval")

        existing = {e.inputs.get("query") for e in client.list_examples(dataset_id=dataset.id)}
        added = 0
        for case, checks, state, passed in results:
            if case.query in existing:
                continue
            client.create_example(
                dataset_id=dataset.id,
                inputs={"query": case.query, "history": case.history},
                outputs={"intent": case.intent, "lead_engine": case.lead_engine},
                metadata={"note": case.note},
            )
            added += 1
        print(f"{GREEN}uploaded{RESET} dataset {name!r}: {added} new example(s), "
              f"{len(existing)} already present")
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}upload failed{RESET}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
