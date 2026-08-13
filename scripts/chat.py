"""Drive the agent from the terminal.

    uv run python -m scripts.chat                      # interactive
    uv run python -m scripts.chat "movies like Heat"   # one shot
    uv run python -m scripts.chat --demo               # the Day 3 routing suite
    uv run python -m scripts.chat --draw               # print the graph structure
"""

from __future__ import annotations

import argparse
import sys

from agent.graph import get_agent, run
from core.config import configure_langsmith, settings

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"

DEMO_QUERIES = [
    ("hi", "general"),
    ("what can you do?", "general"),
    ("what's the weather in Paris", "off_topic"),
    ("write me a python script to sort a list", "off_topic"),
    ("something tense and claustrophobic", "recommend"),
    ("who directed Whiplash", "factual_lookup"),
    ("gritty crime dramas starring Denzel Washington", "recommend"),
    ("recommend some movies", "clarification"),
]


def show(state: dict) -> None:
    print(f"  {DIM}trace: {' -> '.join(state.get('trace', []))}{RESET}")
    print(f"\n{state.get('response', '(no response)')}\n")


def demo() -> None:
    """Assert routing on a fixed set — the Day 3 acceptance check."""
    print(f"\n{BOLD}Routing demo — intent classification and conditional edges{RESET}\n")
    failures = 0
    for query, expected in DEMO_QUERIES:
        state = run(query)
        actual = state.get("intent")
        ok = actual == expected
        failures += not ok
        mark = f"\033[32m✓\033[0m" if ok else f"\033[31m✗\033[0m"
        lead = state.get("lead_engine")
        print(f"  {mark} {query[:46]:<46} intent={actual:<15} lead={lead}")
        if not ok:
            print(f"      {DIM}expected intent={expected}{RESET}")

    print(f"\n  {len(DEMO_QUERIES) - failures}/{len(DEMO_QUERIES)} routed as expected\n")


def draw() -> None:
    agent = get_agent()
    print()
    try:
        print(agent.get_graph().draw_ascii())
    except Exception:  # noqa: BLE001 — draw_ascii needs an optional dependency
        g = agent.get_graph()
        print(f"{BOLD}nodes:{RESET} {', '.join(n for n in g.nodes)}")
        print(f"{BOLD}edges:{RESET}")
        for e in g.edges:
            label = f"  [{e.data}]" if getattr(e, "data", None) else ""
            print(f"   {e.source:>12} -> {e.target}{label}")
    print()


def interactive() -> None:
    print(f"\n{BOLD}CineRAG{RESET} {DIM}(ctrl-c or 'exit' to quit){RESET}")
    tracing = configure_langsmith()
    print(f"{DIM}tracing: {'on -> ' + settings.langsmith_project if tracing else 'off'}{RESET}\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            query = input(f"{CYAN}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return

        state = run(query, history)
        show(state)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": state.get("response", "")})


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to the CineRAG agent")
    parser.add_argument("query", nargs="*", help="one-shot query")
    parser.add_argument("--demo", action="store_true", help="run the routing suite")
    parser.add_argument("--draw", action="store_true", help="print the graph structure")
    args = parser.parse_args()

    if args.draw:
        draw()
    elif args.demo:
        demo()
    elif args.query:
        show(run(" ".join(args.query)))
    else:
        interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
