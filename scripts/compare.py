"""Day 2: run the same query against both stores and watch each one fail.

    uv run python -m scripts.compare

This is the exercise the whole two-store architecture rests on. A vibe query
("heist movie with a twist") and a fact query ("movies directed by Christopher
Nolan") are each put to Qdrant AND Neo4j. Two of the four answers are good and
two are bad, and *why* they're bad is the point:

  - Qdrant can't do exact facts. It matches on meaning, so a filmography query
    returns films that FEEL like Nolan rather than films Nolan directed.
  - Neo4j can't do vibes. It matches exact tokens, so a mood query only works if
    someone happened to tag the movie with that exact keyword.

Neither is a defect. Each is the shape of what that store knows.
"""

from __future__ import annotations

import sys

from stores import neo4j_client
from stores import qdrant_client

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"


def naive_keyword_search(query: str, limit: int = 8) -> list[dict]:
    """The best the GRAPH can do with a vibe query: exact keyword/genre tokens.

    Deliberately naive — this is a demonstration of the graph's limit, not a
    retrieval strategy we ship. Day 4's graph_retrieve works from entities the
    intent node extracts, not from raw query tokens.
    """
    tokens = [t.strip(".,!?'\"").lower() for t in query.split() if len(t) > 3]
    return neo4j_client._query(
        """
        MATCH (m:Movie)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.name IN $tokens
        WITH m, count(DISTINCT k) AS matched
        RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.year AS year,
               m.rating AS rating, matched
        ORDER BY matched DESC, m.popularity DESC
        LIMIT $limit
        """,
        tokens=tokens,
        limit=limit,
    )


def show(rows: list[dict], *, score_key: str | None = None, extra: str | None = None) -> None:
    if not rows:
        print(f"    {RED}(nothing){RESET}")
        return
    for i, r in enumerate(rows, 1):
        bits = f"{r.get('year') or '????'}"
        if score_key and r.get(score_key) is not None:
            bits += f"  {score_key}={r[score_key]:.3f}"
        if extra and r.get(extra):
            bits += f"  {extra}={r[extra]}"
        print(f"    {i:>2}. {r['title'][:42]:<42} {DIM}{bits}{RESET}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'═' * 78}{RESET}\n{BOLD}{title}{RESET}\n{BOLD}{'═' * 78}{RESET}")


def main() -> int:
    vibe = "heist movie with a twist"
    fact = "Christopher Nolan"

    section(f'VIBE QUERY:  "{vibe}"')

    print(f"\n  {GREEN}QDRANT (dense semantic){RESET} — should nail this")
    show(qdrant_client.search(vibe, limit=8), score_key="score")

    print(f"\n  {RED}NEO4J (exact tokens){RESET} — should struggle")
    print(f"    {DIM}best it can do: match query words against Keyword nodes{RESET}")
    show(naive_keyword_search(vibe), extra="matched")

    section(f'FACT QUERY:  "movies directed by {fact}"')

    print(f"\n  {GREEN}NEO4J (graph traversal){RESET} — should nail this")
    show(neo4j_client.movies_by_director(fact, limit=12))

    print(f"\n  {RED}QDRANT (dense semantic){RESET} — should struggle")
    print(f"    {DIM}searching the phrase as MEANING, not as a fact{RESET}")
    rows = qdrant_client.search(f"movies directed by {fact}", limit=8)
    show(rows, score_key="score")
    truth = {m["title"] for m in neo4j_client.movies_by_director(fact, limit=50)}
    hits = [r["title"] for r in rows if r["title"] in truth]
    print(f"\n    {BOLD}actually directed by {fact}: {len(hits)}/{len(rows)}{RESET}"
          f"  {DIM}{hits}{RESET}")

    section("THE FRANCHISE TIMELINE  (one hop through a Collection node)")
    seed = neo4j_client._query(
        """
        MATCH (m:Movie)-[:PART_OF]->(c:Collection)<-[:PART_OF]-(other:Movie)
        WITH m, c, count(other) AS siblings
        WHERE siblings >= 3
        RETURN m.tmdb_id AS tmdb_id, m.title AS title, c.name AS collection
        ORDER BY m.popularity DESC LIMIT 1
        """
    )
    if seed:
        s = seed[0]
        print(f"\n  seed movie: {BOLD}{s['title']}{RESET}  →  collection: {s['collection']}")
        show(neo4j_client.franchise_timeline(s["tmdb_id"]))
        print(f"\n    {DIM}Qdrant cannot produce this at all: 'ordered members of the same"
              f" franchise'\n    is a traversal, not a similarity.{RESET}")

    section("WHY PERSON IDENTITY IS person_id, NOT name  (contract §3.3)")
    dupes = neo4j_client.people_who_share_a_name(limit=6)
    print()
    for d in dupes:
        print(f"    {d['name']:<26} → {d['person_count']} different people: {d['ids']}")
    print(f"\n    {DIM}Under name-keying each row above would be ONE node fusing"
          f" two careers.{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
