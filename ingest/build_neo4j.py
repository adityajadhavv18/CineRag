"""Build the Neo4j knowledge graph from movies.jsonl (contract §3.3).

    uv run python -m ingest.build_neo4j
    uv run python -m ingest.build_neo4j --reset   # wipe the graph first

Two Cypher ideas do all the work here:

  MERGE   "match this node, or create it if it doesn't exist" — idempotent, so
          re-running this script never duplicates anything. CREATE would.
  UNWIND  "take this list parameter and treat each element as a row" — lets us
          send 500 movies in ONE query instead of 500 round-trips.

The build runs as six passes (movies, genres, keywords, directors, cast,
collections) rather than one giant query. Slightly more round-trips, vastly more
readable — and when a pass fails you know exactly which relationship broke.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterator

from neo4j import GraphDatabase
from tqdm import tqdm

from core.config import settings
from core.logger import get_logger

log = get_logger("build_neo4j")

BATCH_SIZE = 500

# Constraints enforce uniqueness AND create the backing index that makes lookups
# fast — two birds. Person is keyed on person_id, never name (contract §3.3).
CONSTRAINTS = [
    "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
    "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
    "CREATE CONSTRAINT collection_id IF NOT EXISTS FOR (c:Collection) REQUIRE c.id IS UNIQUE",
    # Not unique — several distinct people legitimately share a name. This index
    # just makes "find the person called X" fast; it resolves to person_id nodes.
    "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
]

PASSES: list[tuple[str, str]] = [
    (
        "movies",
        """
        UNWIND $rows AS row
        MERGE (m:Movie {tmdb_id: row.tmdb_id})
        SET m.title         = row.title,
            m.year          = row.year,
            m.rating        = row.rating,
            m.popularity    = row.popularity,
            m.runtime       = row.runtime,
            m.release_date  = row.release_date,
            m.poster_path   = row.poster_path,
            m.backdrop_path = row.backdrop_path
        """,
    ),
    (
        "genres",
        """
        UNWIND $rows AS row
        MATCH (m:Movie {tmdb_id: row.tmdb_id})
        UNWIND row.genres AS genre_name
        MERGE (g:Genre {name: genre_name})
        MERGE (m)-[:HAS_GENRE]->(g)
        """,
    ),
    (
        "keywords",
        """
        UNWIND $rows AS row
        MATCH (m:Movie {tmdb_id: row.tmdb_id})
        UNWIND row.keywords AS kw
        MERGE (k:Keyword {name: kw})
        MERGE (m)-[:HAS_KEYWORD]->(k)
        """,
    ),
    (
        "directors",
        # MERGE on person_id, SET the name. Merging on name would collapse the
        # 48 distinct people in this catalogue who share a name with someone else.
        """
        UNWIND $rows AS row
        MATCH (m:Movie {tmdb_id: row.tmdb_id})
        UNWIND row.director AS d
        MERGE (p:Person {person_id: d.tmdb_person_id})
        SET p.name = d.name
        MERGE (p)-[:DIRECTED]->(m)
        """,
    ),
    (
        "cast",
        # The character name lives on the RELATIONSHIP, not the Person — the same
        # actor plays different characters in different films. This is the thing
        # a relational schema makes awkward and a graph makes obvious.
        """
        UNWIND $rows AS row
        MATCH (m:Movie {tmdb_id: row.tmdb_id})
        UNWIND row.cast AS c
        MERGE (p:Person {person_id: c.tmdb_person_id})
        SET p.name = c.name
        MERGE (p)-[r:ACTED_IN]->(m)
        SET r.character = c.character
        """,
    ),
    (
        "collections",
        """
        UNWIND $rows AS row
        WITH row WHERE row.collection_id IS NOT NULL
        MATCH (m:Movie {tmdb_id: row.tmdb_id})
        MERGE (c:Collection {id: row.collection_id})
        SET c.name = row.collection_name
        MERGE (m)-[:PART_OF]->(c)
        """,
    ),
]


def load_records() -> list[dict]:
    path = settings.movies_jsonl
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run: uv run python -m ingest.transform"
        )
    # Iterating the file handle splits on "\n" only, which is what we want.
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def batched(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def run(reset: bool) -> None:
    settings.require("neo4j_password")
    records = load_records()
    log.info("build_start", records=len(records), uri=settings.neo4j_uri, reset=reset)

    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        driver.verify_connectivity()

        if reset:
            # Batched delete: DETACH DELETE on the whole graph at once can blow
            # the transaction heap on a graph this size.
            log.info("resetting_graph")
            while True:
                summary = driver.execute_query(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
                )
                if summary.records[0]["deleted"] == 0:
                    break

        for stmt in CONSTRAINTS:
            driver.execute_query(stmt)
        log.info("constraints_ready", count=len(CONSTRAINTS))

        for name, cypher in PASSES:
            written = 0
            for batch in tqdm(
                list(batched(records, BATCH_SIZE)), desc=f"{name:<12}", unit="batch"
            ):
                driver.execute_query(cypher, rows=batch)
                written += len(batch)
            log.info("pass_complete", pass_name=name, rows=written)

        stats = driver.execute_query(
            """
            CALL (){ MATCH (m:Movie)      RETURN count(m) AS movies }
            CALL (){ MATCH (p:Person)     RETURN count(p) AS people }
            CALL (){ MATCH (g:Genre)      RETURN count(g) AS genres }
            CALL (){ MATCH (k:Keyword)    RETURN count(k) AS keywords }
            CALL (){ MATCH (c:Collection) RETURN count(c) AS collections }
            CALL (){ MATCH ()-[r:DIRECTED]->()  RETURN count(r) AS directed }
            CALL (){ MATCH ()-[r:ACTED_IN]->()  RETURN count(r) AS acted_in }
            CALL (){ MATCH ()-[r:HAS_GENRE]->() RETURN count(r) AS has_genre }
            CALL (){ MATCH ()-[r:HAS_KEYWORD]->() RETURN count(r) AS has_keyword }
            CALL (){ MATCH ()-[r:PART_OF]->()   RETURN count(r) AS part_of }
            RETURN movies, people, genres, keywords, collections,
                   directed, acted_in, has_genre, has_keyword, part_of
            """
        ).records[0]
        log.info("build_complete", **dict(stats))
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Neo4j movie graph")
    parser.add_argument("--reset", action="store_true", help="delete all nodes first")
    args = parser.parse_args()
    run(args.reset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
