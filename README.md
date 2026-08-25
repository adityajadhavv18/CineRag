# CineRAG

Intent-driven movie recommendation agent over a Qdrant vector store + Neo4j knowledge graph,
grounded on real TMDB data. See [PROJECT_CONTRACT.md](PROJECT_CONTRACT.md) — that document is the
single source of truth for the build; this file is just the runbook.

## Setup

```bash
cp .env.example .env      # then fill in TMDB_API_KEY (and OPENAI_API_KEY from Day 2)
uv sync                   # uv provisions Python 3.11+ itself
docker compose up -d      # Qdrant :6333, Neo4j :7474 (browser) / :7687 (bolt)
uv run python -m scripts.doctor
```

`doctor` reports what's configured, what's reachable, and how much data exists. Run it whenever
something feels off.

## Ingestion (two stages — contract §3.1a)

```bash
uv run python -m ingest.tmdb_pull            # stage 1: TMDB -> data/raw/{id}.json
uv run python -m ingest.tmdb_pull --limit 50 # smoke test with 50 movies
uv run python -m ingest.transform            # stage 2: data/raw/*.json -> data/movies.jsonl
uv run python -m ingest.transform --rebuild  # re-project everything after a schema change
```

The pull is resumable — `data/raw/` *is* the checkpoint, so killing it mid-run and restarting
picks up where it stopped. The transform is local and cheap; re-run it freely.

## Building the stores (contract §3.2, §3.3)

```bash
uv run python -m ingest.build_neo4j            # graph: ~20s, no API cost
uv run python -m ingest.build_neo4j --reset    # wipe and rebuild
uv run python -m ingest.build_qdrant           # embed + index: ~60s, ~$0.013
uv run python -m ingest.build_qdrant --recreate
uv run python -m ingest.build_qdrant --limit 200   # cheap partial run
```

Both are idempotent: Neo4j via `MERGE`, Qdrant via `tmdb_id` as the point id.

## Seeing the two stores differ

```bash
uv run python -m scripts.compare
```

Puts a vibe query and a fact query to *both* stores. Two answers are good, two
are bad — and why they're bad is the argument for the architecture.

## Talking to the agent (Day 3+)

```bash
uv run python -m scripts.chat                    # interactive
uv run python -m scripts.chat "who directed Heat"
uv run python -m scripts.chat --demo             # routing suite
uv run python -m scripts.chat --draw             # graph structure
```

Tracing turns on when `LANGSMITH_API_KEY` is set **and** `LANGCHAIN_TRACING_V2=true`.

## Tests

```bash
uv run pytest -q     # store tests skip cleanly when Docker is down
```

## Dashboards

- Qdrant: <http://localhost:6333/dashboard>
- Neo4j Browser: <http://localhost:7474> (user `neo4j`, password from `.env`)

## Serving it (Day 7)

```bash
uv run uvicorn server.main:app --reload      # http://127.0.0.1:8000
open http://127.0.0.1:8000/docs              # interactive OpenAPI
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"gritty crime dramas starring Denzel Washington"}' | python3 -m json.tool

curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool
```

The response carries `sources[]` with `poster_path`, `backdrop_path` and `overview`, so a card
renders with image + blurb from one call. `degraded: true` means a store was unreachable — an
empty `sources` list means something different depending on that flag.

Multi-turn is client-driven (the server is stateless, contract §1): send prior turns back as
`history`.

## Evaluation

```bash
uv run python -m eval.dataset          # what the dataset covers
uv run python -m eval.run_eval         # run it, exits non-zero on failure
uv run python -m eval.run_eval --filter "Denzel"
uv run python -m eval.run_eval --upload    # push examples to LangSmith
```

28 labelled cases, 8 deterministic criteria — intent, routing, grounding, expected titles,
retrieved references, no-citations, asks-a-question, plot-grounding. No LLM judges: an
LLM-graded eval drifts with the judge and cannot gate a regression.
