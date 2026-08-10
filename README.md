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

## Dashboards

- Qdrant: <http://localhost:6333/dashboard>
- Neo4j Browser: <http://localhost:7474> (user `neo4j`, password from `.env`)
