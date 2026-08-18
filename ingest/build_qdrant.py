"""Embed movies.jsonl and upsert into Qdrant (contract §3.2).

    uv run python -m ingest.build_qdrant
    uv run python -m ingest.build_qdrant --limit 200   # cheap partial run
    uv run python -m ingest.build_qdrant --recreate    # drop the collection first

Dense semantic search only — one vector per movie, no sparse/BM25 component.
The knowledge graph is our exact-token precision layer (contract §3.2).
"""

from __future__ import annotations

import argparse
import json
import sys

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from core.config import settings
from core.llm import EMBEDDING_DIM, embed_texts
from core.logger import get_logger

log = get_logger("build_qdrant")

UPSERT_BATCH_SIZE = 256


def compose_embed_text(movie: dict) -> str:
    """The FROZEN embed-text template (contract §3.2).

        {title} ({year})
        {tagline}
        {overview}
        Genres: {comma-separated genres}
        Keywords: {comma-separated keywords}

    Field order, labels and separators are part of the spec: they change the
    resulting vector, so changing this function invalidates every stored point
    and requires a full re-embed. Treat it as a versioned decision, not a tweak.

    Missing optional fields collapse to an empty line rather than the literal
    string "None" — embedding the word "None" would add noise that pulls
    unrelated movies together.
    """
    year = f" ({movie['year']})" if movie.get("year") else ""
    genres = ", ".join(movie.get("genres") or [])
    keywords = ", ".join(movie.get("keywords") or [])
    return (
        f"{movie['title']}{year}\n"
        f"{movie.get('tagline') or ''}\n"
        f"{movie.get('overview') or ''}\n"
        f"Genres: {genres}\n"
        f"Keywords: {keywords}"
    )


def build_payload(movie: dict) -> dict:
    """What Qdrant stores alongside the vector (contract §3.2).

    This is NOT the same as the embed text. The payload exists to filter on and
    to return — which is why director/cast/poster_path live here even though they
    are not embedded: a vibe-led query still needs to filter to a named actor and
    render a card without a second round-trip to Neo4j.
    """
    return {
        "tmdb_id": movie["tmdb_id"],
        "title": movie["title"],
        "year": movie.get("year"),
        # The plot text itself, not just its embedding. We embed the overview into
        # the vector, but a vector cannot be read back — so without keeping the
        # words, nothing downstream can DESCRIBE a film, and an LLM asked to would
        # supply the plot from memory. Storing it is what keeps a description
        # grounded (contract §5).
        "overview": movie.get("overview") or "",
        "tagline": movie.get("tagline") or "",
        "genres": movie.get("genres") or [],
        "director": [d["name"] for d in movie.get("director") or []],
        "cast_names": [c["name"] for c in movie.get("cast") or []],
        "rating": movie.get("rating"),
        "popularity": movie.get("popularity"),
        "runtime": movie.get("runtime"),
        "collection_id": movie.get("collection_id"),
        "poster_path": movie.get("poster_path"),
        "backdrop_path": movie.get("backdrop_path"),
    }


def load_records(limit: int | None) -> list[dict]:
    path = settings.movies_jsonl
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run: uv run python -m ingest.transform")
    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return records[:limit] if limit else records


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    name = settings.qdrant_collection
    exists = client.collection_exists(name)

    if exists and recreate:
        log.info("dropping_collection", collection=name)
        client.delete_collection(name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
        )
        log.info("collection_created", collection=name, dim=EMBEDDING_DIM, distance="cosine")

    # Payload indexes make filtered search fast (contract §3.2). Without them
    # Qdrant still filters correctly, just by scanning — fine at 5k points,
    # not fine later, and free to declare now.
    for field, schema in [
        ("genres", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.INTEGER),
        ("rating", models.PayloadSchemaType.FLOAT),
    ]:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema, wait=True
        )
    log.info("payload_indexes_ready", fields=["genres", "year", "rating"])


def run_payload_only(records: list[dict]) -> None:
    """Refresh payloads WITHOUT re-embedding (contract §3.2).

    A payload is metadata stored beside the vector; changing it does not change
    the vector. So adding a field like `overview` is a metadata update, not a
    re-index — no OpenAI calls, no cost, seconds instead of minutes.

    Worth having as its own path rather than just re-running the full build: at
    5k films re-embedding is 60s and about a cent, but the instinct "changing
    stored text means re-embedding everything" is wrong and gets expensive fast.
    """
    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(settings.qdrant_collection):
        raise RuntimeError(
            f"collection {settings.qdrant_collection!r} does not exist — "
            "run without --payload-only first"
        )

    log.info("payload_update_start", records=len(records), embedding_calls=0)

    updated = 0
    for start in tqdm(range(0, len(records), UPSERT_BATCH_SIZE), desc="payloads", unit="batch"):
        chunk = records[start : start + UPSERT_BATCH_SIZE]
        # One SetPayload op per point (each has a different payload), many ops
        # per request.
        client.batch_update_points(
            collection_name=settings.qdrant_collection,
            update_operations=[
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=build_payload(m), points=[m["tmdb_id"]]
                    )
                )
                for m in chunk
            ],
            wait=True,
        )
        updated += len(chunk)

    info = client.get_collection(settings.qdrant_collection)
    log.info("payload_update_complete", updated=updated, points=info.points_count)


def run(limit: int | None, recreate: bool) -> None:
    settings.require("openai_api_key")
    records = load_records(limit)
    log.info("build_start", records=len(records), collection=settings.qdrant_collection,
             model=settings.embedding_model)

    client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(client, recreate)

    texts = [compose_embed_text(m) for m in records]
    log.info("embed_start", count=len(texts),
             avg_chars=round(sum(len(t) for t in texts) / max(len(texts), 1)))

    vectors: list[list[float]] = []
    batch = 128
    for start in tqdm(range(0, len(texts), batch), desc="embedding", unit="batch"):
        vectors.extend(embed_texts(texts[start : start + batch], batch_size=batch))
    log.info("embed_complete", vectors=len(vectors), dim=len(vectors[0]) if vectors else 0)

    # Point id = tmdb_id, so re-running overwrites rather than duplicating.
    points = [
        models.PointStruct(id=m["tmdb_id"], vector=v, payload=build_payload(m))
        for m, v in zip(records, vectors)
    ]
    for start in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="upserting", unit="batch"):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[start : start + UPSERT_BATCH_SIZE],
            wait=True,
        )

    info = client.get_collection(settings.qdrant_collection)
    log.info("build_complete", points=info.points_count, collection=settings.qdrant_collection)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed movies into Qdrant")
    parser.add_argument("--limit", type=int, default=None, help="only index the first N movies")
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")
    parser.add_argument(
        "--payload-only", action="store_true",
        help="refresh payloads without re-embedding (no OpenAI calls)",
    )
    args = parser.parse_args()
    if args.payload_only:
        run_payload_only(load_records(args.limit))
    else:
        run(args.limit, args.recreate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
