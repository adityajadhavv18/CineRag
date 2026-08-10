"""Stage 2 of ingestion: data/raw/*.json -> data/movies.jsonl (contract §3.1, §3.1a).

Pure, local, and cheap — no network. Re-run this any time the canonical schema
changes; that is exactly what the raw cache exists for.

    uv run python -m ingest.transform              # append records not yet present
    uv run python -m ingest.transform --rebuild    # re-project everything from scratch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from core.config import settings
from core.logger import get_logger

log = get_logger("transform")

# Top N billed actors per movie. Enough to make "what else is this actor in?"
# work, few enough that the graph doesn't drown in one-line background parts.
CAST_LIMIT = 10


class CreditedPerson(BaseModel):
    """A person on a film. `tmdb_person_id` is the identity; `name` is display text.

    Names are NOT unique (several real people share one), so keying on name would
    merge distinct careers into a single graph node — see contract §3.3.
    """

    tmdb_person_id: int
    name: str
    character: str | None = None


class MovieRecord(BaseModel):
    """The canonical record (contract §3.1). Both stores are built from this."""

    tmdb_id: int
    title: str
    year: int | None
    overview: str
    tagline: str
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    director: list[CreditedPerson] = Field(default_factory=list)
    cast: list[CreditedPerson] = Field(default_factory=list)
    collection_id: int | None = None
    collection_name: str | None = None
    rating: float | None = None
    popularity: float | None = None
    runtime: int | None = None
    release_date: str | None = None
    poster_path: str | None = None
    backdrop_path: str | None = None


def project(raw: dict) -> MovieRecord | None:
    """Project one raw TMDB response into the canonical schema.

    Returns None for rows that can't support the build — a movie with no overview
    has nothing meaningful to embed, and a fabricated placeholder would violate
    the "real data only" guardrail (§5). Better to drop it and log why.
    """
    if not raw.get("id") or not raw.get("title"):
        return None
    overview = (raw.get("overview") or "").strip()
    if not overview:
        return None

    release_date = raw.get("release_date") or None
    year = int(release_date[:4]) if release_date and len(release_date) >= 4 else None

    credits = raw.get("credits") or {}

    # Directors come from the crew list, filtered by job. A film can legitimately
    # have several (the Coens, the Wachowskis, the Russos).
    directors = [
        CreditedPerson(tmdb_person_id=c["id"], name=c["name"])
        for c in credits.get("crew", [])
        if c.get("job") == "Director" and c.get("id") and c.get("name")
    ]

    cast = [
        CreditedPerson(
            tmdb_person_id=c["id"], name=c["name"], character=(c.get("character") or None)
        )
        for c in (credits.get("cast") or [])[:CAST_LIMIT]
        if c.get("id") and c.get("name")
    ]

    # TMDB nests keywords under a "keywords" key inside the appended block.
    kw_block = raw.get("keywords") or {}
    keywords = [k["name"] for k in kw_block.get("keywords", []) if k.get("name")]

    collection = raw.get("belongs_to_collection") or {}

    return MovieRecord(
        tmdb_id=raw["id"],
        title=raw["title"],
        year=year,
        overview=overview,
        tagline=(raw.get("tagline") or "").strip(),
        genres=[g["name"] for g in raw.get("genres", []) if g.get("name")],
        keywords=keywords,
        director=directors,
        cast=cast,
        collection_id=collection.get("id"),
        collection_name=collection.get("name"),
        rating=raw.get("vote_average"),
        popularity=raw.get("popularity"),
        runtime=raw.get("runtime"),
        release_date=release_date,
        poster_path=raw.get("poster_path"),
        backdrop_path=raw.get("backdrop_path"),
    )


def existing_ids(path: Path) -> set[int]:
    """tmdb_ids already written, so a re-run appends rather than duplicates."""
    if not path.exists():
        return set()
    ids: set[int] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["tmdb_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def run(rebuild: bool) -> None:
    raw_files = sorted(settings.raw_dir.glob("*.json"))
    if not raw_files:
        log.error("no_raw_data", raw_dir=str(settings.raw_dir),
                  hint="run: uv run python -m ingest.tmdb_pull")
        return

    out = settings.movies_jsonl
    out.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and out.exists():
        out.unlink()

    already = existing_ids(out)
    log.info("transform_start", raw_files=len(raw_files), already_written=len(already),
             mode="rebuild" if rebuild else "append")

    written = skipped_existing = skipped_invalid = 0
    with out.open("a", encoding="utf-8") as fh:
        for path in tqdm(raw_files, desc="transforming", unit="movie"):
            try:
                raw = json.loads(path.read_text())
            except json.JSONDecodeError:
                log.warning("raw_unreadable", file=path.name)
                skipped_invalid += 1
                continue

            if raw.get("id") in already:
                skipped_existing += 1
                continue

            record = project(raw)
            if record is None:
                skipped_invalid += 1
                continue

            # ensure_ascii=True escapes every non-ASCII character, including the
            # exotic line breaks (U+2028 LINE SEPARATOR, U+0085 NEL, …) that TMDB
            # overviews occasionally contain. Those are legal inside a JSON string
            # but Python's str.splitlines() treats them as line breaks, so a
            # JSONL file containing one raw silently splits into a broken record
            # for any consumer that uses splitlines(). Escaping removes the trap.
            fh.write(json.dumps(record.model_dump(), ensure_ascii=True) + "\n")
            written += 1

    log.info("transform_complete", written=written, skipped_existing=skipped_existing,
             skipped_invalid=skipped_invalid, total_records=len(already) + written,
             output=str(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Project data/raw/*.json into movies.jsonl")
    parser.add_argument("--rebuild", action="store_true",
                        help="delete movies.jsonl and re-project every cached movie")
    args = parser.parse_args()
    run(args.rebuild)
    return 0


if __name__ == "__main__":
    sys.exit(main())
