"""Tests for the canonical projection (contract §3.1).

These run against REAL cached TMDB responses in data/raw/ rather than invented
fixtures — §5 forbids synthetic movie facts, and a hand-written fixture would
also drift from TMDB's actual response shape, which is the thing worth testing.

They skip cleanly if the raw cache hasn't been pulled yet.
"""

from __future__ import annotations

import json

import pytest

from core.config import settings
from ingest.transform import CAST_LIMIT, project

RAW_FILES = sorted(settings.raw_dir.glob("*.json"))[:200] if settings.raw_dir.exists() else []

pytestmark = pytest.mark.skipif(
    not RAW_FILES, reason="no raw cache — run: uv run python -m ingest.tmdb_pull --limit 50"
)


@pytest.fixture(scope="module")
def records():
    out = []
    for path in RAW_FILES:
        rec = project(json.loads(path.read_text()))
        if rec is not None:
            out.append(rec)
    assert out, "projection produced no records from a non-empty raw cache"
    return out


def test_every_credited_person_has_a_stable_id(records):
    """The decision from contract §3.3: identity is tmdb_person_id, never name.

    If this ever fails, the Neo4j build will merge distinct people who share a
    name into one node and silently corrupt every filmography query.
    """
    for rec in records:
        for person in rec.director + rec.cast:
            assert person.tmdb_person_id > 0, f"{rec.title}: {person.name} has no tmdb_person_id"


def test_person_ids_are_consistent_across_movies(records):
    """One person_id must always map to one name — otherwise the ID isn't stable."""
    name_by_id: dict[int, str] = {}
    for rec in records:
        for person in rec.director + rec.cast:
            existing = name_by_id.setdefault(person.tmdb_person_id, person.name)
            assert existing == person.name, (
                f"person_id {person.tmdb_person_id} seen as both "
                f"{existing!r} and {person.name!r}"
            )


def test_embedding_fields_are_present(records):
    """Every record must have what the Day-2 embed template needs (§3.2)."""
    for rec in records:
        assert rec.title, f"{rec.tmdb_id}: missing title"
        assert rec.overview.strip(), f"{rec.title}: empty overview — should have been dropped"


def test_records_without_an_overview_are_dropped():
    """A movie with nothing to embed must be skipped, not padded with filler (§5)."""
    raw = json.loads(RAW_FILES[0].read_text())
    assert project({**raw, "overview": ""}) is None
    assert project({**raw, "overview": "   "}) is None


def test_cast_is_capped(records):
    for rec in records:
        assert len(rec.cast) <= CAST_LIMIT


def test_images_are_paths_not_urls(records):
    """Contract §5: store poster/backdrop paths only; the frontend builds URLs."""
    for rec in records:
        for path in (rec.poster_path, rec.backdrop_path):
            if path is not None:
                assert path.startswith("/"), f"{rec.title}: {path!r} is not a bare TMDB path"
                assert "http" not in path, f"{rec.title}: stored a URL instead of a path"


def test_jsonl_is_free_of_exotic_line_breaks():
    """movies.jsonl must have exactly one record per newline, for every reader.

    TMDB overviews occasionally contain U+2028 LINE SEPARATOR (and friends).
    Those are legal inside a JSON string, but Python's str.splitlines() treats
    them as line breaks — so a file containing one raw parses fine with
    `for line in fh` and blows up with `.splitlines()`. We escape all non-ASCII
    on write; this asserts the property that guarantees, rather than the
    implementation that provides it.
    """
    path = settings.movies_jsonl
    if not path.exists():
        pytest.skip("movies.jsonl not built yet")
    text = path.read_text()
    assert text.count("\n") == len(text.splitlines()), (
        "movies.jsonl contains a line-break character that splitlines() honours "
        "but '\\n' does not — records will silently fragment"
    )
    for line in text.splitlines():
        if line.strip():
            json.loads(line)  # every line must independently parse


def test_collection_fields_agree(records):
    """collection_id and collection_name are both-or-neither — the franchise chain
    (§3.3 PART_OF) depends on the pair being coherent."""
    for rec in records:
        assert (rec.collection_id is None) == (rec.collection_name is None), (
            f"{rec.title}: half-populated collection "
            f"({rec.collection_id}, {rec.collection_name})"
        )
