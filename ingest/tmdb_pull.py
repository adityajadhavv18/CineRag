"""Stage 1 of ingestion: TMDB -> data/raw/{tmdb_id}.json (contract §3.1a).

This script's only job is to get raw TMDB responses onto local disk, exactly
once. It does NOT project into the canonical schema — that's ingest/transform.py.

Why the split: this stage costs ~5,000 rate-limited network round-trips; the
transform costs seconds. Keeping the raw JSON means a later schema fix (a field
we forgot to capture) is a local re-transform, not another full crawl.

Resumability: `data/raw/` IS the checkpoint. Any movie whose file already exists
is skipped, so killing this mid-run and restarting resumes where it stopped.

    uv run python -m ingest.tmdb_pull
    uv run python -m ingest.tmdb_pull --limit 50     # small smoke-test run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx
from tqdm import tqdm

from core.config import MissingConfigError, settings
from core.logger import get_logger

log = get_logger("tmdb_pull")

BASE_URL = "https://api.themoviedb.org/3"

# TMDB's discovery lists cap out at 500 pages of 20 results each.
PAGE_SIZE = 20
MAX_PAGES = 500

# Blend ratio across the two discovery endpoints. `popular` skews recent and
# mainstream; `top_rated` skews acclaimed and older. Blending gives the catalogue
# both "what people search for" and "what's actually good" — a catalogue of pure
# `popular` would answer "gritty crime dramas" with this month's releases.
POPULAR_SHARE = 0.6

# Concurrency: TMDB tolerates ~50 req/s. 8 in flight is comfortably under that
# and turns a ~17-minute sequential crawl into ~3 minutes.
CONCURRENCY = 8

# Discovered IDs are cached too, so a resumed run doesn't re-paginate the lists.
DISCOVERED_IDS_PATH = settings.data_dir / "discovered_ids.json"


def _auth(client_kwargs: dict) -> dict:
    """TMDB accepts either a v4 read-access token (a JWT, sent as a Bearer header)
    or a v3 API key (sent as a query param). Prefer the v4 token when present."""
    token = settings.tmdb_api_read_access_token
    if token:
        client_kwargs["headers"] = {"Authorization": f"Bearer {token}"}
        log.info("tmdb_auth", method="v4_bearer_token")
    else:
        client_kwargs["params"] = {"api_key": settings.tmdb_api_key}
        log.info("tmdb_auth", method="v3_api_key")
    return client_kwargs


async def _get(client: httpx.AsyncClient, path: str, **params) -> dict | None:
    """GET with backoff. Returns None if the resource is permanently unavailable.

    429 (rate limited) and 5xx (transient server trouble) are retried; TMDB sends
    a Retry-After header on 429 that we honour rather than guessing. A 404 means
    the movie genuinely isn't there — retrying can't help, so we give up on it.
    """
    delay = 1.0
    for attempt in range(6):
        try:
            r = await client.get(path, params=params)
        except httpx.RequestError as exc:
            log.warning("request_error", path=path, error=type(exc).__name__, attempt=attempt)
            await asyncio.sleep(delay)
            delay *= 2
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", delay))
            log.warning("rate_limited", path=path, retry_after=wait)
            await asyncio.sleep(wait)
            continue
        if r.status_code in (404, 401, 403):
            log.warning("request_rejected", path=path, status=r.status_code)
            return None
        if r.status_code >= 500:
            await asyncio.sleep(delay)
            delay *= 2
            continue
        log.warning("unexpected_status", path=path, status=r.status_code)
        return None

    log.error("gave_up", path=path)
    return None


async def discover_ids(client: httpx.AsyncClient, target: int) -> list[int]:
    """Collect `target` unique movie IDs by blending /movie/popular + /movie/top_rated.

    The two lists overlap heavily (acclaimed blockbusters appear in both), so we
    dedupe as we go and keep paginating until we actually have `target` unique
    IDs — asking for N pages would leave us short.
    """
    if DISCOVERED_IDS_PATH.exists():
        cached = json.loads(DISCOVERED_IDS_PATH.read_text())
        if len(cached) >= target:
            log.info("discovery_cache_hit", count=len(cached), path=str(DISCOVERED_IDS_PATH))
            return cached[:target]

    seen: dict[int, None] = {}  # dict preserves insertion order; acts as ordered set
    endpoints = [("/movie/popular", POPULAR_SHARE), ("/movie/top_rated", 1 - POPULAR_SHARE)]
    exhausted: set[str] = set()
    page = 1

    with tqdm(total=target, desc="discovering ids", unit="id") as bar:
        while len(seen) < target and page <= MAX_PAGES:
            for path, share in endpoints:
                if path in exhausted or len(seen) >= target:
                    continue
                # Take pages from each endpoint in proportion to its share.
                pages_this_round = max(1, round(share * 2))
                for offset in range(pages_this_round):
                    p = page + offset
                    if p > MAX_PAGES:
                        exhausted.add(path)
                        break
                    data = await _get(client, path, page=p)
                    if not data or not data.get("results"):
                        exhausted.add(path)
                        break
                    before = len(seen)
                    for movie in data["results"]:
                        if len(seen) >= target:
                            break
                        seen.setdefault(movie["id"], None)
                    bar.update(len(seen) - before)
            page += 2
            if len(exhausted) == len(endpoints):
                break

    ids = list(seen)
    DISCOVERED_IDS_PATH.write_text(json.dumps(ids))
    log.info("discovery_complete", unique_ids=len(ids), pages_scanned=page - 1)
    return ids


async def fetch_movie(client: httpx.AsyncClient, movie_id: int, sem: asyncio.Semaphore) -> bool:
    """Fetch one movie's full detail and write it to the raw cache verbatim.

    `append_to_response=credits,keywords` folds three API calls into one: the
    movie's own fields, its cast/crew (where tmdb_person_id comes from), and its
    keywords (which feed both the embedding text and the graph).
    """
    out_path = settings.raw_dir / f"{movie_id}.json"
    async with sem:
        data = await _get(
            client, f"/movie/{movie_id}", append_to_response="credits,keywords"
        )
    if data is None:
        return False
    # Write to a temp file then rename: an interrupted run can't leave a
    # half-written JSON file that the transform would later choke on.
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.rename(out_path)
    return True


async def run(target: int) -> None:
    if not settings.has_tmdb_credential:
        raise MissingConfigError(
            "Set TMDB_API_READ_ACCESS_TOKEN (preferred) or TMDB_API_KEY in .env."
        )
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    # The raw cache directory IS the resume checkpoint — no separate state file.
    already_cached = {int(p.stem) for p in settings.raw_dir.glob("*.json")}
    log.info("pull_start", target=target, already_cached=len(already_cached))

    client_kwargs = _auth({"base_url": BASE_URL, "timeout": 30.0})
    async with httpx.AsyncClient(**client_kwargs) as client:
        ids = await discover_ids(client, target)
        todo = [i for i in ids if i not in already_cached]
        log.info("pull_plan", discovered=len(ids), skipping_cached=len(ids) - len(todo), to_fetch=len(todo))

        if not todo:
            log.info("pull_complete", fetched=0, failed=0, cached_total=len(already_cached))
            return

        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [fetch_movie(client, i, sem) for i in todo]
        results = []
        with tqdm(total=len(tasks), desc="fetching movies", unit="movie") as bar:
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                bar.update(1)

    ok = sum(results)
    log.info(
        "pull_complete",
        fetched=ok,
        failed=len(results) - ok,
        cached_total=len(list(settings.raw_dir.glob("*.json"))),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull TMDB movies into data/raw/")
    parser.add_argument(
        "--limit", type=int, default=settings.catalog_size,
        help=f"how many movies to pull (default: CATALOG_SIZE={settings.catalog_size})",
    )
    args = parser.parse_args()
    asyncio.run(run(args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
