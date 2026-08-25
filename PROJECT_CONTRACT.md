# CineRAG — Project Contract & 7-Day Build Plan

> Working name: **CineRAG** (rename freely). A LangGraph-based, intent-driven movie
> recommendation agent over a **Qdrant** vector store + **Neo4j** knowledge graph, grounded
> on real **TMDB** data, served via **FastAPI**, evaluated with **LangSmith**.
>
> This document is the single source of truth for the build. It is written to be handed to
> **Claude Code CLI**. Keep it in the repo root. When decisions change, change them *here* first.

---

## 0. How to use this document (read this first)

**This is a learning build, not a delivery sprint.** The human driving this is learning the
concepts as they implement. Therefore, Claude Code must operate in **observe → learn → implement**
mode, not "implement for the sake of implementing":

1. **Before writing code for a step**, explain the concept in plain language: what it is, why it's
   here, and what would break without it. Keep it short (a few sentences), then build.
2. **Prefer small, runnable vertical slices** over large unrunnable scaffolding. After each step,
   there must be something the human can *run and see work*.
3. **Explain non-obvious choices inline** as short code comments or a one-line note — especially
   anything about embeddings, Cypher, LangGraph state, or fusion scoring.
4. **When something in this contract is ambiguous or seems wrong, STOP and ask** — do not guess and
   proceed. The human will bring open questions back to the planning chat.
5. **Do not skip the "Learning checkpoint" at the end of each day.** It is a self-test, not
   decoration. If the human can't answer it, revisit before advancing.

---

## 1. Locked decisions (do NOT relitigate)

| Decision | Choice | Why (one line) |
|---|---|---|
| Agent framework | **LangGraph** | Explicit state machine; conditional routing is first-class |
| Personalization | **Stateless v1** — no user profiles, no watch history | Ship the core loop first; personalization is a later phase |
| Retrieval strategy | **Intent-driven** — the intent node picks the lead engine | This is what makes two stores worth having |
| Vector store | **Qdrant** (Docker, local) | Dense semantic search, payload filtering, free self-host |
| Graph store | **Neo4j Community** (Docker, local) | Standard for movie graphs; Cypher is readable |
| LLM | **OpenAI** — `gpt-4o-mini` (chat), `text-embedding-3-small` (embeddings) | Only paid dependency; cheap at this scale |
| Data source | **Real TMDB data**, ~5,000 movies (popular + top-rated blend) | Real facts are the whole point of the KG |
| Synthetic data | **FORBIDDEN** for movie facts | Fake edges destroy the graph's only value |
| Images | Store `poster_path` + `backdrop_path` **strings only**; serve from TMDB CDN | Never store image binaries in the DB |
| Observability / eval | **LangSmith** (tracing + eval datasets) | Native LangGraph integration |
| API | **FastAPI + uvicorn** | Async, Pydantic, streaming-ready |
| Package manager | **uv** (fallback: pip) | Fast, reproducible |
| Python | **3.11+** | — |

**Explicitly OUT of scope for the 7-day build (stretch / later):** user personalization &
collaborative filtering, web-search fusion (live streaming availability), SSE streaming, multi-turn
memory beyond simple follow-up. Note where hooks should be left for these, but do not build them.

---

## 2. Architecture

### 2.1 The core idea (why two stores)

- **Vector store (Qdrant) answers "what is it *about* / what's the vibe?"** — fuzzy, meaning-based.
  We embed each movie's plot + tagline + genres + keywords. Good for *"a cozy heist with a twist."*
- **Graph store (Neo4j) answers "what is *connected* to what?"** — exact facts and relationships.
  Movies, people, genres, franchises as nodes and edges. Good for *"movies directed by Nolan"* and
  *"what else is this actor in?"* and *"the sequels in this franchise."*
- **The intent node decides which engine leads.** Not every query fires both. That decision is the
  architecture. A blended query (*"mind-bending sci-fi like Inception with a female lead"*) fires
  both, and the rerank rewards movies that appear in **both** lists — that fusion is where the design
  earns its keep.

### 2.2 Flow

```
User query
    │
    ▼
intent_node ── classifies intent + picks lead_engine + extracts filters/entities
    │
    ├─ general        → general_node ─────────────────────────────► Response
    ├─ off_topic      → off_topic_node ────────────────────────────► Response
    ├─ clarification  → clarification_node (corpus probe + Qs) ─────► Response
    └─ recommend /
       factual_lookup /
       follow_up      → retrieval fan-out (which nodes fire is decided by lead_engine):
                          ├─ vector_retrieve (Qdrant) — dense semantic + payload filters
                          ├─ graph_retrieve  (Neo4j)  — NEW candidates from people/genres/filters
                          └─ graph_enrich    (Neo4j)  — link signals for ALREADY-KNOWN tmdb_ids
                                     │
                                     ▼
                          rerank_node (RRF fusion + additive bonuses + top-k)
                                     │
                                     ▼
                          final_response_node (grounded, cited picks)
                                     │
                                     ▼
                          franchise_node (sequel/collection timeline, if any) ─► Response
```

**lead_engine routing:**
- `vector` — vibe query. Runs `vector_retrieve` + `graph_enrich`. The graph produces **no new
  candidates** here; it only attaches cast/genre/franchise link signals to the vector hits for the
  rerank. Example: *"something tense and claustrophobic."*
- `graph`  — fact query. Runs `graph_retrieve` alone; vector is skipped.
  Example: *"movies directed by Bong Joon-ho."*
- `both`   — blended query. Runs `vector_retrieve` + `graph_retrieve` (both produce real candidate
  lists), then `graph_enrich` over the merged set. Overlap is rewarded automatically by RRF (§3.5).
  Example: *"gritty crime dramas starring Denzel Washington."*

> **Retrieve vs enrich is the distinction to keep straight:** `graph_retrieve` answers *"which movies
> match these facts?"* and returns titles we didn't have. `graph_enrich` answers *"what is connected
> to these specific movies I already found?"* and returns signals, never new titles.

### 2.3 Nodes (this is the module map)

| Node | Role | Notes |
|---|---|---|
| `intent_node` | Classify intent, pick `lead_engine`, extract `filters` + `entities`, refine query | The brain of routing |
| `general_node` | Greetings, "what can you do", capability answers | No retrieval |
| `off_topic_node` | Politely reject non-movie queries | No retrieval |
| `clarification_node` | Vague-but-on-topic → probe catalog, ask grounded narrowing questions | Options drawn only from real catalog values |
| `vector_retrieve` | Qdrant dense semantic search + payload filters | Returns candidate movies + cosine scores |
| `graph_retrieve` | Neo4j Cypher over people/genres/franchises; exact-fact shortcut | Returns **new** candidate movies |
| `graph_enrich` | Neo4j lookup by known `tmdb_id` → cast/genre/collection links | Returns **signals only**, never new titles |
| `rerank_node` | RRF-fuse the lists, dedupe, additive bonuses, top-k | The fusion layer |
| `final_response_node` | Write grounded, cited recommendations | Only cites movies actually retrieved |
| `franchise_node` | Emit sequel/collection timeline for cited movies | The KG "wow" feature; from `belongs_to_collection` |

---

## 3. Data model

### 3.1 Canonical movie record (the snapshot; drives both stores)

Pulled from TMDB once into a local `data/movies.jsonl` (one JSON object per line). This is the
**deterministic source** both stores are built from — never hit TMDB live per query.

```jsonc
{
  "tmdb_id": 27205,
  "title": "Inception",
  "year": 2010,
  "overview": "…full plot summary…",          // for embedding
  "tagline": "Your mind is the scene of the crime.", // for embedding
  "genres": ["Action", "Science Fiction", "Adventure"], // both stores
  "keywords": ["dream", "heist", "subconscious"],       // for embedding + graph
  "director": [                                 // graph — objects, NOT bare names
    {"tmdb_person_id": 525, "name": "Christopher Nolan"}
  ],
  "cast": [                                     // graph (top ~10)
    {"tmdb_person_id": 6193, "name": "Leonardo DiCaprio", "character": "Cobb"},
    {"tmdb_person_id": 27578, "name": "Elliot Page",      "character": "Ariadne"}
  ],
  "collection_id": 999999,                      // franchise chain (nullable)
  "collection_name": "…",                       // nullable
  "rating": 8.4,                                // vote_average
  "popularity": 82.1,
  "runtime": 148,
  "release_date": "2010-07-15",
  "poster_path": "/edv5CZvWj09up…jpg",          // frontend (string only)
  "backdrop_path": "/s3TBrRGB1iav…jpg"          // frontend (string only)
}
```

> TMDB image URL = `https://image.tmdb.org/t/p/{size}/{poster_path}` (e.g. size `w500`).
> The backend stores only the path; the frontend builds the URL and loads from TMDB's CDN.

**`tmdb_person_id` is mandatory on every cast and crew entry.** TMDB's `credits` response already
carries a stable `id` per person — capture it, never discard it. Person names are *not* unique
(multiple real people share a name), so a name-keyed graph silently merges distinct people into one
node and corrupts every filmography query that traverses it. The ID is the identity; the name is
display text. See §3.3.

### 3.1a Two-stage pull (raw cache → transform)

Ingestion is **two separate steps**, and this is deliberate:

1. **Pull** → write the raw TMDB JSON response verbatim to `data/raw/{tmdb_id}.json`, one file per
   movie. This is the expensive, rate-limited, network-bound operation. Do it exactly once.
2. **Transform** → read `data/raw/*.json`, project into the canonical schema (3.1), append to
   `data/movies.jsonl`.

The reason: when a missing field is discovered later — `tmdb_person_id` very nearly was one — you
re-run the *transform* over the local cache in seconds, instead of re-hitting the TMDB API 5,000
times. The raw cache is the safety net for schema mistakes.

**Resumability:** maintain a `seen_ids` set (derived from what's already in `data/raw/`, or a
checkpoint file). Re-running the pull skips IDs already cached. `movies.jsonl` is **appended to**,
never rewritten wholesale.

### 3.2 Qdrant schema (one collection, one point per movie)

- **Search mode**: **dense semantic search only** (`text-embedding-3-small`) + payload filters.
  No sparse vector in the collection config for v1.
- **Payload** (stored for filtering + returning): `tmdb_id, title, year, genres, director,
  cast_names, rating, popularity, runtime, collection_id, poster_path, backdrop_path,
  **overview, tagline**`.
  > `overview`/`tagline` are stored as TEXT as well as being embedded. A vector cannot be read
  > back, so without the words nothing can describe a film and an LLM asked to would supply the
  > plot from memory. Refreshing them needs no re-embedding:
  > `uv run python -m ingest.build_qdrant --payload-only`.
- **Indexed payload fields** (for fast filtered search): `genres`, `year`, `rating`.

> **Phase-2 hook (deliberately deferred):** sparse/BM25 fusion is a *measured* Phase-2 add. It
> requires declaring a second named vector at collection-creation time, so retrofitting it means a
> full re-index of every point. It is deferred because the knowledge graph already supplies the
> exact-token precision BM25 would otherwise buy us — the graph is our keyword-precision layer.

#### Frozen embed-text template

The literal text that gets embedded, **verbatim**. Field order, separators, and labels are part of
the spec because they change the resulting vector:

```
{title} ({year})
{tagline}
{overview}
Genres: {comma-separated genres}
Keywords: {comma-separated keywords}
```

Missing optional fields (e.g. no tagline) collapse to an empty line rather than the literal string
`None`. **This template is a versioned decision:** changing it invalidates every stored vector and
requires re-embedding all ~5,000 points. Do not tweak it casually mid-build.

### 3.3 Neo4j schema

**Nodes:** `Movie {tmdb_id, title, year, overview, tagline, rating, popularity, runtime,
release_date, poster_path, backdrop_path}`,
`Person {person_id, name}`, `Genre {name}`, `Keyword {name}`, `Collection {id, name}`.

**Relationships:**
- `(Person)-[:DIRECTED]->(Movie)`
- `(Person)-[:ACTED_IN {character}]->(Movie)`
- `(Movie)-[:HAS_GENRE]->(Genre)`
- `(Movie)-[:HAS_KEYWORD]->(Keyword)`
- `(Movie)-[:PART_OF]->(Collection)`  ← franchise chain: movies in same Collection ordered by `year`

**Constraints/indexes:** unique on `Movie.tmdb_id`, **`Person.person_id`**, `Genre.name`,
`Collection.id`. Add a non-unique index on `Person.name` for lookup by name.

**Person merge is by ID, name is a set property.** In `build_neo4j.py`:

```cypher
MERGE (p:Person {person_id: $person_id})
SET   p.name = $name
```

Never `MERGE (p:Person {name: $name})`. Merging on name collapses distinct people who share a name
into one node — one "John Williams" node accumulating three different careers — which makes every
filmography and "what else is this actor in?" query silently wrong. One real person = one node,
keyed by TMDB's stable `person_id`. Lookups *by name* still work via the `Person.name` index; they
just resolve to the correct node (or legitimately to several, if the name is genuinely shared).

### 3.4 Intent node output schema

```jsonc
{
  "intent": "recommend | factual_lookup | follow_up | clarification | general | off_topic",
  "lead_engine": "vector | graph | both",
  "refined_query": "self-contained restatement of what the user wants",
  "filters": { "genres": [], "year_range": null, "min_rating": null, "people": [] },
  "entities": { "people": [], "titles": [], "genres": [] }
}
```

**`filters.people` vs `entities.people` — these are not duplicates.** They carry opposite strengths
and the intent prompt must distinguish them:

| Field | Strength | Meaning | Retrieval behaviour | Example phrasing |
|---|---|---|---|---|
| `filters.people` | **Hard constraint** | The person **must** appear in the movie | Exclude any movie without them | *"movies **with** Denzel Washington"*, *"**directed by** Bong Joon-ho"*, *"**starring** …"* |
| `entities.people` | **Soft signal** | The person is *mentioned* as a reference point | Use for graph traversal/enrichment and scoring; **never exclude** on it | *"something **like** a Tarantino film"*, *"**in the spirit of** Kubrick"*, *"gives me **Wes Anderson** vibes"* |

The failure this prevents: *"something like a Tarantino film"* treated as a hard filter returns only
the ~10 movies Tarantino actually directed, when the user wanted the *style*. Conversely, *"movies
with Denzel Washington"* treated as a soft signal returns crime dramas he isn't in.

### 3.5 Rerank fusion — Reciprocal Rank Fusion (RRF)

**Why not multipliers.** Qdrant returns cosine similarities; Cypher returns unscored rows. Those two
things live on incomparable scales — there is no meaningful base score for a graph-only candidate, so
multiplying it by 1.8 multiplies an undefined number. RRF sidesteps this entirely by throwing the raw
scores away and fusing on **rank position**, which both stores do have.

**The fusion score.** For each candidate movie, sum its reciprocal rank across every list it appears
in:

```
rrf_score(movie) = Σ over lists L containing movie:  1 / (k + rank_L(movie))

k = 60          # standard damping constant; blunts the dominance of rank-1 items
rank_L          # 1-based position of the movie within list L
```

**The "appears in both lists" boost is not a separate rule — it emerges.** A movie present in both
the vector and graph lists collects **two** reciprocal-rank contributions that sum, so it
automatically outranks a movie that appears in only one list at a comparable position. The fusion win
the architecture is built around falls out of the formula rather than being bolted on as a ×1.8
special case. This is the point of choosing RRF.

**Secondary signals are additive, never multiplicative** — they nudge the fused ordering, they don't
rescale it:

| Signal | Adjustment |
|---|---|
| Exact genre match to a requested genre | small additive bonus |
| High rating / popularity | small additive tiebreaker |
| Graph link signals from `graph_enrich` (shared cast/genre/collection with a strong hit) | small additive bonus |

Keep the additive bonuses **small relative to the RRF spread** — they should break ties and nudge
neighbours, not reorder the fused list wholesale. Tune the constants against real queries on Day 5
and record the chosen values here.

Dedupe key: `tmdb_id` — accumulate contributions into one entry per movie rather than keeping the
higher of two rows. Output: top ~15, plus a `citations` list of only the movies actually cited in the
final answer.

---

## 4. Repo structure

```
cinerag/
├── PROJECT_CONTRACT.md          ← this file
├── pyproject.toml               ← uv-managed deps
├── .env.example                 ← every var documented
├── docker-compose.yml           ← Qdrant + Neo4j
├── core/
│   ├── config.py                ← pydantic-settings singleton
│   ├── logger.py                ← structured logging
│   └── llm.py                   ← OpenAI client (chat + embeddings) + LangSmith wiring
├── data/
│   ├── raw/                     ← raw TMDB JSON, one file per movie (git-ignored; the cache)
│   ├── movies.jsonl             ← the TMDB snapshot (git-ignored; large)
│   └── .gitkeep
├── ingest/
│   ├── tmdb_pull.py             ← resumable TMDB fetch → data/raw/{id}.json   (stage 1)
│   ├── transform.py             ← data/raw/*.json → movies.jsonl              (stage 2)
│   ├── build_qdrant.py          ← embed + upsert into Qdrant
│   └── build_neo4j.py           ← build the graph
├── stores/
│   ├── qdrant_client.py         ← thin wrapper: search + filter
│   └── neo4j_client.py          ← thin wrapper: Cypher queries
├── agent/
│   ├── state.py                 ← AgentState TypedDict
│   ├── graph.py                 ← StateGraph builder + compiled singleton
│   └── nodes/
│       ├── intent.py
│       ├── general.py
│       ├── off_topic.py
│       ├── clarification.py
│       ├── vector_retrieve.py
│       ├── graph_retrieve.py
│       ├── graph_enrich.py
│       ├── rerank.py
│       ├── final_response.py
│       └── franchise.py
├── server/
│   ├── schemas.py               ← Pydantic request/response models
│   ├── routes.py                ← POST /api/v1/chat
│   └── main.py                  ← FastAPI app + lifespan + CORS
├── eval/
│   ├── dataset.py               ← builds the LangSmith eval dataset
│   └── run_eval.py              ← runs evaluators
└── tests/
    └── …                        ← per-node unit tests
```

---

## 5. Conventions & guardrails (Claude Code MUST obey)

- **Real data only.** Never fabricate movie facts, cast, directors, plots, or franchise links.
  Every graph edge and every recommended title must trace to the TMDB snapshot.
- **Grounding is non-negotiable, and it covers CLAIMS as well as TITLES.** `final_response_node`
  may only recommend/cite movies present in the retrieved results — if retrieval is empty, say so
  rather than inventing a title. It must equally not invent *facts about* those movies: a plot
  summary must paraphrase the stored `overview`, never the model's own memory of the film.
  Citation validation catches the first failure; supplying the real text is what prevents the
  second, because nothing downstream can detect a fluent, wrong synopsis.
- **Never store image binaries.** Only `poster_path` / `backdrop_path` strings. Images load from
  TMDB CDN client-side.
- **Ingestion must be resumable, and pulled once.** `tmdb_pull.py` caches raw JSON to `data/raw/` and
  checkpoints via a `seen_ids` set; re-running resumes, never restarts from zero. Respect TMDB rate
  limits (backoff on 429). Transform is a *separate* step over the cache (§3.1a) — never re-pull the
  API to fix a schema mistake.
- **Person identity is `tmdb_person_id`, never name.** Applies to the record schema, the graph merge,
  and any lookup path (§3.3).
- **The embed-text template is frozen** (§3.2). Changing it means re-embedding every point; treat it
  as a versioned decision, not a tweak.
- **Intent decides the lead engine.** Do not blindly fire both retrievers on every query.
- **Filter values must be normalised against real catalogue vocabulary.** Both stores match payload
  values *exactly*, so a filter for `"crime"` against a catalogue that says `"Crime"` matches nothing
  and fails **silently** — zero results are indistinguishable from "no such movies". Any value the
  LLM produces that reaches a store as a filter (genres today; anything similar later) must be mapped
  onto real catalogue values first, and unmappable values dropped rather than passed through. See
  `agent/catalog.py`.
- **Every node is traceable.** Wire LangSmith from Day 3 onward; don't bolt it on at the end.
- **Every node emits structured logs from Day 3.** Each node logs its key decisions as structured
  fields as it is written — `intent` classified, `lead_engine` chosen, candidate counts per store,
  reranked top-k size, final citation count. This costs almost nothing while building and is what
  makes the Day-7 LangSmith eval and all debugging tractable; retrofitting observability at the end
  is far harder. No node ships without its log line.
- **Config via `core/config.py` only.** No hardcoded URLs, keys, or model names in nodes.
- **Fail gracefully.** A store being down, an LLM error, or a bad TMDB row must degrade cleanly
  (log + fallback), not crash the request.
- **Explain as you build.** This is a learning build — see Section 0.

---

## 6. Environment variables (`.env.example`)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key (chat + embeddings) |
| `CHAT_MODEL` | Default `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Default `text-embedding-3-small` |
| `TMDB_API_KEY` | TMDB API key (ingestion only) |
| `QDRANT_URL` | Default `http://localhost:6333` |
| `QDRANT_COLLECTION` | e.g. `movies` |
| `NEO4J_URI` | Default `bolt://localhost:7687` |
| `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j auth |
| `LANGSMITH_API_KEY` | LangSmith tracing + eval |
| `LANGSMITH_PROJECT` | Project name in LangSmith |
| `LANGCHAIN_TRACING_V2` | `true` to enable tracing |
| `CATALOG_SIZE` | Target movie count for ingestion (default 5000) |

---

## 7. The 7-Day Plan

Each day: **Goal → Learn first → Build → Acceptance criteria → Learning checkpoint.**
Do not advance a day until acceptance criteria pass *and* the checkpoint question is answerable.

### Day 1 — Foundations, infra, and the TMDB snapshot
**Goal:** A running local environment and a real dataset of ~5,000 movies on disk.

**Learn first:** What a vector DB vs a graph DB each store and why we need both (Section 2.1).
How TMDB's API is shaped — basic movie fields vs `append_to_response` for cast/crew/keywords.

**Build:**
1. Init repo, `pyproject.toml` (uv), `.env.example`, `core/config.py`, `core/logger.py`.
2. `docker-compose.yml` bringing up Qdrant + Neo4j; verify both are reachable.
3. `ingest/tmdb_pull.py` (**stage 1**): pull `CATALOG_SIZE` movies (blend TMDB *popular* +
   *top_rated*, deduped by `tmdb_id`), each with `append_to_response=credits,keywords`. Write the
   **raw** response verbatim to `data/raw/{tmdb_id}.json`. **Resumable** via a `seen_ids` set,
   rate-limit aware (backoff on 429).
4. `ingest/transform.py` (**stage 2**): read `data/raw/*.json` → project into the canonical schema
   (3.1) → append to `data/movies.jsonl`. Must capture `tmdb_person_id` on every cast and crew entry.

**Acceptance:** `docker compose up` works; `movies.jsonl` has ~5,000 well-formed records, every
cast/crew entry carrying a `tmdb_person_id`; killing and re-running the pull *resumes* rather than
restarts; re-running *transform* alone rebuilds `movies.jsonl` from cache with zero API calls.

**Learning checkpoint:** *Which fields of a movie record feed the vector store, which feed the graph,
and which are frontend-only — and why?*

---

### Day 2 — Populate both stores; test raw retrieval (no agent yet)
**Goal:** Query Qdrant and Neo4j directly and *feel* the difference between them.

**Learn first:** What an embedding is (text → vector), why similar meaning ⇒ nearby vectors. Basic
Cypher: `MATCH`, `-[:REL]->`, `WHERE`, `RETURN`. Graph modeling: nodes vs relationships.

**Build:**
1. `ingest/build_qdrant.py`: compose the embed-text per movie **using the frozen template in §3.2**,
   batch-embed with `text-embedding-3-small`, upsert points + payload, create payload indexes.
   Dense vector only — no sparse config.
2. `ingest/build_neo4j.py`: create constraints (`Person.person_id` unique, **not** name), then MERGE
   movies/people/genres/keywords/collections and all relationships (3.3).
3. `stores/qdrant_client.py` + `stores/neo4j_client.py`: thin query wrappers.
4. A scratch script or notebook: run *"heist movie with a twist"* against Qdrant, and
   *"movies directed by Christopher Nolan"* against Neo4j. Observe which store nails which.

**Acceptance:** Qdrant returns semantically sensible movies for a vibe query; Neo4j returns exact,
correct filmographies and franchise members for fact queries.

**Learning checkpoint:** *Give one query each store answers well and one it answers badly. Why?*

---

### Day 3 — LangGraph skeleton + the cheap paths + tracing
**Goal:** A compiled LangGraph that runs end-to-end for `general` and `off_topic`, with LangSmith
tracing live.

**Learn first:** LangGraph core — `StateGraph`, state (`TypedDict`), nodes as functions, edges,
**conditional edges** (routing). Why state flows through, not around, nodes. **And reducers:** when
`lead_engine=both`, two retrieval nodes run in parallel and write to the same state field. LangGraph
does *not* silently keep the last write — it raises `InvalidUpdateError` ("Can receive only one value
per step"). An `Annotated[list, operator.add]` reducer tells it how to *combine* the two writes, and
is therefore what makes the parallel fan-out legal at all. Introduce the annotation here, when
`AgentState` is first defined, with a one-line comment explaining why it's there — not as a Day-4
debugging surprise.

**Build:**
1. `agent/state.py` (`AgentState`, with reducers on the concurrently-written retrieval fields),
   `agent/graph.py` (builder + compiled singleton).
2. `agent/nodes/intent.py`: LLM classifier emitting the schema (3.4). Start simple: reliably separate
   `general` / `off_topic` / (everything-else → placeholder).
3. `general.py`, `off_topic.py`. Conditional routing from intent → these.
4. Wire LangSmith (`LANGCHAIN_TRACING_V2=true`); confirm traces appear.

**Acceptance:** "hi" → general; "what's the weather" → off_topic; both visible as traces in LangSmith
with per-node spans.

**Learning checkpoint:** *How does a conditional edge decide the next node, and where does the
routing value come from?*

---

### Day 4 — Retrieval nodes + intent-driven routing (the core)
**Goal:** Real retrieval wired into the graph, with the intent node choosing the lead engine.

**Learn first:** The `lead_engine` idea (2.2) — vibe→vector, fact→graph, blended→both. The
**retrieve vs enrich** split: `graph_retrieve` produces new candidates, `graph_enrich` only attaches
link signals to movies we already found.

**Build:**
1. Extend `intent_node` to emit `lead_engine`, `filters`, `entities` reliably — including the
   hard-constraint vs soft-signal distinction for people (§3.4).
2. `vector_retrieve.py` (Qdrant + payload filters), `graph_retrieve.py` (Cypher + exact-fact
   shortcut for pure lookups), `graph_enrich.py` (link signals by known `tmdb_id`).
3. Routing per `lead_engine`:
   - `vector` → `vector_retrieve` + `graph_enrich`
   - `graph`  → `graph_retrieve`
   - `both`   → `vector_retrieve` + `graph_retrieve` in parallel, then `graph_enrich` over the
     merged set (this is where the Day-3 reducers earn their keep)
4. For now, `final_response` can be a stub that dumps retrieved titles — real generation is Day 5.

**Acceptance:** A fact query routes graph-led and returns correct titles; a vibe query routes
vector-led; a blended query runs both and produces two candidate lists.

**Learning checkpoint:** *Trace one blended query: what did each store return, and which movies
appeared in both?*

---

### Day 5 — Rerank/fusion + grounded, cited response
**Goal:** A genuinely working recommender: fused results, a real answer, honest grounding.

**Learn first:** Why naive concatenation of two lists is wrong, and why the two stores' raw scores
can't be compared directly — which is exactly why we fuse on *rank* with RRF (3.5). Why grounding +
citation stops the LLM inventing movies.

**Build:**
1. `rerank.py`: RRF-fuse the lists (`k=60`), accumulate per `tmdb_id`, apply the small additive
   bonuses, return top-k + citations. Verify the emergent overlap win: a movie in both lists should
   outrank a same-position movie in one list, with no special-case code doing it.
2. `final_response.py`: build a numbered context block from reranked movies, prompt the LLM to write
   recommendations that cite `[N]` and explain *why* each fits — using ONLY retrieved movies. Strip
   any citation the LLM didn't actually use.

**Acceptance:** "gritty crime dramas with Denzel Washington" returns a ranked, explained, correctly
cited set; every cited movie exists in the retrieval results; empty retrieval yields an honest
"nothing found" rather than a hallucinated list.

**Learning checkpoint:** *What happens to the ranking of a movie that appeared in both lists vs one
that appeared in only one — and why does RRF produce that behaviour without a rule that says so?*

---

### Day 6 — Clarification, follow-up, and the franchise timeline
**Goal:** The "smart" behaviors that make it feel like an agent, plus the KG showcase.

**Learn first:** When a query is on-topic but too vague to retrieve well. How a follow-up refines the
previous turn. How a franchise chain falls out of the `Collection` node for free.

**Build:**
1. `clarification_node`: probe the catalog for the vague topic, ask 1–3 grounded narrowing questions
   whose options are drawn only from real catalog values; degrade gracefully if the probe is empty.
2. Follow-up handling: intent node fuses prior turn + new message into a self-contained query
   (stateless-friendly — history passed in the request, no server memory).
3. `franchise_node`: for cited movies in a collection, emit the ordered sequel/prequel timeline
   from Neo4j.

**Acceptance:** "recommend some action movies" (too vague) → clarifying questions with real options;
answering them → a sharp retrieval; a cited franchise film → a correct ordered timeline.

**Learning checkpoint:** *Why must clarification options come from the catalog and not the LLM's
imagination?*

---

### Day 7 — API, evaluation, and hardening
**Goal:** Serve it over HTTP, measure it with LangSmith, and make it not fall over.

**Learn first:** What makes an eval trustworthy — a fixed dataset + explicit success criteria per
query type (did the right intent fire? right lead engine? real cited movies? clarification when
vague?).

**Build:**
1. `server/schemas.py`, `server/routes.py` (`POST /api/v1/chat`), `server/main.py` (lifespan opens
   store clients + CORS for the frontend). Response includes `sources` with `poster_path` /
   `backdrop_path` so the Netflix-clone frontend can render cards.
2. `eval/dataset.py`: ~20–30 labeled test queries spanning all intents. `eval/run_eval.py`:
   LangSmith evaluators (intent correctness, grounding = "all cited movies are real", routing
   correctness).
3. Hardening pass: graceful degradation when a store/LLM fails; input validation; structured logs on
   every node.

**Acceptance:** `POST /api/v1/chat` returns grounded recommendations with poster paths; the LangSmith
eval runs and reports pass/fail per criterion; killing Neo4j mid-request degrades cleanly instead of
500-crashing the whole app.

**Learning checkpoint:** *Which eval would catch a regression where the agent starts recommending
movies that aren't in the catalog?*

---

## 8. Definition of done (v1)

- Real ~5,000-movie TMDB snapshot loaded into both Qdrant and Neo4j.
- LangGraph agent with all six intents and intent-driven lead-engine routing.
- RRF fusion rerank that provably rewards cross-store overlap.
- Grounded, cited responses that never invent titles.
- Clarification, follow-up, and franchise-timeline working.
- `POST /api/v1/chat` serving poster/backdrop paths to a frontend.
- A LangSmith eval dataset that gates regressions.
- Graceful degradation when any dependency is unavailable.

---

## 9. Frontend note (built separately by the human)

The Netflix-clone frontend is **not** part of this backend build, but the API is shaped for it:
each recommended movie in the response carries `poster_path` and `backdrop_path`. The frontend builds
`https://image.tmdb.org/t/p/w500{poster_path}` for card thumbnails and a larger size /
`backdrop_path` for hero banners. No images are stored or served by this backend.

---

## 10. Open-questions protocol

When Claude Code hits something this contract doesn't cover or that seems wrong: **stop, state the
question clearly, propose the two or three options you see, and wait.** The human takes open
questions back to the planning chat and returns with a decision to record here. Do not silently
choose and move on.

---

## 11. Decision log

Resolved open questions, newest last. Each entry names the sections it changed.

**Pre-Day-1 review (5 questions raised, all resolved):**

| # | Question | Decision | Sections changed |
|---|---|---|---|
| 1 | `Person` uniqueness on `name` merges distinct people sharing a name | **Bug fix.** Capture `tmdb_person_id` on every cast/crew entry; `Person` unique on `person_id`, name is a display property. `MERGE (p:Person {person_id: $id}) SET p.name = $name` | §3.1, §3.3, §5, Day 1, Day 2 |
| 2 | "Hybrid search" — dense-only or dense + sparse? | **Dense-only for v1**, wording corrected everywhere. Sparse/BM25 fusion is a measured Phase-2 add, deferred because it needs a full re-index and the KG already provides exact-token precision | §1, §2.2, §2.3, §3.2, Day 2 |
| 3 | Multiplicative boosts (×1.8/×1.3) aren't implementable across incomparable score scales | **Replaced with Reciprocal Rank Fusion**, `k=60`. The both-lists boost *emerges* from summed reciprocal ranks rather than being a special case; genre/rating/position become small additive bonuses | §3.5, §8, Day 5 |
| 4 | `filters.people` vs `entities.people` — duplicate? | **Both kept, semantics recorded.** `filters` = hard constraint ("with X", "directed by X") → exclude; `entities` = soft signal ("like X") → traverse/score, never exclude | §3.4, Day 4 |
| 5 | Where does graph *enrichment* live? | **Split into two nodes.** `graph_retrieve` returns new candidates; `graph_enrich` returns link signals for known `tmdb_id`s only | §2.2, §2.3, §4, Day 4 |

**Additional guidelines adopted at the same time:**

- **A — Two-stage pull with a raw cache.** `tmdb_pull.py` → `data/raw/{id}.json`, then `transform.py`
  → `movies.jsonl`. Schema mistakes are fixed by re-transforming locally, never by re-pulling 5,000
  movies. (§3.1a, §4, §5, Day 1)
- **B — Frozen embed-text template.** The literal template is in §3.2 and is a versioned decision;
  changing it invalidates every stored vector. (§3.2, §5, Day 2)
- **C — Structured logs from Day 3.** Every node logs its key decisions as structured fields *as it
  is written*. Observability is not retrofitted at the end. (§5, Day 3, Day 7)
- **Parallel fan-out reducers** — confirmed as a Day-3 teaching point: concurrent writes to shared
  `AgentState` fields need `Annotated[list, operator.add]`. Introduced when `AgentState` is defined,
  not discovered as a Day-4 bug. (Day 3, Day 4)

**Day 3 build (corrections found while implementing):**

| # | Finding | Resolution | Sections changed |
|---|---|---|---|
| 6 | The reducer rationale as originally written was **wrong**: LangGraph does not silently clobber a concurrently-written key | It raises `InvalidUpdateError` ("Can receive only one value per step"). The reducer is therefore what makes parallel fan-out *legal*, not what prevents silent data loss. Corrected wording; proven by `tests/test_agent.py::test_without_a_reducer_concurrent_writes_are_rejected` | Day 3, `agent/state.py` |
| 7 | LLM-extracted genres came back lower-cased (`"crime"`), and both stores match payload values exactly → **zero results, silently** | New `agent/catalog.py`: canonical vocabulary read from the graph, injected into the intent prompt, plus `normalize_genres()` as a code-level backstop. Unmappable values are dropped, never passed through | §5 (new guardrail), Day 3, Day 4 |
| 8 | Named people were being dropped from both `filters.people` and `entities.people` (e.g. "films directed by Bong Joon-ho" extracted nobody) | Intent prompt now requires every named person to land in exactly one of the two, with worked examples | §3.4 prompt, Day 3 |

**Post-Day-6 addition — film detail (`overview` in the stores):**

Raised because "tell me about Inception" returned 48 characters. Three stacked causes, and the
data one is the reason the other two could not simply be prompted away.

| # | Finding | Decision | Sections changed |
|---|---|---|---|
| 9 | `overview` and `tagline` were embedded into the Qdrant vector and then **discarded** — present in `movies.jsonl`, absent from both stores. Nothing could describe a film | Store both fields in **both** stores, same reasoning as `poster_path`: either store may answer alone. Qdrant via a **payload-only** update (no re-embedding, vectors untouched, ~3s); Neo4j via a normal rebuild (~20s) | §3.2 payload, §3.3 Movie node, `ingest/build_qdrant.py --payload-only` |
| 10 | **A grounding hole**: `validate_citations` checks *which films are named*, never *what is claimed about them*. Asked to describe a film with no plot supplied, the model writes a fluent synopsis from memory and nothing catches it | Supply the real overview, labelled `plot:`, and require the answer to paraphrase it. Plot is included **only for detail-shaped queries** — 15 recommendation candidates × ~70 tokens is ~1,000 wasted tokens per request. Tested in `tests/test_detail.py` | §5 (grounding now covers claims, not just titles) |
| 11 | No 7th intent added. "Tell me about X" is `factual_lookup` with a different response shape | `wants_detail()` branches on the **user's own words** (a named title + no attribute word), not on `refined_query` — which is model-written and had rewritten "tell me about Inception" into "who directed Inception", silently narrowing the request | Day 5 `final_response`, taxonomy unchanged at six intents |
| 12 | `refined_query` was **answering** questions instead of restating them ("what is Inception about" → "Inception is a film about a thief who enters dreams") — an ungrounded answer from model memory, sitting in state before retrieval ran | Intent prompt now forbids two things explicitly: never answer the question, never narrow the request | §3.4 prompt |
| 13 | Titles that are also ordinary words (**Inception**, Alien, Up, Her, Heat) classified as `general` about 3 times in 4 — the model read "inception" as the noun | `general` is now defined as *only* about the conversation or the assistant: any message naming a film, person or genre can never be `general`. Stable 4/4 after the change, with `hi`/`thanks` unaffected | §3.4 prompt |

**Day 7 build — found by the eval on its first run (25/28), all fixed to 28/28:**

| # | Finding | Resolution | Sections changed |
|---|---|---|---|
| 14 | "Who is the president of France" classified `factual_lookup`. The prompt defined it as "asks a specific answerable fact" — never saying *about a film* | `factual_lookup` now requires a FILM subject, and the prompt says explicitly to check the question's **subject, not its shape**: "who is the president of France" and "who directed Whiplash" are the same shape, different subjects. 3/3 after | §3.4 prompt |
| 15 | "Something like a Tarantino film" routed `vector` though `entities.people` held him — the decision table was being read as applying only to *hard* constraints | Table now states a SOFT mention counts for question A, and that naming someone as a style reference counts for B, with four worked examples | §2.2 / §3.4 prompt |
| 16 | "Only the 90s ones" (with history) classified `recommend`, not `follow_up` — though `refined_query` fused correctly | `follow_up` redefined mechanically: *if resolving the message required reading the history, it is follow_up* — even when what it asks for is a recommendation | §3.4 prompt |
| 17 | **The eval's own label was wrong**: "mind-bending sci-fi like Inception" was asserted to *cite* Inception. It correctly cited *The Prestige, Dark City, The Matrix, Primer* instead — recommending the reference film back is a worse answer | Split the criterion: `must_include` (the answer cites it) vs `must_retrieve` (retrieval surfaces it). A reference film belongs in the second | `eval/dataset.py`, `eval/run_eval.py` |

> Worth recording *why* #17 was fixed in the eval rather than the agent: the agent's behaviour was
> right and the label was testing the wrong property. The discipline is that a failing case must be
> diagnosed before it is changed — an eval edited to go green is worse than no eval. #14–#16 were
> genuine system faults and were fixed in the prompt, not the labels.