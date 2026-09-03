/**
 * Mirrors server/schemas.py. Kept hand-written rather than generated: it is one
 * small file, and the comments below record what the *shapes mean*, which a
 * generator would strip.
 */

export interface MovieCard {
  tmdb_id: number
  title: string
  year: number | null
  rating: number | null
  poster_path: string | null
  backdrop_path: string | null
  overview: string | null
  genres: string[]
  /** Only on a person's acting credits. */
  character: string | null
}

export interface BrowseRow {
  title: string
  genre: string | null
  films: MovieCard[]
}

export interface BrowseResponse {
  /** Carousel slides. Every one is guaranteed to have a backdrop. */
  heroes: MovieCard[]
  rows: BrowseRow[]
  /** Empty rows + degraded=false means an empty catalogue; + degraded=true
   *  means the graph was unreachable. The UI must not confuse the two. */
  degraded: boolean
}

export interface Credit {
  person_id: number
  name: string
  character: string | null
}

export interface MovieDetail {
  tmdb_id: number
  title: string
  year: number | null
  overview: string | null
  tagline: string | null
  rating: number | null
  runtime: number | null
  release_date: string | null
  poster_path: string | null
  backdrop_path: string | null
  genres: string[]
  keywords: string[]
  cast: Credit[]
  directors: Credit[]
  collection_id: number | null
  collection_name: string | null
}

export interface FranchiseFilm {
  tmdb_id: number
  title: string
  year: number | null
  /** The film the timeline was seeded from — worth marking in the row. */
  is_seed: boolean
  poster_path: string | null
}

export interface FranchiseTimeline {
  collection_id: number | null
  collection_name: string | null
  films: FranchiseFilm[]
}

export interface SimilarResponse {
  /** Vector neighbours: films that FEEL like this one. */
  films: MovieCard[]
  /** The graph's exact answer: films that BELONG with it. Never conflate. */
  franchise: FranchiseTimeline | null
  degraded: boolean
}

export interface PersonResponse {
  person_id: number
  name: string
  acted: MovieCard[]
  directed: MovieCard[]
}

// ── chat ─────────────────────────────────────────────────────────────────────

export type Intent =
  | 'recommend'
  | 'factual_lookup'
  | 'follow_up'
  | 'clarification'
  | 'general'
  | 'off_topic'

export interface Turn {
  role: 'user' | 'assistant'
  content: string
}

export interface Source {
  tmdb_id: number
  title: string
  year: number | null
  overview: string | null
  poster_path: string | null
  backdrop_path: string | null
}

export interface ClarifyOption {
  label: string
  /** How many films sit behind this option, counted out of the graph. An option
   *  the catalogue cannot satisfy is never offered, and this is the proof. */
  films: number
}

export interface ClarifyQuestion {
  id: string
  prompt: string
  /** Where the counts came from, when it is worth saying. */
  note: string | null
  options: ClarifyOption[]
  /** How this answer reads once composed back into a query: `phrase` holds one
   *  `{value}` placeholder, and `slot` says which side of `subject` it goes on
   *  — "Thriller crime films" (before) against "from the 1990s" (after). The
   *  server owns the wording; see `composeQuery`. */
  slot: 'before' | 'after'
  phrase: string
}

/**
 * The agent's narrowing questions, before they were flattened into prose.
 *
 * `response` says the same thing in words and stays authoritative — it is what
 * goes back to the agent as history. This is the same set with its options
 * intact, so they can be offered as choices. Null unless the agent actually
 * asked, and null on its ungrounded fallback: no counted options, no buttons.
 */
export interface Clarification {
  lead: string
  /** The noun the answers compose around: "crime films", or just "films". */
  subject: string
  questions: ClarifyQuestion[]
}

export interface ChatResponse {
  /** Markdown-ish, with [1] [2] markers indexing 1-based into `sources`. */
  response: string
  intent: Intent | null
  lead_engine: 'vector' | 'graph' | 'both' | null
  sources: Source[]
  franchise: FranchiseTimeline[]
  clarification: Clarification | null
  degraded: boolean
  trace: string[]
}

/** A cited film as it arrives mid-stream: a Source plus its marker number. */
export interface StreamedSource extends Source {
  /** 1-based; matches the `[n]` marker that arrived in the same breath. */
  n: number
}

/**
 * What POST /api/v1/chat/stream sends, in order.
 *
 * `stage` is a human sentence about the wait, not a node name. `token` is a
 * piece of the answer. `source` always arrives with (or just before) the marker
 * that references it, so a `[2]` is never a dead button. `done` carries what can
 * only be known at the end — including the authoritative `response`, which is
 * what gets stored in history.
 */
export type ChatEvent =
  | { type: 'stage'; label: string }
  | { type: 'token'; text: string }
  | { type: 'source'; source: StreamedSource }
  | { type: 'done'; result: Omit<ChatResponse, 'sources'> }
  | { type: 'error'; detail: string }

/**
 * An empty `sources` list is CORRECT for these intents — the agent is asking a
 * question or making small talk, not failing to find films. Rendering "no
 * results" here would be a lie about what happened.
 */
export const INTENTS_WITHOUT_FILMS: ReadonlySet<string> = new Set([
  'clarification',
  'general',
  'off_topic',
])
