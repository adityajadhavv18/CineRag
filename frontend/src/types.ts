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

export interface ChatResponse {
  /** Markdown-ish, with [1] [2] markers indexing 1-based into `sources`. */
  response: string
  intent: Intent | null
  lead_engine: 'vector' | 'graph' | 'both' | null
  sources: Source[]
  franchise: FranchiseTimeline[]
  degraded: boolean
  trace: string[]
}

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
