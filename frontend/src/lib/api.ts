/**
 * The CineRAG API client.
 *
 * One rule shapes this whole file: THE SERVER IS STATELESS. It holds no
 * conversation, so `ask()` takes the history as an argument and the caller owns
 * it. Follow-ups like "only the 90s ones" work only because the client replays
 * what came before — forget that and the agent has no idea what "ones" means.
 */

import type {
  BrowseResponse,
  ChatResponse,
  MovieDetail,
  PersonResponse,
  SimilarResponse,
  Turn,
} from '../types'

const BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

/** The server sends 20 turns max; sending more is a 422. */
const MAX_HISTORY_TURNS = 20

export class ApiError extends Error {
  // Declared and assigned explicitly rather than as a constructor parameter
  // property: this project builds with `erasableSyntaxOnly`, which rejects the
  // TS-only shorthand because it cannot be erased to plain JavaScript.
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { signal })
  if (!response.ok) {
    // 404 and 503 mean genuinely different things here (no such film vs. the
    // store is down), so the status is carried rather than flattened to a
    // message — callers branch on it.
    throw new ApiError(response.status, `GET ${path} failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const browse = (rows = 6, signal?: AbortSignal) =>
  get<BrowseResponse>(`/api/v1/browse?rows=${rows}`, signal)

export const movie = (tmdbId: number, signal?: AbortSignal) =>
  get<MovieDetail>(`/api/v1/movie/${tmdbId}`, signal)

export const similar = (tmdbId: number, limit = 12, signal?: AbortSignal) =>
  get<SimilarResponse>(`/api/v1/movie/${tmdbId}/similar?limit=${limit}`, signal)

export const person = (personId: number, signal?: AbortSignal) =>
  get<PersonResponse>(`/api/v1/person/${personId}`, signal)

export const health = (signal?: AbortSignal) =>
  get<{ status: 'ok' | 'degraded'; stores: Record<string, boolean>; catalogue_size: number | null }>(
    '/api/v1/health',
    signal,
  )

/**
 * One conversational turn.
 *
 * `history` is trimmed to the most recent 20 turns — the server's limit. Keeping
 * the RECENT end matters: a follow-up refers to what was just said, so if
 * anything has to be dropped it must be the oldest turns.
 */
export async function ask(
  message: string,
  history: Turn[],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history: history.slice(-MAX_HISTORY_TURNS) }),
    signal,
  })

  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new ApiError(response.status, detail || `Chat failed (${response.status})`)
  }
  return response.json() as Promise<ChatResponse>
}
