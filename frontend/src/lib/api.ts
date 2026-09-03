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
  ChatEvent,
  ChatResponse,
  MovieDetail,
  PersonResponse,
  SimilarResponse,
  StreamedSource,
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

/**
 * The same turn, read as it happens.
 *
 * Not `EventSource`, which is the browser's built-in SSE client: it only makes
 * GET requests, and a turn is a message plus up to 20 replayed turns — a body.
 * So the stream is read by hand, which costs about twenty lines and keeps the
 * `AbortSignal` wiring every other call in this file already uses.
 */
export async function* askStream(
  message: string,
  history: Turn[],
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${BASE}/api/v1/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ message, history: history.slice(-MAX_HISTORY_TURNS) }),
    signal,
  })

  // A rejected request still fails normally — validation happens before the
  // first frame, so there is a real status code to report here.
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new ApiError(response.status, detail || `Chat failed (${response.status})`)
  }
  if (!response.body) throw new ApiError(response.status, 'This browser cannot stream.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break

      // `stream: true` matters: a multi-byte character (an em dash, an accent)
      // can be split across two network chunks, and decoding each chunk
      // independently would render the halves as garbage.
      buffer += decoder.decode(value, { stream: true })

      // A blank line closes a message. Everything after the last one is a
      // half-arrived message and waits for the next chunk.
      for (;;) {
        const end = buffer.indexOf('\n\n')
        if (end === -1) break
        const frame = buffer.slice(0, end)
        buffer = buffer.slice(end + 2)
        const event = parseFrame(frame)
        if (event) yield event
      }
    }
  } finally {
    // Reached when the caller stops consuming (an abort, or a `break` in their
    // loop). Releases the connection so the server learns nobody is listening
    // and stops generating.
    await reader.cancel().catch(() => {})
  }
}

/** One SSE message -> a typed event, or null for anything we do not handle. */
function parseFrame(frame: string): ChatEvent | null {
  let name = 'message'
  const data: string[] = []

  for (const line of frame.split('\n')) {
    // Lines starting with ':' are comments. The server sends one immediately so
    // the headers arrive before the first real event.
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) name = line.slice(6).trim()
    else if (line.startsWith('data:')) data.push(line.slice(5).trim())
  }

  if (data.length === 0) return null

  let payload: Record<string, unknown>
  try {
    payload = JSON.parse(data.join('\n')) as Record<string, unknown>
  } catch {
    return null
  }

  switch (name) {
    case 'stage':
      return { type: 'stage', label: String(payload.label ?? '') }
    case 'token':
      return { type: 'token', text: String(payload.text ?? '') }
    case 'source':
      return { type: 'source', source: payload as unknown as StreamedSource }
    case 'done':
      return { type: 'done', result: payload as unknown as Omit<ChatResponse, 'sources'> }
    case 'error':
      return { type: 'error', detail: String(payload.detail ?? 'Something went wrong.') }
    default:
      return null
  }
}
