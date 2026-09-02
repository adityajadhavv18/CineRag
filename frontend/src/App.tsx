/**
 * The browse page, and the state the whole app turns on.
 *
 * The conversation lives HERE, not in the chat panel, for one reason: asking a
 * question changes the page. Sources become the shelf and the carousel, so the
 * answer and the browse surface are two views of the same state and cannot be
 * owned by two components.
 *
 * `messages` is also the only record of the conversation anywhere — the server
 * is stateless (contract §1) and every request replays what came before.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import * as api from './lib/api'
import type { BrowseResponse, ChatResponse, MovieCard as Card, Turn } from './types'
import ChatDrawer, { type ChatMessage } from './components/ChatDrawer'
import DetailModal from './components/DetailModal'
import Hero from './components/Hero'
import NavBar from './components/NavBar'
import Row from './components/Row'
import { RowSkeleton } from './components/Skeletons'

export default function App() {
  const [shelf, setShelf] = useState<BrowseResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [openId, setOpenId] = useState<number | null>(null)

  const [chatOpen, setChatOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [pending, setPending] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  /** The last answer that actually produced films; what the shelf now shows. */
  const [results, setResults] = useState<ChatResponse | null>(null)
  const [highlightedId, setHighlightedId] = useState<number | null>(null)

  useEffect(() => {
    // Aborting on unmount stops StrictMode's double-invoked effect leaving a
    // second in-flight request to resolve into a stale render.
    const controller = new AbortController()

    api
      .browse(6, controller.signal)
      .then(setShelf)
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return
        setError(
          err instanceof api.ApiError
            ? `The API answered ${err.status}.`
            : 'Could not reach the API on :8000.',
        )
      })

    return () => controller.abort()
  }, [])

  const send = useCallback(
    async (text: string) => {
      // History is everything said BEFORE this message, which is exactly what
      // `messages` holds at this moment. Capturing it here rather than after the
      // optimistic append is what keeps the new question out of its own context.
      const history: Turn[] = messages.map(({ role, content }) => ({ role, content }))

      setMessages((m) => [...m, { role: 'user', content: text }])
      setPending(true)
      setChatError(null)

      try {
        const answer = await api.ask(text, history)
        setMessages((m) => [...m, { role: 'assistant', content: answer.response, response: answer }])

        // Only an answer with films replaces the shelf. A clarification or a
        // "hi" must leave the page alone — wiping the rows because the agent
        // asked a question would punish the user for being asked one.
        if (answer.sources.length > 0) {
          setResults(answer)
          window.scrollTo({ top: 0, behavior: 'smooth' })
        }
      } catch {
        setChatError('That question could not be answered right now. Try again?')
        // Drop the optimistic user turn: leaving it would put a message in the
        // replayed history that the agent never actually saw.
        setMessages((m) => m.slice(0, -1))
      } finally {
        setPending(false)
      }
    },
    [messages],
  )

  const reset = useCallback(() => {
    setMessages([])
    setResults(null)
    setChatError(null)
  }, [])

  const cite = useCallback((tmdbId: number) => {
    setHighlightedId(tmdbId)
    document
      .getElementById(`film-${tmdbId}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' })
    // The ring is a pointer, not a state — it should fade once it has done its
    // job of saying "this one".
    window.setTimeout(() => setHighlightedId((id) => (id === tmdbId ? null : id)), 2400)
  }, [])

  // Sources carry no genres or rating, so they are widened into cards with the
  // fields a tile needs left empty rather than faked.
  const resultRows = useMemo(() => {
    if (!results) return null

    const asCard = (s: ChatResponse['sources'][number]): Card => ({
      tmdb_id: s.tmdb_id,
      title: s.title,
      year: s.year,
      rating: null,
      poster_path: s.poster_path,
      backdrop_path: s.backdrop_path,
      overview: s.overview,
      genres: [],
      character: null,
    })

    return {
      films: results.sources.map(asCard),
      citations: new Map(results.sources.map((s, i) => [s.tmdb_id, i + 1])),
      franchises: results.franchise.filter((f) => f.films.length > 1),
    }
  }, [results])

  const heroes = resultRows?.films.filter((f) => f.backdrop_path) ?? shelf?.heroes ?? []

  return (
    <div className="min-h-full bg-ink pb-20">
      <NavBar onAsk={() => setChatOpen(true)} degraded={shelf?.degraded || results?.degraded} />

      {error ? (
        <ApiDown message={error} />
      ) : !shelf ? (
        <LoadingShelf />
      ) : (
        <>
          <Hero films={heroes} onOpen={setOpenId} />

          {/* Rows sit slightly over the hero so the banner reads as continuous
              with the shelf rather than as a separate block. */}
          <main className="relative z-10 -mt-16 space-y-2">
            {resultRows ? (
              <>
                <div className="flex items-center gap-3 px-4 pt-4 md:px-12">
                  <button
                    type="button"
                    onClick={() => setResults(null)}
                    className="rounded-full border border-white/25 px-3 py-1 text-xs transition hover:border-white/60"
                  >
                    ← Back to browsing
                  </button>
                  <span className="text-xs text-[color:var(--text-muted)]">
                    Showing what you asked for
                  </span>
                </div>

                <Row
                  title="Your results"
                  films={resultRows.films}
                  onOpen={setOpenId}
                  citations={resultRows.citations}
                  highlightedId={highlightedId}
                />

                {resultRows.franchises.map((franchise) => (
                  <Row
                    key={franchise.collection_id ?? franchise.collection_name}
                    title={franchise.collection_name ?? 'The collection'}
                    films={franchise.films.map((f) => ({
                      tmdb_id: f.tmdb_id,
                      title: f.title,
                      year: f.year,
                      rating: null,
                      poster_path: f.poster_path,
                      backdrop_path: null,
                      overview: null,
                      genres: [],
                      character: null,
                    }))}
                    onOpen={setOpenId}
                    seedId={franchise.films.find((f) => f.is_seed)?.tmdb_id}
                    highlightedId={highlightedId}
                  />
                ))}
              </>
            ) : (
              shelf.rows.map((row) => (
                <Row key={row.title} title={row.title} films={row.films} onOpen={setOpenId} />
              ))
            )}

            {!resultRows && shelf.rows.length === 0 && (
              <p className="px-12 pt-24 text-[color:var(--text-muted)]">
                {/* The distinction the `degraded` flag exists to preserve: an
                    empty shelf is not the same fact as an unreachable store. */}
                {shelf.degraded
                  ? 'The catalogue is temporarily unreachable. Nothing is missing — it just cannot be loaded right now.'
                  : 'The catalogue is empty. Has ingestion been run?'}
              </p>
            )}
          </main>
        </>
      )}

      <ChatDrawer
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        messages={messages}
        pending={pending}
        error={chatError}
        onSend={send}
        onReset={reset}
        onCite={cite}
        onHoverCite={setHighlightedId}
      />

      <DetailModal filmId={openId} onClose={() => setOpenId(null)} />
    </div>
  )
}

function LoadingShelf() {
  return (
    <>
      <div className="aspect-video min-h-[420px] w-full animate-pulse bg-surface" />
      <div className="relative z-10 -mt-16 space-y-2">
        {[0, 1, 2].map((i) => (
          <RowSkeleton key={i} />
        ))}
      </div>
    </>
  )
}

function ApiDown({ message }: { message: string }) {
  return (
    <div className="grid min-h-screen place-items-center px-6 text-center">
      <div className="max-w-md space-y-3">
        <h1 className="text-2xl font-bold">CineRAG cannot reach its backend</h1>
        <p className="text-sm text-[color:var(--text-muted)]">{message}</p>
        <pre className="overflow-x-auto rounded-lg bg-surface p-4 text-left text-xs text-white/70">
          {`docker compose up -d
uv run uvicorn server.main:app --reload`}
        </pre>
      </div>
    </div>
  )
}
