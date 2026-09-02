/**
 * The panel that asks the agent things.
 *
 * No scrim behind it, on purpose. A citation marker in here scrolls to and
 * highlights a card out on the page, so the page has to stay visible and
 * clickable — a modal overlay would break the one interaction that makes a
 * cited answer worth having.
 *
 * The message list is the SOURCE OF TRUTH for the conversation, because the
 * server keeps none (contract §1). Every request replays what came before, and
 * that is the only reason a follow-up like "only the 90s ones" resolves.
 */

import { useEffect, useRef, useState } from 'react'
import type { ChatResponse } from '../types'
import { INTENTS_WITHOUT_FILMS } from '../types'
import { RichText } from '../lib/richText'
import { Close, Send, Sparkle } from './icons'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  /** Present on assistant turns: the full response, for chips and citations. */
  response?: ChatResponse
}

/** Shown on an empty panel. Each exercises a different path through the agent,
 *  which is the point — this is the demo surface for the whole build. */
const STARTERS = [
  'gritty crime dramas starring Denzel Washington',
  'recommend the Harry Potter films',
  'tell me about Inception',
  'something horror',
]

interface Props {
  open: boolean
  onClose: () => void
  messages: ChatMessage[]
  pending: boolean
  error: string | null
  onSend: (message: string) => void
  onReset: () => void
  /** Scroll the page to a cited film and flash it. */
  onCite: (tmdbId: number) => void
  onHoverCite: (tmdbId: number | null) => void
}

export default function ChatDrawer({
  open,
  onClose,
  messages,
  pending,
  error,
  onSend,
  onReset,
  onCite,
  onHoverCite,
}: Props) {
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (open) input.current?.focus()
  }, [open])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, pending])

  const submit = () => {
    const text = draft.trim()
    // The server rejects blank and anything over 1000 chars with a 422; there is
    // no reason to spend a round trip discovering that.
    if (!text || text.length > 1000 || pending) return
    onSend(text)
    setDraft('')
  }

  return (
    <aside
      aria-hidden={!open}
      className={`fixed right-0 top-0 z-40 flex h-full w-full flex-col border-l border-hairline bg-surface shadow-2xl shadow-black/60 transition-transform duration-300 ease-out sm:w-[420px] ${
        open ? 'translate-x-0' : 'pointer-events-none translate-x-full'
      }`}
    >
      <header className="flex items-center gap-2 border-b border-hairline px-4 py-3">
        <Sparkle size={16} className="text-brand" />
        <h2 className="font-semibold">Ask CineRAG</h2>

        {messages.length > 0 && (
          <button
            type="button"
            onClick={onReset}
            className="ml-auto text-xs text-[color:var(--text-muted)] transition hover:text-white"
          >
            Clear
          </button>
        )}

        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className={`grid size-8 place-items-center rounded-full text-white/70 transition hover:bg-white/10 hover:text-white ${
            messages.length > 0 ? '' : 'ml-auto'
          }`}
        >
          <Close size={16} />
        </button>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto overscroll-contain p-4">
        {messages.length === 0 && (
          <div className="space-y-3 pt-4">
            <p className="text-sm text-[color:var(--text-muted)]">
              Ask in your own words. Films come back on the page behind this panel.
            </p>
            {STARTERS.map((starter) => (
              <button
                key={starter}
                type="button"
                onClick={() => onSend(starter)}
                className="block w-full rounded-lg border border-hairline bg-surface-2 px-3 py-2 text-left text-sm transition hover:border-white/30"
              >
                {starter}
              </button>
            ))}
          </div>
        )}

        {messages.map((message, i) =>
          message.role === 'user' ? (
            <p
              key={i}
              className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-sm bg-brand px-3.5 py-2 text-sm"
            >
              {message.content}
            </p>
          ) : (
            <AssistantTurn
              key={i}
              message={message}
              onCite={onCite}
              onHoverCite={onHoverCite}
            />
          ),
        )}

        {pending && (
          <div className="flex gap-1.5 px-1 py-2" aria-label="Thinking">
            {[0, 1, 2].map((dot) => (
              <span
                key={dot}
                className="size-2 animate-bounce rounded-full bg-white/40"
                style={{ animationDelay: `${dot * 120}ms` }}
              />
            ))}
          </div>
        )}

        {error && (
          <p className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </p>
        )}

        <div ref={bottom} />
      </div>

      <div className="border-t border-hairline p-3">
        <div className="flex items-end gap-2 rounded-xl bg-surface-2 p-2">
          <textarea
            ref={input}
            rows={1}
            value={draft}
            maxLength={1000}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter breaks the line — the convention for a
              // chat box, and the reason this is a textarea and not an input.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder="A film like Heat, but funnier…"
            className="max-h-32 flex-1 resize-none bg-transparent px-1 text-sm outline-none placeholder:text-white/35"
          />
          <button
            type="button"
            onClick={submit}
            disabled={!draft.trim() || pending}
            aria-label="Send"
            className="grid size-8 shrink-0 place-items-center rounded-full bg-brand text-white transition disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </aside>
  )
}

function AssistantTurn({
  message,
  onCite,
  onHoverCite,
}: {
  message: ChatMessage
  onCite: (tmdbId: number) => void
  onHoverCite: (tmdbId: number | null) => void
}) {
  const [showTrace, setShowTrace] = useState(false)
  const answer = message.response

  return (
    <div className="space-y-2">
      <RichText
        text={message.content}
        renderCitation={(index) => {
          // 1-based into `sources`. A marker past the end of the list is an
          // agent slip, not something to render as a dead control.
          const source = answer?.sources[index - 1]
          if (!source) return null
          return (
            <button
              type="button"
              onClick={() => onCite(source.tmdb_id)}
              onMouseEnter={() => onHoverCite(source.tmdb_id)}
              onMouseLeave={() => onHoverCite(null)}
              title={source.title}
              className="mx-0.5 rounded bg-brand/25 px-1.5 align-baseline text-[11px] font-bold text-brand transition hover:bg-brand hover:text-white"
            >
              {index}
            </button>
          )
        }}
      />

      {answer && (
        <div className="flex flex-wrap items-center gap-1.5 pt-1 text-[11px]">
          {answer.intent && (
            <span className="rounded-full bg-white/10 px-2 py-0.5 text-white/60">
              {answer.intent}
            </span>
          )}
          {answer.lead_engine && (
            <span className="rounded-full bg-white/10 px-2 py-0.5 text-white/60">
              {answer.lead_engine}
            </span>
          )}

          {/* An empty source list is CORRECT here — the agent is asking a
              question or making small talk, not failing to find films. Saying
              "no results" would be a lie about what just happened. */}
          {answer.sources.length === 0 && !INTENTS_WITHOUT_FILMS.has(answer.intent ?? '') && (
            <span className="rounded-full bg-white/10 px-2 py-0.5 text-white/60">
              {answer.degraded ? 'data unavailable' : 'nothing matched'}
            </span>
          )}

          {answer.trace.length > 0 && (
            <button
              type="button"
              onClick={() => setShowTrace((v) => !v)}
              className="rounded-full bg-white/10 px-2 py-0.5 text-white/60 transition hover:text-white"
            >
              {showTrace ? 'hide path' : 'path'}
            </button>
          )}
        </div>
      )}

      {answer?.degraded && (
        <p className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          A data store was unreachable, so this answer may be thinner than usual.
        </p>
      )}

      {showTrace && answer && (
        <ol className="space-y-0.5 rounded-lg bg-black/40 p-2.5 font-mono text-[11px] text-white/50">
          {answer.trace.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
      )}
    </div>
  )
}
