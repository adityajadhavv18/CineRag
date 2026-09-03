/**
 * The panel that asks the agent things.
 *
 * No scrim behind it, on purpose. A citation marker in here scrolls to and
 * highlights a card out on the page, so the page has to stay visible and
 * clickable — a modal overlay would break the one interaction that makes a
 * cited answer worth having.
 *
 * For the same reason the page RESERVES this panel's width rather than being
 * overlapped by it (`--chat-inset`, published below). Sitting on top of a
 * full-width page, the panel covered the right-hand end of every shelf, its
 * "scroll right" arrow, and any card a citation scrolled to — the answer
 * pointed at films that could only be seen by closing the answer.
 *
 * The message list is the SOURCE OF TRUTH for the conversation, because the
 * server keeps none (contract §1). Every request replays what came before, and
 * that is the only reason a follow-up like "only the 90s ones" resolves.
 */

import { useEffect, useRef, useState } from 'react'
import type { ChatResponse } from '../types'
import { INTENTS_WITHOUT_FILMS } from '../types'
import { RichText } from '../lib/richText'
import ClarifyCard from './ClarifyCard'
import { Close, Send, Sparkle } from './icons'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  /**
   * Present on assistant turns: the full response, for chips and citations.
   *
   * Absent while an answer is still streaming — the text arrives before the
   * intent and the trace do, so a turn with content and no `response` is a
   * normal, temporary state rather than a broken one.
   */
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

/**
 * Panel width, in px, and the bounds a drag is allowed to reach.
 *
 * The width only applies from `sm` up. Below that the panel is full-bleed and
 * there is no page beside it to make room for, so there is nothing to resize.
 */
const MIN_WIDTH = 320
const DEFAULT_WIDTH = 420
const WIDTH_KEY = 'cinerag:chat-width'
/** Leave a strip of page showing — a panel dragged flush to the left edge hides
 *  the very cards its citations point at. */
const maxWidth = () => Math.max(MIN_WIDTH, Math.min(920, window.innerWidth - 64))

/** localStorage throws outright in some privacy modes, so neither side of this
 *  is allowed to take the panel down with it. */
const readWidth = () => {
  try {
    const saved = Number(localStorage.getItem(WIDTH_KEY))
    if (Number.isFinite(saved) && saved >= MIN_WIDTH) return saved
  } catch {
    /* fall through to the default */
  }
  return DEFAULT_WIDTH
}

interface Props {
  open: boolean
  onClose: () => void
  messages: ChatMessage[]
  pending: boolean
  /** What the agent is doing right now, in plain words. Null once text starts. */
  stage: string | null
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
  stage,
  error,
  onSend,
  onReset,
  onCite,
  onHoverCite,
}: Props) {
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)

  const [width, setWidth] = useState(readWidth)
  const [resizing, setResizing] = useState(false)
  // Tracks the same breakpoint as the `sm:` classes below. Under it the width
  // is left to CSS, so a width dragged on a desktop cannot leak into the
  // full-bleed mobile layout.
  const [resizable, setResizable] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(min-width: 640px)').matches,
  )

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 640px)')
    const sync = () => setResizable(mq.matches)
    // Re-clamping on every window resize is what stops a width saved on a wide
    // monitor from stranding the panel off-screen in a narrow window.
    const clamp = () => setWidth((w) => Math.max(MIN_WIDTH, Math.min(w, maxWidth())))

    clamp()
    mq.addEventListener('change', sync)
    window.addEventListener('resize', clamp)
    return () => {
      mq.removeEventListener('change', sync)
      window.removeEventListener('resize', clamp)
    }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(WIDTH_KEY, String(width))
    } catch {
      /* a width that does not survive a reload is still a working panel */
    }
  }, [width])

  /**
   * Tell the page how much room to leave on the right.
   *
   * A CSS variable rather than a prop: the layout below reads it in CSS, so
   * dragging the edge reflows the page without re-rendering a shelf of a
   * hundred tiles on every pointer move.
   *
   * It is deliberately not transitioned. Animating the page's padding would
   * mean 300ms of continuous layout while the panel slides, and the panel is
   * sliding into a strip of ground the same colour as the page.
   */
  useEffect(() => {
    const root = document.documentElement
    // Only when the panel is beside the page. Full-bleed on mobile, it covers
    // the page outright and there is nothing to make room for.
    root.style.setProperty('--chat-inset', open && resizable ? `${width}px` : '0px')
    return () => root.style.setProperty('--chat-inset', '0px')
  }, [open, resizable, width])

  /**
   * Drag from the left edge.
   *
   * Listeners go on the window rather than the handle so the drag keeps
   * tracking once the pointer outruns a 6px strip — which it does immediately.
   * The pointer moving LEFT widens the panel, hence `start - current`.
   */
  const startResize = (event: React.PointerEvent) => {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = width
    const limit = maxWidth()
    setResizing(true)

    const move = (e: PointerEvent) =>
      setWidth(Math.max(MIN_WIDTH, Math.min(startWidth + (startX - e.clientX), limit)))
    const stop = () => {
      setResizing(false)
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', stop)
      window.removeEventListener('pointercancel', stop)
    }

    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', stop)
    window.addEventListener('pointercancel', stop)
  }

  /** The same resize without a mouse — the handle is a focusable separator. */
  const nudge = (event: React.KeyboardEvent) => {
    if (event.key === 'Home') {
      event.preventDefault()
      setWidth(DEFAULT_WIDTH)
      return
    }
    const step = event.shiftKey ? 64 : 16
    const delta = event.key === 'ArrowLeft' ? step : event.key === 'ArrowRight' ? -step : 0
    if (!delta) return
    event.preventDefault()
    setWidth((w) => Math.max(MIN_WIDTH, Math.min(w + delta, maxWidth())))
  }

  // A request in flight always ends with the assistant turn it is filling, so
  // that pair is the whole test. Deliberately NOT "has no response yet": the
  // first citation attaches one long before the answer is finished, which put
  // the thinking dots back underneath a half-written reply.
  const last = messages[messages.length - 1]
  const streaming = pending && last?.role === 'assistant'
  // Nothing written yet, so the dots are still the only thing to show.
  const waiting = streaming && last.content.length === 0

  useEffect(() => {
    if (open) input.current?.focus()
  }, [open])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
    // Also follows the text as it grows, not just when a turn is added —
    // otherwise a long answer streams straight off the bottom of the panel.
  }, [messages.length, last?.content, pending])

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
      // Only `transition-transform`, so a drag tracks the pointer exactly —
      // animating width would put the edge behind the cursor for 300ms.
      style={resizable ? { width } : undefined}
      className={`fixed right-0 top-0 z-40 flex h-full w-full flex-col border-l border-hairline bg-surface shadow-2xl shadow-black/60 transition-transform duration-300 ease-out sm:w-[420px] ${
        open ? 'translate-x-0' : 'pointer-events-none translate-x-full'
      } ${resizing ? 'select-none' : ''}`}
    >
      {resizable && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize panel"
          aria-valuenow={width}
          aria-valuemin={MIN_WIDTH}
          // Not a tab stop while the panel is closed — it is off-screen and
          // inside an `aria-hidden` subtree.
          tabIndex={open ? 0 : -1}
          onPointerDown={startResize}
          onKeyDown={nudge}
          // Double-click is the usual escape hatch from a width you have
          // dragged somewhere unhelpful.
          onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
          // `touch-none` keeps a touch drag from scrolling the page instead.
          // The grab strip straddles the border so the edge itself is the
          // target, which is where a hand goes looking for it.
          className={`absolute left-0 top-0 z-10 h-full w-1.5 -translate-x-1/2 cursor-col-resize touch-none transition-colors hover:bg-brand focus-visible:bg-brand focus-visible:outline-none ${
            resizing ? 'bg-brand' : 'bg-transparent'
          }`}
        />
      )}

      <header className="flex items-center gap-2 border-b border-hairline px-4 py-3">
        <Sparkle size={16} className="text-brand" />
        <h2 className="font-semibold">Ask Netflix</h2>

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
              // The caret belongs to the turn being written, and only while it
              // is genuinely mid-sentence — a finished answer wearing a blinking
              // cursor reads as an answer that stopped short.
              writing={streaming && i === messages.length - 1 && message.content.length > 0}
              // Only the newest question is still open. An older clarification
              // has already been answered — by the user turn directly below it —
              // and leaving its options clickable would offer to re-ask a
              // question the conversation has moved past.
              live={i === messages.length - 1 && !pending}
              onSend={onSend}
              onCite={onCite}
              onHoverCite={onHoverCite}
            />
          ),
        )}

        {/* Dots until the first word lands, then the text speaks for itself.
            `waiting` is what makes them disappear the moment writing starts —
            leaving them under a half-written answer would say the agent is
            about to start something it is already doing. */}
        {(waiting || (pending && !streaming)) && (
          <div className="space-y-2 px-1 py-2">
            <div className="flex gap-1.5" aria-hidden="true">
              {[0, 1, 2].map((dot) => (
                <span
                  key={dot}
                  className="size-2 animate-bounce rounded-full bg-white/40"
                  style={{ animationDelay: `${dot * 120}ms` }}
                />
              ))}
            </div>
            {stage && (
              // aria-live so the wait is narrated rather than being a silent
              // pause for anyone not watching the dots.
              <p aria-live="polite" className="text-xs italic text-[color:var(--text-muted)]">
                {stage}
              </p>
            )}
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
  writing,
  live,
  onSend,
  onCite,
  onHoverCite,
}: {
  message: ChatMessage
  writing?: boolean
  /** This is the newest turn and nothing is in flight, so its questions (if it
   *  asked any) are still the open ones. */
  live?: boolean
  onSend: (message: string) => void
  onCite: (tmdbId: number) => void
  onHoverCite: (tmdbId: number | null) => void
}) {
  const [showTrace, setShowTrace] = useState(false)
  const answer = message.response

  // When the agent asked grounded narrowing questions, they are rendered as
  // choices and the prose version is not shown at all — `lead` is the same
  // opening sentence without the flattened option lists underneath it. The full
  // prose still lives in `message.content`, which is what history replays, so
  // the agent knows exactly what it offered.
  const clarify = answer?.clarification ?? null
  const text = clarify ? clarify.lead : message.content

  return (
    <div className="space-y-2">
      <RichText
        // The caret rides along inside the text so it lands on the last line
        // rather than in a block of its own. It sits flush against the last
        // character because the server holds trailing whitespace back until it
        // knows more is coming — so `content` never ends mid-gap.
        text={writing ? `${text}▌` : text}
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

      {clarify && clarify.questions.length > 0 && live && (
        <ClarifyCard
          clarification={clarify}
          // Straight down the ordinary send path: the picks become a sentence
          // and that sentence is the next turn, which is why this needs no new
          // endpoint and leaves the server as stateless as it was.
          onSubmit={onSend}
        />
      )}

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
