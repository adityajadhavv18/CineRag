/**
 * The overlay that opens on any card click.
 *
 * It holds a small NAVIGATION STACK rather than a single id, because a film and
 * a person are the same journey: open Training Day, click Denzel Washington,
 * click Man on Fire. One overlay that pushes and pops keeps that continuous —
 * closing the film modal to open a person modal would lose the thread and the
 * scroll position behind it.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronLeft, Close } from './icons'
import MovieDetailView from './MovieDetailView'
import PersonDetailView from './PersonDetailView'

export type View = { kind: 'movie'; id: number } | { kind: 'person'; id: number }

interface Props {
  /** The film the overlay opened on; null closes it. */
  filmId: number | null
  onClose: () => void
}

export default function DetailModal({ filmId, onClose }: Props) {
  const [stack, setStack] = useState<View[]>([])
  const panel = useRef<HTMLDivElement>(null)

  // A new film replaces the whole stack: opening something from the page behind
  // is a fresh journey, not a step deeper into the last one.
  useEffect(() => {
    setStack(filmId == null ? [] : [{ kind: 'movie', id: filmId }])
  }, [filmId])

  const push = useCallback((view: View) => setStack((s) => [...s, view]), [])
  const pop = useCallback(() => setStack((s) => s.slice(0, -1)), [])

  const open = stack.length > 0

  // An overlay that swallows the Back button is the single most common way a
  // modal misbehaves: on a phone, Back is how people close things, and here it
  // would leave the site entirely. Pushing one history entry while the overlay
  // is open makes Back close it instead — and only one entry, not one per view,
  // so a reader four films deep still leaves with a single press.
  useEffect(() => {
    if (!open) return

    window.history.pushState({ cineragModal: true }, '')
    const onPop = () => onClose()
    window.addEventListener('popstate', onPop)

    return () => {
      window.removeEventListener('popstate', onPop)
      // Closing by Escape or the X leaves our entry on the stack; drop it so
      // Back does not have to be pressed twice to get to the previous page.
      if (window.history.state?.cineragModal) window.history.back()
    }
  }, [open, onClose])

  // Focus moves into the overlay and returns to whatever opened it, so keyboard
  // and screen-reader users are not dropped back at the top of the document.
  useEffect(() => {
    if (!open) return
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    return () => opener?.focus?.()
  }, [open])

  useEffect(() => {
    if (!open) return

    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      // Escape unwinds one step at a time. Closing outright from three levels
      // deep would throw away a trail the reader deliberately followed.
      if (stack.length > 1) pop()
      else onClose()
    }

    // Locking the page behind stops the browser scrolling THAT when the modal's
    // own content reaches its end — the "scroll chaining" that makes an overlay
    // feel like it is sitting on quicksand.
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', onKey)

    return () => {
      document.body.style.overflow = previous
      window.removeEventListener('keydown', onKey)
    }
  }, [open, stack.length, pop, onClose])

  // Each view starts at its own top; without this, opening a person from
  // halfway down a film's page lands you halfway down theirs.
  useEffect(() => {
    panel.current?.scrollTo({ top: 0 })
  }, [stack.length])

  if (!open) return null
  const current = stack[stack.length - 1]

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto overscroll-contain bg-black/80 px-4 py-10 backdrop-blur-sm"
      // Only a click that both starts and ends on the backdrop closes it —
      // `onClick` alone fires when a text selection begun inside the panel is
      // released out here, closing the modal mid-drag.
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      role="dialog"
      aria-modal="true"
    >
      <div
        ref={panel}
        // -1 rather than 0: focusable by script so focus can be moved here on
        // open, but skipped by Tab so it never becomes a stop of its own.
        tabIndex={-1}
        className="relative mx-auto max-w-5xl animate-[modalIn_.32s_cubic-bezier(.16,1,.3,1)] overflow-hidden rounded-xl bg-surface shadow-2xl shadow-black/70 outline-none ring-1 ring-hairline"
      >
        <div className="absolute right-4 top-4 z-20 flex gap-2">
          {stack.length > 1 && (
            <button
              type="button"
              onClick={pop}
              aria-label="Back"
              className="grid size-9 place-items-center rounded-full bg-black/70 text-white transition hover:bg-black"
            >
              <ChevronLeft size={18} />
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="grid size-9 place-items-center rounded-full bg-black/70 text-white transition hover:bg-black"
          >
            <Close size={18} />
          </button>
        </div>

        {current.kind === 'movie' ? (
          <MovieDetailView key={`m${current.id}`} tmdbId={current.id} onNavigate={push} />
        ) : (
          <PersonDetailView key={`p${current.id}`} personId={current.id} onNavigate={push} />
        )}
      </div>
    </div>
  )
}
