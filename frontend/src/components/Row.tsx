/**
 * One horizontal shelf: a heading and a scroller.
 *
 * The arrows page by a viewport-width at a time rather than by a fixed number of
 * cards, so the same click feels right on a laptop and on a wide monitor. They
 * hide at each end — an arrow that does nothing is worse than no arrow.
 *
 * Everything about the scrolling here is written to keep the main thread free
 * during a gesture, because a trackpad fires scroll events faster than the
 * screen repaints and this row can hold a hundred tiles:
 *
 *   - No `scroll-behavior: smooth` on the scroller. That property applies to
 *     wheel input too, so every tick was retargeting a browser animation that
 *     had not finished the last one — the stutter it looks like it should fix.
 *     Arrow clicks get their own eased tween instead.
 *   - Nothing calls setState per scroll event. Edge state changes twice per
 *     row (at each end), and that is the only time this component re-renders.
 *   - Tiles stop accepting the pointer while the row is in motion, set as a
 *     DOM attribute rather than React state so the tiles are not re-rendered.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { MovieCard as Card } from '../types'
import MovieCard from './MovieCard'
import { ChevronLeft, ChevronRight } from './icons'

/** How long an arrow click takes to travel. The browser's own smooth scroll is
 *  quicker and flatter; a shelf of posters reads better gliding to a stop. */
const PAGE_MS = 620
/** Quiet time after the last scroll event before the row counts as settled. */
const IDLE_MS = 140
/** Ease-out cubic: leaves at speed, arrives gently. */
const ease = (t: number) => 1 - (1 - t) ** 3

interface Props {
  title: string
  films: Card[]
  onOpen: (tmdbId: number) => void
  /** tmdb_id -> 1-based citation number, for rows built from a chat answer. */
  citations?: Map<number, number>
  seedId?: number
  highlightedId?: number | null
}

export default function Row({ title, films, onOpen, citations, seedId, highlightedId }: Props) {
  const scroller = useRef<HTMLDivElement>(null)
  const [atStart, setAtStart] = useState(true)
  const [atEnd, setAtEnd] = useState(false)

  /** Mirrors the two pieces of edge state, so a scroll can tell whether an
   *  arrow needs to change without asking React and without re-rendering. */
  const edges = useRef({ atStart: true, atEnd: false })
  const measuring = useRef<number | null>(null)
  const settle = useRef<number | undefined>(undefined)
  const tween = useRef<number | null>(null)

  const measure = useCallback(() => {
    const el = scroller.current
    if (!el) return
    const start = el.scrollLeft < 8
    // The 8px slack absorbs sub-pixel rounding, which otherwise leaves the
    // right-hand arrow visible on a row that is already fully scrolled.
    const end = el.scrollLeft + el.clientWidth >= el.scrollWidth - 8

    if (start !== edges.current.atStart) {
      edges.current.atStart = start
      setAtStart(start)
    }
    if (end !== edges.current.atEnd) {
      edges.current.atEnd = end
      setAtEnd(end)
    }
  }, [])

  useEffect(() => {
    const el = scroller.current
    if (!el) return
    measure()

    // Observing the row rather than listening for a window resize: the row also
    // narrows when the chat panel opens or its edge is dragged, and neither of
    // those is a window resize. Missing them left the arrows describing a width
    // the row no longer had.
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [measure, films])

  // Timers and frames outlive a render, so they have to be cancelled by hand
  // when the row goes away — a shelf is replaced wholesale every time a chat
  // answer lands.
  useEffect(
    () => () => {
      window.clearTimeout(settle.current)
      if (measuring.current !== null) cancelAnimationFrame(measuring.current)
      if (tween.current !== null) cancelAnimationFrame(tween.current)
    },
    [],
  )

  const onScroll = () => {
    const el = scroller.current
    if (!el) return

    // Written straight to the DOM, on purpose. Cards sliding under a cursor
    // that has not moved fire mouseenter after mouseenter, and each one arms a
    // hover preview that the next scroll event tears back down — mount,
    // measure, dismiss, repeat, all mid-gesture. Suppressing that through
    // React state would instead re-render every tile as the row starts moving.
    el.dataset.scrolling = 'true'
    window.clearTimeout(settle.current)
    settle.current = window.setTimeout(() => {
      el.dataset.scrolling = 'false'
    }, IDLE_MS)

    // One measurement per frame rather than one per event.
    if (measuring.current === null) {
      measuring.current = requestAnimationFrame(() => {
        measuring.current = null
        measure()
      })
    }
  }

  /** Any real input outranks an arrow's animation — a tween that keeps running
   *  under a hand on the trackpad is the fight this whole file avoids. */
  const stopTween = () => {
    if (tween.current !== null) {
      cancelAnimationFrame(tween.current)
      tween.current = null
    }
  }

  const page = (direction: 1 | -1) => {
    const el = scroller.current
    if (!el) return
    stopTween()

    const from = el.scrollLeft
    const furthest = el.scrollWidth - el.clientWidth
    // Clamped up front so the easing runs over the distance actually available
    // — otherwise the last page of a row eases towards a position it cannot
    // reach and appears to stall early.
    const to = Math.max(0, Math.min(from + direction * el.clientWidth * 0.9, furthest))
    if (to === from) return

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.scrollLeft = to
      return
    }

    const began = performance.now()
    const step = (now: number) => {
      const t = Math.min((now - began) / PAGE_MS, 1)
      el.scrollLeft = from + (to - from) * ease(t)
      tween.current = t < 1 ? requestAnimationFrame(step) : null
    }
    tween.current = requestAnimationFrame(step)
  }

  if (films.length === 0) return null

  return (
    <section className="group/row relative py-3">
      <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight md:px-12">{title}</h2>

      <div
        ref={scroller}
        onScroll={onScroll}
        onWheel={stopTween}
        onTouchStart={stopTween}
        onPointerDown={stopTween}
        // `overscroll-x-contain` keeps a flick past either end from turning
        // into the browser's swipe-back gesture, which is the other thing that
        // made these rows feel like they were snagging on something.
        className="no-scrollbar flex gap-2 overflow-x-auto overscroll-x-contain px-4 md:px-12"
      >
        {films.map((film) => (
          <MovieCard
            key={film.tmdb_id}
            film={film}
            onOpen={onOpen}
            isSeed={seedId === film.tmdb_id}
            citation={citations?.get(film.tmdb_id)}
            highlighted={highlightedId === film.tmdb_id}
          />
        ))}
      </div>

      {!atStart && (
        <button
          type="button"
          onClick={() => page(-1)}
          aria-label={`Scroll ${title} left`}
          className="absolute left-0 top-12 bottom-3 z-20 hidden w-10 place-items-center bg-black/50 text-white opacity-0 transition hover:bg-black/70 group-hover/row:opacity-100 md:grid"
        >
          <ChevronLeft size={24} />
        </button>
      )}
      {!atEnd && (
        <button
          type="button"
          onClick={() => page(1)}
          aria-label={`Scroll ${title} right`}
          className="absolute right-0 top-12 bottom-3 z-20 hidden w-10 place-items-center bg-black/50 text-white opacity-0 transition hover:bg-black/70 group-hover/row:opacity-100 md:grid"
        >
          <ChevronRight size={24} />
        </button>
      )}
    </section>
  )
}
