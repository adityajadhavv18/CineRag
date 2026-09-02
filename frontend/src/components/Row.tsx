/**
 * One horizontal shelf: a heading and a scroller.
 *
 * The arrows page by a viewport-width at a time rather than by a fixed number of
 * cards, so the same click feels right on a laptop and on a wide monitor. They
 * hide at each end — an arrow that does nothing is worse than no arrow.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { MovieCard as Card } from '../types'
import MovieCard from './MovieCard'
import { ChevronLeft, ChevronRight } from './icons'

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

  const measure = useCallback(() => {
    const el = scroller.current
    if (!el) return
    setAtStart(el.scrollLeft < 8)
    // The 8px slack absorbs sub-pixel rounding, which otherwise leaves the
    // right-hand arrow visible on a row that is already fully scrolled.
    setAtEnd(el.scrollLeft + el.clientWidth >= el.scrollWidth - 8)
  }, [])

  useEffect(() => {
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [measure, films])

  const page = (direction: 1 | -1) => {
    const el = scroller.current
    if (!el) return
    el.scrollBy({ left: direction * (el.clientWidth * 0.9), behavior: 'smooth' })
  }

  if (films.length === 0) return null

  return (
    <section className="group/row relative py-3">
      <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight md:px-12">{title}</h2>

      <div
        ref={scroller}
        onScroll={measure}
        className="no-scrollbar flex gap-2 overflow-x-auto scroll-smooth px-4 md:px-12"
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
