/**
 * A row tile, plus the preview that grows out of it on hover.
 *
 * Landscape rather than portrait because that is what the reference shows and
 * what the data supports: TMDB backdrops are 16:9 scene stills. They carry no
 * title text of their own (unlike Netflix's bespoke artwork), so the title is
 * drawn over the image — otherwise half the shelf is unidentifiable.
 *
 * The preview is rendered through a PORTAL, positioned from the card's bounding
 * box. It has to be: a row scrolls horizontally, and CSS will not give you
 * `overflow-x: auto` with `overflow-y: visible` — asking for one axis to scroll
 * silently clips the other. An in-flow preview would be cut off at the row's top
 * and bottom edges. Taking it out of the flow entirely sidesteps the whole
 * problem, at the cost of having to compute its position by hand.
 *
 * It is also only mounted while hovering. Rendering all 120 previews up front
 * and hiding them with CSS would put a hundred hidden panels in the tree for the
 * one that is ever visible.
 */

import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { MovieCard as Card } from '../types'
import Artwork from './Artwork'
import { ChevronDown, Play, Plus, ThumbUp } from './icons'

/** Long enough that dragging the cursor across a row doesn't strobe six
 *  previews, short enough that a deliberate hover feels deliberate. */
const HOVER_DELAY_MS = 400
/** Grow-in and fade-out. Long enough to read as an expansion of the card rather
 *  than a panel appearing on top of it. */
const ENTER_MS = 340
const EXIT_MS = 180
const PREVIEW_WIDTH = 340
const VIEWPORT_MARGIN = 12

interface Anchor {
  left: number
  top: number
  /** Where the card sits inside the preview, so it grows OUT of the card
   *  rather than out of the middle of a panel that is wider than it. */
  originX: number
}

interface Props {
  film: Card
  onOpen: (tmdbId: number) => void
  /** Marks the film a franchise timeline was seeded from. */
  isSeed?: boolean
  /** 1-based citation number, when this card is the referent of a [n] marker. */
  citation?: number
  /** Set while the matching [n] marker in the chat panel is hovered. */
  highlighted?: boolean
}

export default function MovieCard({ film, onOpen, isSeed, citation, highlighted }: Props) {
  const [anchor, setAnchor] = useState<Anchor | null>(null)
  // Mounting and *showing* are two steps. The panel mounts in its collapsed
  // state, then a frame later flips to its open state so the browser has
  // something to transition FROM — set both at once and CSS sees only the final
  // value, which is exactly the instant pop this replaces.
  const [open, setOpen] = useState(false)
  const timer = useRef<number | undefined>(undefined)
  const exitTimer = useRef<number | undefined>(undefined)
  const frame = useRef<number | undefined>(undefined)
  const ref = useRef<HTMLDivElement>(null)

  // Backdrops make the better landscape tile; a poster cropped to 16:9 is the
  // fallback, and a tinted placeholder is the fallback to that. All three live
  // in <Artwork>, along with recovering from a path the CDN no longer serves.
  const artwork = (
    <Artwork title={film.title} backdropPath={film.backdrop_path} posterPath={film.poster_path} />
  )

  useEffect(
    () => () => {
      window.clearTimeout(timer.current)
      window.clearTimeout(exitTimer.current)
      if (frame.current) cancelAnimationFrame(frame.current)
    },
    [],
  )

  // A portal is positioned in viewport coordinates, so anything that moves the
  // card underneath it — a row scroll, a window resize — leaves the preview
  // stranded. Closing on either is both simpler and less jarring than trying to
  // chase the card around the screen.
  useEffect(() => {
    if (!anchor) return
    // Dismissed outright rather than faded: the card has already moved out from
    // under the panel, so animating it closed would leave it hanging in empty
    // space for a third of a second.
    const dismiss = () => {
      window.clearTimeout(exitTimer.current)
      setOpen(false)
      setAnchor(null)
    }
    window.addEventListener('scroll', dismiss, true)
    window.addEventListener('resize', dismiss)
    return () => {
      window.removeEventListener('scroll', dismiss, true)
      window.removeEventListener('resize', dismiss)
    }
  }, [anchor])

  const enter = () => {
    window.clearTimeout(exitTimer.current)
    timer.current = window.setTimeout(() => {
      const box = ref.current?.getBoundingClientRect()
      if (!box) return
      // Centre on the card, then clamp so a card at either end of a row opens a
      // preview that is fully on screen rather than half past the edge.
      const centred = box.left + box.width / 2 - PREVIEW_WIDTH / 2
      const left = Math.min(
        Math.max(centred, VIEWPORT_MARGIN),
        window.innerWidth - PREVIEW_WIDTH - VIEWPORT_MARGIN,
      )
      setAnchor({ left, top: box.top - 28, originX: box.left + box.width / 2 - left })
      // Two frames, not one. A single rAF still lands in the same paint as the
      // mount in some browsers, and the transition is skipped.
      frame.current = requestAnimationFrame(() => {
        frame.current = requestAnimationFrame(() => setOpen(true))
      })
    }, HOVER_DELAY_MS)
  }

  const leave = () => {
    window.clearTimeout(timer.current)
    if (frame.current) cancelAnimationFrame(frame.current)
    setOpen(false)
    // Unmount only after the fade-out has actually run.
    exitTimer.current = window.setTimeout(() => setAnchor(null), EXIT_MS)
  }

  return (
    <div
      ref={ref}
      // The scroll target for a [n] citation in the chat panel. Prefixed because
      // a bare tmdb_id is not a valid CSS identifier to start with a digit.
      id={`film-${film.tmdb_id}`}
      className="relative w-[240px] shrink-0 scroll-mx-12"
      onMouseEnter={enter}
      onMouseLeave={leave}
    >
      <button
        type="button"
        onClick={() => onOpen(film.tmdb_id)}
        aria-label={`Open ${film.title}`}
        className={`relative block w-full aspect-video overflow-hidden rounded-md bg-surface-2
          outline-none transition ring-offset-2 ring-offset-ink
          focus-visible:ring-2 focus-visible:ring-white
          ${highlighted ? 'ring-2 ring-brand' : ''}`}
      >
        {artwork}

        {/* TMDB stills have no title burned in, so one is drawn here. */}
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 to-transparent px-3 pb-2 pt-8 text-left">
          <p className="truncate text-sm font-semibold drop-shadow">{film.title}</p>
        </div>

        {citation !== undefined && (
          <span className="absolute left-2 top-2 rounded bg-brand px-1.5 py-0.5 text-[11px] font-bold tabular-nums">
            {citation}
          </span>
        )}
        {isSeed && (
          <span className="absolute right-2 top-2 rounded bg-white/95 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-black">
            You asked
          </span>
        )}
      </button>

      {anchor &&
        createPortal(
          <div
            onMouseEnter={() => {
              window.clearTimeout(exitTimer.current)
              setOpen(true)
            }}
            onMouseLeave={leave}
            style={{
              left: anchor.left,
              top: anchor.top,
              width: PREVIEW_WIDTH,
              // Grow from the card's centre, and from its baseline, so the panel
              // reads as the card expanding rather than as a box fading in over it.
              transformOrigin: `${anchor.originX}px 40%`,
              transform: open ? 'scale(1)' : 'scale(0.72)',
              opacity: open ? 1 : 0,
              // Slower and eased-out on the way in, quicker on the way out: an
              // opening should feel considered, a dismissal should feel finished.
              transition: open
                ? `transform ${ENTER_MS}ms cubic-bezier(.16,1,.3,1), opacity ${ENTER_MS / 2}ms ease-out`
                : `transform ${EXIT_MS}ms ease-in, opacity ${EXIT_MS}ms ease-in`,
              willChange: 'transform, opacity',
            }}
            className="fixed z-50 overflow-hidden rounded-lg bg-surface shadow-2xl shadow-black/80 ring-1 ring-hairline"
          >
            <button
              type="button"
              onClick={() => onOpen(film.tmdb_id)}
              className="block w-full aspect-video overflow-hidden"
              aria-label={`Open ${film.title}`}
            >
              {artwork}
            </button>

            <div className="space-y-2 p-3">
              <div className="flex items-center gap-2">
                <span className="grid size-8 place-items-center rounded-full bg-white text-black">
                  <Play size={13} />
                </span>
                <span className="grid size-8 place-items-center rounded-full border border-white/40 text-white/80">
                  <Plus size={14} />
                </span>
                <span className="grid size-8 place-items-center rounded-full border border-white/40 text-white/80">
                  <ThumbUp size={14} />
                </span>
                <button
                  type="button"
                  onClick={() => onOpen(film.tmdb_id)}
                  aria-label={`More about ${film.title}`}
                  className="ml-auto grid size-8 place-items-center rounded-full border border-white/40 text-white/80 transition hover:border-white hover:text-white"
                >
                  <ChevronDown size={14} />
                </button>
              </div>

              <p className="text-sm font-semibold">{film.title}</p>

              <div className="flex flex-wrap items-center gap-x-2 text-xs text-[color:var(--text-muted)]">
                {film.rating != null && (
                  <span className="font-semibold text-emerald-400">{film.rating.toFixed(1)}</span>
                )}
                {film.year != null && <span>{film.year}</span>}
              </div>

              {film.genres.length > 0 && (
                <p className="text-xs text-[color:var(--text-muted)]">
                  {film.genres.slice(0, 3).join(' • ')}
                </p>
              )}
            </div>
          </div>,
          document.body,
        )}
    </div>
  )
}
