/**
 * The full-bleed banner: a six-film carousel.
 *
 * Two scrims, not one: a bottom fade so the rows underneath emerge out of the
 * image instead of butting against a hard edge, and a left fade so white title
 * text stays legible over whatever happens to be in that corner of the still.
 * Without the left one, a bright scene makes the title disappear.
 *
 * Slides are stacked and cross-faded rather than swapped, because swapping the
 * `src` of one <img> shows a blank frame while the next backdrop downloads. All
 * six are in the DOM from the start; they are ~200KB each and it buys an
 * instant, flicker-free transition.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { MovieCard } from '../types'
import Artwork from './Artwork'
import { Info, Play } from './icons'

const SLIDE_MS = 8000

interface Props {
  films: MovieCard[]
  onOpen: (tmdbId: number) => void
}

export default function Hero({ films, onOpen }: Props) {
  const [index, setIndex] = useState(0)
  const [paused, setPaused] = useState(false)
  // Autoplay is motion the user did not ask for, so honour the OS setting: with
  // reduced motion on, the carousel becomes a static first slide plus manual
  // controls rather than something that moves on its own.
  const reducedMotion = useRef(
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )

  const go = useCallback(
    (next: number) => setIndex(((next % films.length) + films.length) % films.length),
    [films.length],
  )

  useEffect(() => {
    if (paused || reducedMotion.current || films.length < 2) return
    const id = window.setInterval(() => setIndex((i) => (i + 1) % films.length), SLIDE_MS)
    return () => window.clearInterval(id)
  }, [paused, films.length])

  if (films.length === 0) return null
  const active = films[index]

  return (
    <header
      // 16:9 — the backdrop's own aspect ratio, so the frame is shown whole and
      // nothing is cropped. Height follows the viewport width rather than its
      // height, which is why this is `aspect-video` and not a `vh` value.
      //
      // `min-h` is the narrow-screen escape hatch: below ~750px wide, 16:9 is
      // too short to hold the title and buttons, so the banner stops shrinking
      // and the frame is trimmed at the sides instead.
      className="relative aspect-video min-h-[420px] w-full overflow-hidden"
      // Pausing on hover is what makes the buttons usable: a slide that changes
      // under the cursor sends the click to a different film than the one aimed at.
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      aria-roledescription="carousel"
      aria-label="Featured films"
    >
      {films.map((film, i) => (
        <div
          key={film.tmdb_id}
          aria-hidden={i !== index}
          className={`absolute inset-0 transition-opacity duration-700 ease-out ${
            i === index ? 'opacity-100' : 'opacity-0'
          }`}
        >
          <Artwork
            title={film.title}
            backdropPath={film.backdrop_path}
            posterPath={film.poster_path}
            backdropSize="original"
            // The first slide is the largest thing on the page and is visible
            // immediately; the rest can wait their turn.
            eager={i === 0}
            // At 16:9 `cover` crops nothing — the container is the image's own
            // shape, so this is an exact fit. It only ever crops on narrow
            // screens, where `min-h` holds the banner open and the frame is
            // trimmed at the left and right edges instead.
            className="h-full w-full object-cover"
          />
        </div>
      ))}

      <div className="scrim-left absolute inset-0" />
      <div className="scrim-bottom absolute inset-x-0 bottom-0 h-2/3" />

      {/* Keyed on the film so the copy re-mounts and fades in with each slide,
          rather than the text swapping instantly under a still-fading image. */}
      <div
        key={active.tmdb_id}
        className="absolute bottom-20 left-4 max-w-xl animate-[fadeUp_.6s_ease-out] space-y-4 md:left-12"
      >
        <h1 className="text-4xl font-black leading-tight drop-shadow-lg md:text-6xl">
          {active.title}
        </h1>

        <div className="flex flex-wrap items-center gap-x-3 text-sm font-medium">
          {active.rating != null && (
            <span className="text-emerald-400">{active.rating.toFixed(1)} rating</span>
          )}
          {active.year != null && <span>{active.year}</span>}
          {active.genres.length > 0 && (
            <span className="text-[color:var(--text-muted)]">
              {active.genres.slice(0, 3).join(' • ')}
            </span>
          )}
        </div>

        {active.overview && (
          <p className="line-clamp-3 text-sm text-white/85 drop-shadow md:text-base">
            {active.overview}
          </p>
        )}

        <div className="flex gap-3 pt-1">
          <button
            type="button"
            onClick={() => onOpen(active.tmdb_id)}
            className="flex items-center gap-2 rounded bg-white px-6 py-2.5 font-semibold text-black transition hover:bg-white/85"
          >
            <Play size={18} />
            Play
          </button>
          <button
            type="button"
            onClick={() => onOpen(active.tmdb_id)}
            className="flex items-center gap-2 rounded bg-white/20 px-6 py-2.5 font-semibold backdrop-blur transition hover:bg-white/30"
          >
            <Info size={18} />
            More Info
          </button>
        </div>
      </div>

      {films.length > 1 && (
        <div className="absolute bottom-20 right-4 flex items-center gap-2 md:right-12">
          {films.map((film, i) => (
            <button
              key={film.tmdb_id}
              type="button"
              onClick={() => go(i)}
              aria-label={`Show ${film.title}`}
              aria-current={i === index}
              // A 4px dot is a 4px target. The button stays finger-sized and the
              // visible dot is drawn inside it.
              className="grid h-6 w-6 place-items-center"
            >
              <span
                className={`block h-[3px] rounded-full transition-all duration-300 ${
                  i === index ? 'w-6 bg-white' : 'w-3 bg-white/40 hover:bg-white/70'
                }`}
              />
            </button>
          ))}
        </div>
      )}
    </header>
  )
}
