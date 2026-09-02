/**
 * One film inside the overlay: banner, credits, franchise, More Like This.
 *
 * Detail and neighbours are fetched SEPARATELY and rendered as they arrive. The
 * detail is what the reader clicked for and comes from one graph query; the
 * similarity search is a vector lookup that takes longer. Awaiting both would
 * hold a fast answer hostage to a slow one.
 */

import { useEffect, useState } from 'react'
import * as api from '../lib/api'
import type { MovieDetail, SimilarResponse } from '../types'
import { formatRuntime } from '../lib/images'
import Artwork from './Artwork'
import { CardGridSkeleton } from './Skeletons'
import { Play, Plus, ThumbUp } from './icons'
import type { View } from './DetailModal'

interface Props {
  tmdbId: number
  onNavigate: (view: View) => void
}

export default function MovieDetailView({ tmdbId, onNavigate }: Props) {
  const [film, setFilm] = useState<MovieDetail | null>(null)
  const [related, setRelated] = useState<SimilarResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()

    api
      .movie(tmdbId, controller.signal)
      .then(setFilm)
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return
        // 404 and 503 are genuinely different facts and the reader deserves to
        // know which one happened — one is permanent, the other is worth a retry.
        setError(
          err instanceof api.ApiError && err.status === 404
            ? 'That film is not in the catalogue.'
            : 'Could not load this film right now.',
        )
      })

    api
      .similar(tmdbId, 12, controller.signal)
      .then(setRelated)
      // A failed neighbours call is not worth an error message: the page above
      // it is complete and useful on its own, so the row simply does not appear.
      .catch(() => setRelated({ films: [], franchise: null, degraded: true }))

    return () => controller.abort()
  }, [tmdbId])

  if (error) {
    return <p className="p-16 text-center text-[color:var(--text-muted)]">{error}</p>
  }

  if (!film) {
    return (
      <>
        <div className="aspect-video w-full animate-pulse bg-surface-2" />
        <div className="space-y-4 p-8">
          <div className="h-8 w-2/3 animate-pulse rounded bg-surface-2" />
          <div className="h-24 animate-pulse rounded bg-surface-2" />
        </div>
      </>
    )
  }

  const runtime = formatRuntime(film.runtime)

  return (
    <>
      <div className="relative aspect-video w-full overflow-hidden">
        <Artwork
          title={film.title}
          backdropPath={film.backdrop_path}
          posterPath={film.poster_path}
          backdropSize="original"
          eager
        />
        <div className="scrim-bottom absolute inset-x-0 bottom-0 h-3/4" />

        <div className="absolute bottom-6 left-6 right-6 space-y-4 md:left-10">
          <h2 className="max-w-3xl text-3xl font-black drop-shadow-lg md:text-5xl">{film.title}</h2>
          {film.tagline && (
            <p className="max-w-xl text-sm italic text-white/70 md:text-base">{film.tagline}</p>
          )}
          <div className="flex items-center gap-3">
            <button
              type="button"
              className="flex items-center gap-2 rounded bg-white px-6 py-2 font-semibold text-black transition hover:bg-white/85"
            >
              <Play size={16} />
              Play
            </button>
            <span className="grid size-10 place-items-center rounded-full border-2 border-white/50 text-white/85">
              <Plus size={18} />
            </span>
            <span className="grid size-10 place-items-center rounded-full border-2 border-white/50 text-white/85">
              <ThumbUp size={18} />
            </span>
          </div>
        </div>
      </div>

      <div className="grid gap-8 p-6 md:grid-cols-[1.6fr_1fr] md:p-10">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
            {film.rating != null && (
              <span className="font-semibold text-emerald-400">{film.rating.toFixed(1)}</span>
            )}
            {film.year != null && <span>{film.year}</span>}
            {runtime && <span>{runtime}</span>}
            <span className="rounded border border-white/30 px-1.5 text-[11px] text-white/70">
              HD
            </span>
          </div>

          {film.overview && <p className="leading-relaxed text-white/90">{film.overview}</p>}
        </div>

        <dl className="space-y-3 text-sm">
          {film.cast.length > 0 && (
            <Credits
              label="Cast"
              // Ten is TMDB's top billing and all the graph stores. Saying "more"
              // when there is nothing more to show would be a dead promise.
              people={film.cast.map((c) => ({
                id: c.person_id,
                name: c.name,
                note: c.character,
              }))}
              onPick={(id) => onNavigate({ kind: 'person', id })}
            />
          )}

          {film.directors.length > 0 && (
            <Credits
              label={film.directors.length > 1 ? 'Directors' : 'Director'}
              people={film.directors.map((d) => ({ id: d.person_id, name: d.name, note: null }))}
              onPick={(id) => onNavigate({ kind: 'person', id })}
            />
          )}

          {film.genres.length > 0 && (
            <div>
              <dt className="text-[color:var(--text-muted)]">Genres</dt>
              <dd className="mt-0.5">{film.genres.join(', ')}</dd>
            </div>
          )}

          {film.keywords.length > 0 && (
            <div>
              <dt className="text-[color:var(--text-muted)]">This film is</dt>
              <dd className="mt-0.5 capitalize">{film.keywords.slice(0, 6).join(', ')}</dd>
            </div>
          )}
        </dl>
      </div>

      {/* The graph's exact answer, kept separate from the vector store's
          approximate one below. Conflating "belongs with" and "feels like" is
          the confusion this architecture exists to avoid. */}
      {related?.franchise && related.franchise.films.length > 1 && (
        <section className="border-t border-hairline px-6 py-8 md:px-10">
          <h3 className="mb-1 text-xl font-bold">
            {related.franchise.collection_name ?? 'The collection'}
          </h3>
          <p className="mb-4 text-sm text-[color:var(--text-muted)]">
            The whole series, in release order.
          </p>

          <div className="no-scrollbar flex gap-3 overflow-x-auto pb-2">
            {related.franchise.films.map((entry) => (
              <button
                  key={entry.tmdb_id}
                  type="button"
                  onClick={() => onNavigate({ kind: 'movie', id: entry.tmdb_id })}
                  className={`group w-[132px] shrink-0 text-left ${
                    entry.is_seed ? '' : 'opacity-80 transition hover:opacity-100'
                  }`}
                >
                  <div
                    className={`aspect-[2/3] overflow-hidden rounded-md bg-surface-2 ${
                      entry.is_seed ? 'ring-2 ring-brand' : ''
                    }`}
                  >
                    {/* Posters here, not stills: a timeline is a shelf of
                        spines, and 2:3 is how a series reads as a series. */}
                    <Artwork
                      title={entry.title}
                      posterPath={entry.poster_path}
                      prefer="poster"
                      posterSize="w342"
                    />
                  </div>
                  <p className="mt-1.5 line-clamp-2 text-xs font-medium">{entry.title}</p>
                  <p className="text-xs text-[color:var(--text-muted)]">
                    {entry.year} {entry.is_seed && <span className="text-brand">• this one</span>}
                  </p>
                </button>
            ))}
          </div>
        </section>
      )}

      <section className="border-t border-hairline px-6 py-8 md:px-10">
        <h3 className="mb-4 text-xl font-bold">More Like This</h3>

        {!related ? (
          <CardGridSkeleton />
        ) : related.films.length === 0 ? (
          <p className="text-sm text-[color:var(--text-muted)]">
            {related.degraded
              ? 'Similar films could not be loaded right now.'
              : 'Nothing else in the catalogue is close to this one.'}
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {related.films.map((neighbour) => (
              <button
                key={neighbour.tmdb_id}
                type="button"
                onClick={() => onNavigate({ kind: 'movie', id: neighbour.tmdb_id })}
                className="overflow-hidden rounded-lg bg-surface-2 text-left transition hover:bg-[#26262c]"
              >
                  <div className="relative aspect-video overflow-hidden">
                    <Artwork
                      title={neighbour.title}
                      backdropPath={neighbour.backdrop_path}
                      posterPath={neighbour.poster_path}
                    />
                    <span className="absolute bottom-2 left-2 right-2 truncate text-sm font-semibold drop-shadow">
                      {neighbour.title}
                    </span>
                  </div>

                  <div className="space-y-2 p-3">
                    <div className="flex items-center gap-2 text-xs">
                      {neighbour.rating != null && (
                        <span className="font-semibold text-emerald-400">
                          {neighbour.rating.toFixed(1)}
                        </span>
                      )}
                      {neighbour.year != null && (
                        <span className="text-[color:var(--text-muted)]">{neighbour.year}</span>
                      )}
                      <span className="ml-auto grid size-7 place-items-center rounded-full border border-white/35 text-white/70">
                        <Plus size={13} />
                      </span>
                    </div>
                    {neighbour.overview && (
                      <p className="line-clamp-4 text-xs leading-relaxed text-white/70">
                        {neighbour.overview}
                      </p>
                    )}
                  </div>
              </button>
            ))}
          </div>
        )}
      </section>
    </>
  )
}

function Credits({
  label,
  people,
  onPick,
}: {
  label: string
  people: { id: number; name: string; note: string | null }[]
  onPick: (id: number) => void
}) {
  return (
    <div>
      <dt className="text-[color:var(--text-muted)]">{label}</dt>
      <dd className="mt-0.5 leading-relaxed">
        {people.map((person, i) => (
          <span key={`${person.id}-${i}`}>
            <button
              type="button"
              onClick={() => onPick(person.id)}
              // Addressed by person_id, never by name: this catalogue has 48
              // names owned by more than one person, and a name-keyed jump would
              // show you the wrong filmography for every one of them.
              title={person.note ? `as ${person.note}` : undefined}
              className="underline decoration-white/25 underline-offset-2 transition hover:text-brand hover:decoration-brand"
            >
              {person.name}
            </button>
            {i < people.length - 1 && ', '}
          </span>
        ))}
      </dd>
    </div>
  )
}
