/**
 * A person inside the overlay: what they acted in, what they directed.
 *
 * The two are kept as separate sections rather than merged into one filmography.
 * They are different relationships in the graph and different facts about the
 * person — Clint Eastwood acting in a film and directing it are not the same
 * credit, and a merged list would have to silently pick one label for both.
 */

import { useEffect, useState } from 'react'
import * as api from '../lib/api'
import type { MovieCard, PersonResponse } from '../types'
import Artwork from './Artwork'
import type { View } from './DetailModal'

interface Props {
  personId: number
  onNavigate: (view: View) => void
}

export default function PersonDetailView({ personId, onNavigate }: Props) {
  const [profile, setProfile] = useState<PersonResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()

    api
      .person(personId, controller.signal)
      .then(setProfile)
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return
        setError(
          err instanceof api.ApiError && err.status === 404
            ? 'That person is not in the catalogue.'
            : 'Could not load this person right now.',
        )
      })

    return () => controller.abort()
  }, [personId])

  if (error) {
    return <p className="p-16 text-center text-[color:var(--text-muted)]">{error}</p>
  }

  if (!profile) {
    return (
      <div className="space-y-4 p-10">
        <div className="h-9 w-56 animate-pulse rounded bg-surface-2" />
        <div className="h-40 animate-pulse rounded bg-surface-2" />
      </div>
    )
  }

  const total = profile.acted.length + profile.directed.length

  return (
    <div className="p-6 md:p-10">
      <header className="mb-8">
        <h2 className="text-3xl font-black md:text-4xl">{profile.name}</h2>
        <p className="mt-1 text-sm text-[color:var(--text-muted)]">
          {total === 0
            ? 'No credits in this catalogue.'
            : [
                profile.acted.length && `${profile.acted.length} acting`,
                profile.directed.length && `${profile.directed.length} directing`,
              ]
                .filter(Boolean)
                .join(' • ') + ` credit${total === 1 ? '' : 's'} in this catalogue`}
        </p>
      </header>

      {profile.acted.length > 0 && (
        <Filmography
          title="Acting"
          films={profile.acted}
          onOpen={(id) => onNavigate({ kind: 'movie', id })}
        />
      )}

      {profile.directed.length > 0 && (
        <Filmography
          title="Directing"
          films={profile.directed}
          onOpen={(id) => onNavigate({ kind: 'movie', id })}
        />
      )}
    </div>
  )
}

function Filmography({
  title,
  films,
  onOpen,
}: {
  title: string
  films: MovieCard[]
  onOpen: (tmdbId: number) => void
}) {
  return (
    <section className="mb-8 last:mb-0">
      <h3 className="mb-3 text-lg font-bold">{title}</h3>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {films.map((film) => (
          <button
            key={film.tmdb_id}
            type="button"
            onClick={() => onOpen(film.tmdb_id)}
            className="overflow-hidden rounded-lg bg-surface-2 text-left transition hover:bg-[#26262c]"
          >
              <div className="relative aspect-video overflow-hidden">
                <Artwork
                  title={film.title}
                  backdropPath={film.backdrop_path}
                  posterPath={film.poster_path}
                />
              </div>
              <div className="p-2.5">
                <p className="line-clamp-1 text-sm font-semibold">{film.title}</p>
                <p className="mt-0.5 text-xs text-[color:var(--text-muted)]">
                  {film.year}
                  {film.rating != null && (
                    <span className="ml-2 text-emerald-400">{film.rating.toFixed(1)}</span>
                  )}
                </p>
                {/* The character sits on the ACTED_IN edge, so it is specific to
                    THIS film — the same actor plays someone else in the next. */}
                {film.character && (
                  <p className="mt-0.5 line-clamp-1 text-xs italic text-white/50">
                    as {film.character}
                  </p>
                )}
              </div>
          </button>
        ))}
      </div>
    </section>
  )
}
