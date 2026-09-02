/**
 * TMDB image URLs.
 *
 * The backend stores PATHS and never serves image bytes (contract §9), so every
 * URL in this app is built here. Both paths are nullable in every response —
 * roughly one film in twenty has no poster — so callers get `null` back and are
 * expected to render a placeholder rather than an <img> pointed at "undefined".
 */

const BASE = 'https://image.tmdb.org/t/p'

export type PosterSize = 'w185' | 'w342' | 'w500' | 'original'
export type BackdropSize = 'w780' | 'w1280' | 'original'

export function posterUrl(path: string | null | undefined, size: PosterSize = 'w342') {
  return path ? `${BASE}/${size}${path}` : null
}

export function backdropUrl(path: string | null | undefined, size: BackdropSize = 'w1280') {
  return path ? `${BASE}/${size}${path}` : null
}

/**
 * What a card shows when TMDB has no image. Deterministic per title so the same
 * film keeps the same placeholder between renders — a colour that changes on
 * every paint reads as a bug.
 */
export function placeholderTint(title: string): string {
  let hash = 0
  for (let i = 0; i < title.length; i++) hash = (hash * 31 + title.charCodeAt(i)) | 0
  const hue = Math.abs(hash) % 360
  return `linear-gradient(150deg, hsl(${hue} 18% 18%), hsl(${(hue + 40) % 360} 16% 11%))`
}

/** "1h 47m" — TMDB gives minutes, nobody reads a runtime in minutes. */
export function formatRuntime(minutes: number | null | undefined): string | null {
  if (!minutes || minutes <= 0) return null
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return h ? `${h}h ${m}m` : `${m}m`
}
