/**
 * Every poster and backdrop in the app goes through here.
 *
 * The same three-step fallback was written out four times before this existed:
 * use the backdrop, else the poster, else a tinted placeholder — and separately,
 * swap to the placeholder if the CDN 404s a path the catalogue still holds.
 * That second case is not theoretical: TMDB paths in the snapshot are as old as
 * the snapshot, and an <img> whose src fails renders a broken-image icon, which
 * is uglier than any placeholder.
 */

import { useEffect, useState } from 'react'
import { backdropUrl, placeholderTint, posterUrl } from '../lib/images'
import type { BackdropSize, PosterSize } from '../lib/images'

interface Props {
  title: string
  posterPath?: string | null
  backdropPath?: string | null
  /** Which image to prefer; the other is the fallback when it is missing. */
  prefer?: 'poster' | 'backdrop'
  posterSize?: PosterSize
  backdropSize?: BackdropSize
  className?: string
  /** The first hero slide is the page's largest paint and must not be lazy. */
  eager?: boolean
}

export default function Artwork({
  title,
  posterPath,
  backdropPath,
  prefer = 'backdrop',
  posterSize = 'w500',
  backdropSize = 'w780',
  className = 'h-full w-full object-cover',
  eager = false,
}: Props) {
  const [failed, setFailed] = useState(false)

  // A card can be recycled onto a different film as a row re-renders, so a
  // previous failure must not condemn the new image.
  useEffect(() => setFailed(false), [posterPath, backdropPath])

  const poster = posterUrl(posterPath, posterSize)
  const backdrop = backdropUrl(backdropPath, backdropSize)
  const src = failed ? null : prefer === 'poster' ? (poster ?? backdrop) : (backdrop ?? poster)

  if (!src) {
    return <div className={className} style={{ background: placeholderTint(title) }} />
  }

  return (
    <img
      src={src}
      // Decorative: the title is always rendered as text beside or over it, so
      // alt text here would make a screen reader announce every film twice.
      alt=""
      loading={eager ? 'eager' : 'lazy'}
      fetchPriority={eager ? 'high' : 'auto'}
      onError={() => setFailed(true)}
      className={className}
    />
  )
}
