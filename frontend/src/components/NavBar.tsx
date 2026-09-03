/**
 * Top chrome. Transparent over the hero, solid once the page scrolls — the
 * reference does this and it matters: a solid bar at the top of a full-bleed
 * banner cuts the image in half.
 *
 * The search control is not a search box. It opens the chat panel, because this
 * app has no title-substring search — it has an agent. Labelling it "Ask" is the
 * honest description of what pressing it does.
 */

import { useEffect, useState } from 'react'
import { Search, Sparkle } from './icons'

interface Props {
  onAsk: () => void
  /** From /health — a store being down is worth admitting in the chrome. */
  degraded?: boolean
}

const LINKS = ['Home', 'Movies', 'Genres', 'My List']

export default function NavBar({ onAsk, degraded }: Props) {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav
      // Fixed, so the page's own inset does not apply to it — it has to stop at
      // the chat panel's edge itself, or the "Ask" button ends up underneath
      // the panel it opens.
      className={`fixed left-0 right-[var(--chat-inset)] top-0 z-30 flex items-center gap-6 px-4 py-3 transition-colors duration-300 md:px-12 ${
        scrolled ? 'bg-ink/95 backdrop-blur' : 'bg-gradient-to-b from-black/80 to-transparent'
      }`}
    >
      {/* Tighter tracking and a slight vertical squeeze — the wordmark in the
          reference is condensed, and plain bold at default tracking reads as a
          different logo sitting in the same spot. */}
      <span className="select-none text-[26px] font-bold leading-none tracking-[-0.055em] text-brand">
        NETFLIX
      </span>

      <ul className="hidden items-center gap-5 text-sm text-white/80 md:flex">
        {LINKS.map((link, i) => (
          <li
            key={link}
            className={i === 0 ? 'font-semibold text-white' : 'transition hover:text-white'}
          >
            {link}
          </li>
        ))}
      </ul>

      <div className="ml-auto flex items-center gap-3">
        {degraded && (
          <span
            title="A data store is unreachable — some results may be missing."
            className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-300"
          >
            Degraded
          </span>
        )}

        <button
          type="button"
          onClick={onAsk}
          className="flex items-center gap-2 rounded-full border border-white/25 bg-white/5 px-4 py-2 text-sm font-medium transition hover:border-white/50 hover:bg-white/10"
        >
          <Search size={16} />
          <span className="hidden sm:inline">Ask for something</span>
          <Sparkle size={14} className="text-brand" />
        </button>
      </div>
    </nav>
  )
}
