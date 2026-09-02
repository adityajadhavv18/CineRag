/**
 * Renders the agent's answer text.
 *
 * Deliberately not a markdown library. The agent emits a narrow, known subset —
 * paragraphs, bullets, numbered clarification options, **bold** — and one thing
 * markdown has no concept of: `[1]` citation markers that index 1-based into
 * `sources`. Those are the whole point of a grounded answer, so they are turned
 * into controls that point at the film they cite rather than left as noise or
 * stripped out. A general renderer would need a plugin for that anyway.
 */

import type { JSX, ReactNode } from 'react'

const BULLET = /^\s*[-*•]\s+/
const NUMBERED = /^\s*\d+[.)]\s+/
/** Split on the two inline forms, keeping the delimiters. */
const INLINE = /(\*\*[^*]+\*\*|\[\d+\])/g

interface Props {
  text: string
  /** Renders one citation marker. Omit to strip markers entirely. */
  renderCitation?: (index: number) => ReactNode
}

export function RichText({ text, renderCitation }: Props) {
  const blocks: JSX.Element[] = []
  let bullets: string[] = []

  const flush = () => {
    if (bullets.length === 0) return
    blocks.push(
      <ul key={`ul${blocks.length}`} className="ml-4 list-disc space-y-1 text-sm">
        {bullets.map((item, i) => (
          <li key={i}>{inline(item, renderCitation)}</li>
        ))}
      </ul>,
    )
    bullets = []
  }

  for (const raw of text.split('\n')) {
    const line = raw.trimEnd()

    if (!line.trim()) {
      flush()
      continue
    }

    if (BULLET.test(line)) {
      bullets.push(line.replace(BULLET, ''))
      continue
    }

    flush()

    // A numbered line is a clarification question, and the agent's questions
    // carry the weight of the answer — they get emphasis, not body styling.
    blocks.push(
      NUMBERED.test(line) ? (
        <p key={blocks.length} className="text-sm font-semibold">
          {inline(line, renderCitation)}
        </p>
      ) : (
        <p key={blocks.length} className="text-sm leading-relaxed">
          {inline(line, renderCitation)}
        </p>
      ),
    )
  }

  flush()
  return <div className="space-y-2">{blocks}</div>
}

function inline(text: string, renderCitation?: (index: number) => ReactNode): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }

    const citation = /^\[(\d+)\]$/.exec(part)
    if (citation) {
      const index = Number(citation[1])
      // Markers are 1-based into `sources`. With no renderer supplied they are
      // dropped rather than shown raw — "[3]" pointing at nothing is worse than
      // no marker at all.
      return renderCitation ? <span key={i}>{renderCitation(index)}</span> : null
    }

    return part
  })
}
