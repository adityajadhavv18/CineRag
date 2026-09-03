/**
 * The agent's narrowing questions, as choices rather than a typing exercise.
 *
 * "recommend some action movies" matches 1,156 films, so the agent asks what
 * kind — and every option it offers was counted out of the graph first, which is
 * the guarantee `clarification_node` exists to make. Printing those options as
 * prose threw that away at the last step: the user had to read "• 1990s (96)"
 * and type "1990s" back, where a typo or a paraphrase ("the nineties") lands
 * outside the vocabulary the stores match on EXACTLY.
 *
 * So the counted option is the thing you click, and the value that reaches
 * retrieval is the catalogue's own string.
 *
 * One question at a time, then a review, then it submits as an ordinary query —
 * there is no second endpoint and the server stays stateless. What gets sent is
 * a sentence the questions themselves describe how to build (`slot`/`phrase`),
 * so the wording lives with the counts on the server rather than being
 * reinvented here.
 */

import { useState } from 'react'
import type { Clarification } from '../types'

/**
 * Picks -> the query to send.
 *
 * The interesting part: each answer knows whether it reads before the subject
 * or after it, so three picks become "Recommend Thriller crime films from the
 * 1990s starring Denzel Washington" rather than a list of fragments. Unanswered
 * questions contribute nothing. This is the mirror of `slot`/`phrase` in
 * server/schemas.py — the only knowledge of the server's wording that lives here.
 */
function composeQuery(clarification: Clarification, picks: Record<string, string>): string {
  const chosen = clarification.questions.filter((question) => picks[question.id])
  const fill = (phrase: string, value: string) => phrase.replace('{value}', value)

  const before = chosen
    .filter((question) => question.slot === 'before')
    .map((question) => fill(question.phrase, picks[question.id]))
  const after = chosen
    .filter((question) => question.slot === 'after')
    .map((question) => fill(question.phrase, picks[question.id]))

  return ['Recommend', ...before, clarification.subject, ...after].join(' ')
}

interface Props {
  clarification: Clarification
  /** Sends the composed sentence as a normal turn. */
  onSubmit: (query: string) => void
}

export default function ClarifyCard({ clarification, onSubmit }: Props) {
  const { questions } = clarification
  const [step, setStep] = useState(0)
  const [picks, setPicks] = useState<Record<string, string>>({})

  const answered = questions.filter((question) => picks[question.id])
  const query = composeQuery(clarification, picks)

  // Past the last question: show what is about to be sent. A pick is a step
  // towards a search, not the search itself — seeing the sentence first is what
  // makes that honest, and it is the only place the whole set is visible at once.
  if (step >= questions.length) {
    return (
      <div className="space-y-3 rounded-xl border border-hairline bg-surface-2/60 p-3">
        <p className="text-[11px] uppercase tracking-wide text-[color:var(--text-muted)]">
          {answered.length > 0 ? "I'll go looking for" : 'Nothing picked'}
        </p>

        {answered.length > 0 ? (
          <p className="text-sm leading-snug">{query}</p>
        ) : (
          <p className="text-sm text-[color:var(--text-muted)]">
            Every question was skipped — say what you are after in your own words instead.
          </p>
        )}

        <div className="flex items-center gap-2">
          {answered.length > 0 && (
            <button
              type="button"
              onClick={() => onSubmit(query)}
              className="rounded-lg bg-brand px-3 py-1.5 text-sm font-medium text-white transition hover:bg-brand-dim"
            >
              Find these films
            </button>
          )}
          <button
            type="button"
            onClick={() => setStep(questions.length - 1)}
            className="text-xs text-[color:var(--text-muted)] transition hover:text-white"
          >
            Back
          </button>
        </div>
      </div>
    )
  }

  const question = questions[step]
  const picked = picks[question.id]

  const choose = (label: string) => {
    setPicks((current) => ({ ...current, [question.id]: label }))
    setStep(step + 1)
  }

  const skip = () => {
    // Dropped rather than left empty, so going back and skipping actually
    // retracts an earlier answer instead of quietly keeping it.
    setPicks(({ [question.id]: _dropped, ...rest }) => rest)
    setStep(step + 1)
  }

  return (
    <div className="space-y-2.5 rounded-xl border border-hairline bg-surface-2/60 p-3">
      <div className="flex items-baseline gap-2">
        <p className="text-sm font-medium leading-snug">{question.prompt}</p>
        {questions.length > 1 && (
          <span className="ml-auto shrink-0 text-[11px] tabular-nums text-[color:var(--text-muted)]">
            {step + 1} of {questions.length}
          </span>
        )}
      </div>

      {question.note && (
        <p className="text-xs text-[color:var(--text-muted)]">{question.note}</p>
      )}

      <div className="space-y-1.5">
        {question.options.map((option) => (
          <button
            key={option.label}
            type="button"
            onClick={() => choose(option.label)}
            // Marked when it is the standing answer, so stepping Back shows
            // what was chosen rather than an untouched question.
            aria-pressed={picked === option.label}
            className={`flex w-full items-center gap-3 rounded-lg border bg-surface px-3 py-2 text-left text-sm transition hover:border-brand hover:bg-brand/10 ${
              picked === option.label ? 'border-brand' : 'border-hairline'
            }`}
          >
            <span className="flex-1 truncate">{option.label}</span>
            {/* The count is the grounding, made visible: it is why this option
                is on the list at all, and it separates a rich direction from a
                technically-possible one. */}
            <span className="shrink-0 text-[11px] tabular-nums text-[color:var(--text-muted)]">
              {option.films.toLocaleString()} films
            </span>
          </button>
        ))}
      </div>

      <div className="flex items-center gap-3 text-xs">
        {step > 0 && (
          <button
            type="button"
            onClick={() => setStep(step - 1)}
            className="text-[color:var(--text-muted)] transition hover:text-white"
          >
            Back
          </button>
        )}
        {/* The prose version says "answer any of these", so a question has to be
            skippable — not every one of them has an answer the user cares about. */}
        <button
          type="button"
          onClick={skip}
          className="ml-auto text-[color:var(--text-muted)] transition hover:text-white"
        >
          No preference
        </button>
      </div>
    </div>
  )
}
