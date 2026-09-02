/**
 * Loading placeholders shaped like the thing that is loading.
 *
 * A spinner would say "something is happening"; these say "a row of six cards is
 * about to appear here", so the page does not jump when it does.
 */

export function RowSkeleton({ cards = 6 }: { cards?: number }) {
  return (
    <section className="py-3">
      <div className="mb-2 ml-4 h-5 w-48 animate-pulse rounded bg-surface-2 md:ml-12" />
      <div className="flex gap-2 overflow-hidden px-4 md:px-12">
        {Array.from({ length: cards }, (_, i) => (
          <div
            key={i}
            className="aspect-video w-[240px] shrink-0 animate-pulse rounded-md bg-surface-2"
          />
        ))}
      </div>
    </section>
  )
}

export function CardGridSkeleton({ cards = 6 }: { cards?: number }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: cards }, (_, i) => (
        <div key={i} className="h-52 animate-pulse rounded-lg bg-surface-2" />
      ))}
    </div>
  )
}
