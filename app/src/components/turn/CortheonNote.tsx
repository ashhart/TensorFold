import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { MarkdownLink } from '../../lib/markdown-link'
import { cortheonSummary } from '../../lib/turn-stats-line'
import type { AssistantTurnStats } from '../../lib/types'

// Under an answer the built-in engine shaped with Cortheon: what it searched and read, with its sources.
export function CortheonNote({ stats }: { stats: AssistantTurnStats }) {
  const tx = useAuthoredText()
  const [open, setOpen] = useState(false)
  const summary = cortheonSummary(stats)
  if (!summary) return null
  const sources = stats.cortheon?.sources ?? []
  return (
    <div data-testid="cortheon-note" className="mt-1 text-[13px] leading-4 text-ink-4">
      <button
        type="button"
        disabled={!sources.length}
        onClick={() => setOpen((value) => !value)}
        aria-expanded={sources.length ? open : undefined}
        className="rounded-sm text-info-ink/80 enabled:hover:text-info-ink disabled:cursor-default focus-visible:outline-2 focus-visible:outline-accent"
      >
        {summary}{sources.length > 0 && tx(" · Sources: {count}", { count: sources.length })}
      </button>
      {open && sources.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {sources.map((source) => (
            <li key={source} className="truncate">
              <MarkdownLink href={source} className="text-ink-3 underline decoration-ink-4 underline-offset-2 hover:text-ink">{source}</MarkdownLink>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
