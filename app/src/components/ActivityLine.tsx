import { useEffect, useState } from 'react'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { Icon } from '../lib/icons'
import { activityGroupLabel, activityGroupStateSummary } from '../lib/activity-grouping'
import { ToolCard, toolIcon, StateGlyph } from './timeline'
import type { TimelineItem } from '../lib/types'

// Task #40 — a run of consecutive same-tool calls rendered as a single compact
// "activity line" that expands to the individual tool cards on click. Reuses
// the existing ToolCard for the expanded view so card behaviour (diffs,
// approvals, copy) stays identical.
export function ActivityLineGroup({
  items,
  defaultOpen = false,
}: {
  items: TimelineItem[]
  defaultOpen?: boolean
}) {
  const tx = useAuthoredText()
  const summary = activityGroupStateSummary(items)
  // Auto-expand when any tool in the run is awaiting approval so the approval
  // UI can never be hidden behind a collapsed line.
  const forceOpen = summary.awaiting
  const [open, setOpen] = useState(defaultOpen)
  useEffect(() => { if (forceOpen) setOpen(true) }, [forceOpen])
  const expanded = open || forceOpen

  const label = activityGroupLabel(items)
  const icon = toolIcon(items[0]?.tool)
  const detailsId = `activity-${items[0]?.id}`

  return (
    <div className="fade-in">
      <button
        type="button"
        onClick={() => !forceOpen && setOpen((o) => !o)}
        aria-expanded={expanded}
        aria-controls={detailsId}
        className="group flex w-full items-center gap-2 rounded-lg px-1.5 py-2 text-left text-[13px] transition-colors hover:bg-surface-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <Icon name={icon} size={14} className="shrink-0 text-ink-3" />
        <span className="min-w-0 flex-1 truncate text-[13px] text-ink-2">{label}</span>
        <span className={clsx('max-w-36 shrink-0 text-right text-[11px]', summary.error ? 'text-danger' : 'text-ink-3')}>{summary.text || tx('Completed')}</span>
        <span className="flex w-4 shrink-0 items-center justify-center">
          <StateGlyph state={summary.awaiting ? 'awaiting' : summary.pending ? 'running' : summary.error ? 'error' : summary.denied ? 'denied' : 'ok'} />
        </span>
        <Icon name={expanded ? 'chevronDown' : 'chevronRight'} size={14} className="shrink-0 text-ink-3" />
      </button>

      {expanded && (
        <div id={detailsId} className="ml-[6px] mt-1 space-y-1 border-l border-line pl-3.5">
          {items.map((item) => (
            <ToolCard key={item.id} item={item} />
          ))}
        </div>
      )}
    </div>
  )
}
