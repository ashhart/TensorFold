import { useEffect, useState } from 'react'
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
  const summary = activityGroupStateSummary(items)
  // Auto-expand when any tool in the run is awaiting approval so the approval
  // UI can never be hidden behind a collapsed line.
  const forceOpen = summary.awaiting
  const [open, setOpen] = useState(defaultOpen)
  useEffect(() => { if (forceOpen) setOpen(true) }, [forceOpen])
  const expanded = open || forceOpen

  const label = activityGroupLabel(items)
  const icon = toolIcon(items[0]?.tool)

  return (
    <div className="fade-in">
      <button
        onClick={() => !forceOpen && setOpen((o) => !o)}
        aria-expanded={expanded}
        className="group flex w-full items-center gap-2 py-[3px] text-left text-[13px]"
      >
        <Icon name={icon} size={14} className="shrink-0 text-ink-3" />
        <span className="min-w-0 flex-1 truncate text-[13px] text-ink-2">{label}</span>
        {summary.text && (
          <span className={clsx('shrink-0 text-[12px]', summary.error ? 'text-danger' : 'text-ink-4')}>{summary.text}</span>
        )}
        <span className="flex w-4 shrink-0 items-center justify-center">
          <StateGlyph state={summary.error ? 'error' : summary.pending ? 'running' : 'ok'} />
        </span>
        <Icon name={expanded ? 'chevronDown' : 'chevronRight'} size={13} className="shrink-0 text-ink-4 opacity-0 transition-opacity group-hover:opacity-100" />
      </button>

      {expanded && (
        <div className="ml-[6px] mt-1 space-y-0.5 border-l border-line pl-3.5">
          {items.map((item) => (
            <ToolCard key={item.id} item={item} />
          ))}
        </div>
      )}
    </div>
  )
}
