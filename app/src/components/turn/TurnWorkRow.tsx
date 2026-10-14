import { useEffect, useState } from 'react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { remarkCitationAutolinks } from '../../lib/markdown-citations'
import { Icon } from '../../lib/icons'
import { useStore } from '../../lib/store'
import { mdComponents } from '../../lib/highlight'
import { groupToolRuns } from '../../lib/activity-grouping'
import { workRowLabel, type TurnSummary } from '../../lib/turn-blocks'
import type { TimelineItem } from '../../lib/types'
import { ActivityLineGroup } from '../ActivityLine'
import { Reasoning, Spinner, ToolCard } from '../timeline'

// Ticks once a second while a turn is live so the elapsed time moves.
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [active])
  return now
}

// One collapsible row per turn: "Worked for 12s · 4 steps", or live "Working · 12s · Editing App.tsx".
export function TurnWorkRow({ turn, status, phaseLabel, defaultOpen = false, reveal = 0 }: {
  turn: TurnSummary
  status?: string
  phaseLabel?: string
  defaultOpen?: boolean
  reveal?: number
}) {
  const [open, setOpen] = useState(defaultOpen)
  useEffect(() => { if (reveal > 0) setOpen(true) }, [reveal])
  const now = useNow(turn.running)
  const label = workRowLabel(turn, { now, status, phaseLabel })
  const hasDetails = turn.work.length > 0
  return (
    <div className="fade-in">
      <span role="status" aria-live="polite" className="sr-only">{turn.running ? phaseLabel : undefined}</span>
      <button
        type="button"
        onClick={() => hasDetails && setOpen((value) => !value)}
        aria-expanded={hasDetails ? open : undefined}
        className={clsx('group -ml-2 flex max-w-full items-center gap-2 rounded-[10px] px-2 py-1 text-left text-[13px] leading-[18px] transition-colors', hasDetails ? 'text-ink-3 hover:bg-ink-5 hover:text-ink-2' : 'cursor-default text-ink-3')}
      >
        {turn.running && <Spinner size={13} />}
        <span className={clsx('min-w-0 truncate', turn.running && 'shimmer')}>{label}</span>
        {hasDetails && <Icon name={open ? 'chevronDown' : 'chevronRight'} size={14} className="shrink-0 opacity-60 transition-opacity group-hover:opacity-100" />}
      </button>
      {open && hasDetails && (
        <div className="ml-[3px] mt-1.5 space-y-1.5 border-l border-line pb-1 pl-4">
          <WorkItems items={turn.work} running={turn.running} status={status} />
        </div>
      )}
    </div>
  )
}

function WorkItems({ items, running, status }: { items: TimelineItem[]; running: boolean; status?: string }) {
  const collapseReasoningDefault = useStore((s) => s.collapseReasoningDefault)
  const lastReasoningId = [...items].reverse().find((item) => item.kind === 'reasoning')?.id
  return (
    <>
      {groupToolRuns(items).map((group) => {
        if (group.length > 1) return <ActivityLineGroup key={`g-${group[0].id}`} items={group} />
        const item = group[0]
        if (item.kind === 'tool') return <ToolCard key={item.id} item={item} foldDiff />
        if (item.kind === 'reasoning') return <Reasoning key={item.id} text={item.text || ''} active={running && status === 'thinking' && item.id === lastReasoningId} forceExpanded={!collapseReasoningDefault} />
        if (item.kind === 'assistant') return (
          <div key={item.id} className="prose-tf text-[13px] leading-5 text-ink-2">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkCitationAutolinks]} components={mdComponents}>{item.text || ''}</ReactMarkdown>
          </div>
        )
        if (item.kind === 'notice') return <WorkNote key={item.id} item={item} />
        return null
      })}
    </>
  )
}

// Run bookkeeping (contract, receipt, guard and verify notes) as one quiet line each.
function WorkNote({ item }: { item: TimelineItem }) {
  const icon = item.runReceipt || item.runContract ? 'check' : item.title === 'Guard' || item.title === 'Verify-on-stop' ? 'shield' : 'layers'
  return (
    <div className="flex items-start gap-2 py-0.5 text-[13px] leading-4 text-ink-3">
      <Icon name={icon} size={14} className="mt-px shrink-0 text-ink-4" />
      <span className="min-w-0 break-words">{item.text || item.title}</span>
    </div>
  )
}
