import { DocumentEvidenceButtons } from '../documents/DocumentEvidenceButtons'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { remarkCitationAutolinks } from '../../lib/markdown-citations'
import { Icon } from '../../lib/icons'
import { useStore } from '../../lib/store'
import { mdComponents } from '../../lib/highlight'
import { groupToolRuns } from '../../lib/activity-grouping'
import { turnActivityState, workRowLabel, type TurnSummary } from '../../lib/turn-blocks'
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
export function TurnWorkRow({ turn, status, phaseLabel, defaultOpen = false, reveal = 0, searchItemId }: {
  turn: TurnSummary
  status?: string
  phaseLabel?: string
  defaultOpen?: boolean
  reveal?: number
  searchItemId?: string | null
}) {
  const tx = useAuthoredText()
  const [open, setOpen] = useState(defaultOpen)
  useEffect(() => { if (reveal > 0) setOpen(true) }, [reveal])
  const locale = useStore(state => state.settings?.uiPreferences?.locale)
  const now = useNow(turn.running)
  const label = workRowLabel(turn, { now, status, phaseLabel, locale })
  const hasDetails = turn.work.length > 0
  const state = turnActivityState(turn, status)
  const stateLabel = { running: 'Running', waiting: 'Needs you', completed: 'Finished', failed: 'Failed', stopped: 'Stopped', review: 'Review needed', interrupted: 'Interrupted' }[state]
  const needsAttention = state === 'failed' || state === 'review' || state === 'interrupted'
  const detailsId = `turn-details-${turn.key}`
  return (
    <div className={clsx('fade-in min-w-0 rounded-xl border bg-card', turn.running ? 'border-accent/25' : 'border-line')}>
      <span role="status" aria-live="polite" className="sr-only">{turn.running ? phaseLabel || tx(stateLabel) : undefined}</span>
      <button
        type="button"
        onClick={() => hasDetails && setOpen((value) => !value)}
        disabled={!hasDetails}
        aria-expanded={hasDetails ? open : undefined}
        aria-controls={hasDetails ? detailsId : undefined}
        className="group flex w-full items-start gap-2.5 rounded-xl px-3 py-2.5 text-left text-[13px] leading-5 transition-colors hover:bg-surface-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-default"
      >
        <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
          {state === 'running' ? <Spinner size={14} /> : <Icon name={state === 'waiting' ? 'shield' : needsAttention ? 'warning' : state === 'stopped' ? 'stop' : 'check'} size={15} className={needsAttention ? 'text-danger' : state === 'waiting' ? 'text-accent' : 'text-ink-3'} />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="min-w-0 break-words font-medium text-ink-2">{label}</span>
            <span className={clsx('rounded px-1.5 text-[11px] font-medium', needsAttention ? 'bg-danger/8 text-danger' : state === 'running' || state === 'waiting' ? 'bg-accent-soft text-accent' : 'bg-surface text-ink-3')}>{tx(stateLabel)}</span>
          </span>
          {needsAttention && <span className="mt-0.5 block text-[12px] text-ink-3">{tx('Review the activity before continuing.')}</span>}
        </span>
        {hasDetails && <Icon name={open ? 'chevronDown' : 'chevronRight'} size={15} className="mt-0.5 shrink-0 text-ink-3" />}
      </button>
      {open && hasDetails && (
        <div id={detailsId} className="space-y-2 border-t border-line px-3 py-2.5">
          <WorkItems items={turn.work} running={turn.running} status={status} searchItemId={searchItemId} />
        </div>
      )}
    </div>
  )
}

function WorkItems({ items, running, status, searchItemId }: { items: TimelineItem[]; running: boolean; status?: string; searchItemId?: string | null }) {
  const collapseReasoningDefault = useStore((s) => s.collapseReasoningDefault)
  const lastReasoningId = [...items].reverse().find((item) => item.kind === 'reasoning')?.id
  return (
    <>
      {(searchItemId ? items.map(item => [item]) : groupToolRuns(items)).map((group) => {
        if (group.length > 1) return <ActivityLineGroup key={`g-${group[0].id}`} items={group} />
        const item = group[0]
        const wrap = (child: React.ReactNode) => <div key={item.id} id={`ti-${item.id}`} className={item.id === searchItemId ? 'rounded-lg outline outline-2 outline-accent outline-offset-4' : undefined}>{child}</div>
        if (item.kind === 'tool') return wrap(<><ToolCard item={item} foldDiff forceExpanded={item.id === searchItemId} />{item.documentEvidence?.length ? <DocumentEvidenceButtons references={item.documentEvidence} /> : null}</>)
        if (item.kind === 'reasoning') return wrap(<Reasoning text={item.text || ''} active={running && status === 'thinking' && item.id === lastReasoningId} forceExpanded={!collapseReasoningDefault || item.id === searchItemId} />)
        if (item.kind === 'assistant') return (
          <div key={item.id} id={`ti-${item.id}`} className="prose-tf text-[13px] leading-5 text-ink-2">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkCitationAutolinks]} components={mdComponents}>{item.text || ''}</ReactMarkdown>
          </div>
        )
        if (item.kind === 'notice') return wrap(<WorkNote item={item} />)
        return null
      })}
    </>
  )
}

// Run bookkeeping (contract, receipt, guard and verify notes) as one quiet line each.
function WorkNote({ item }: { item: TimelineItem }) {
  const verdict = item.runReceipt?.verdict
  const concern = verdict === 'failed' || verdict === 'blocked' || verdict === 'unverified'
  const icon = concern ? 'warning' : verdict === 'cancelled' ? 'stop' : item.runReceipt ? 'check' : item.runContract || item.title === 'Guard' || item.title === 'Verify-on-stop' ? 'shield' : 'layers'
  return (
    <div className="flex items-start gap-2 py-0.5 text-[13px] leading-4 text-ink-3">
      <Icon name={icon} size={14} className={clsx('mt-px shrink-0', concern ? 'text-danger' : 'text-ink-4')} />
      <span className="min-w-0 break-words">{item.text || item.title}</span>
    </div>
  )
}
