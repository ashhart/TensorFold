import { useEffect, useId, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { fullWorkspaceActivity, type FullWorkspaceActivity, type WorkActivityReason } from '../../lib/full-workspace-activity'
import { Icon } from '../../lib/icons'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useStore } from '../../lib/store'
import type { View } from '../../lib/store/types'
import { Menu } from '../ui'

const reasonLabels: Record<WorkActivityReason, string> = {
  approval: 'Approval needed', question: 'Question to answer', input: 'Waiting for you',
  recovery: 'Review run outcome', stalled: 'Work stalled', error: 'Run error',
  running: 'Running', queued: 'Queued', waiting: 'Waiting for a response',
}
const kindLabels: Record<FullWorkspaceActivity['kind'], string> = {
  task: 'Task', chat: 'Chat', room: 'Room', automation: 'Routine', job: 'Background work',
}

export function WorkActivityMenu({ variant = 'toolbar' }: { variant?: 'toolbar' | 'sidebar' }) {
  const tx = useAuthoredText()
  const threads = useStore((state) => state.threads)
  const status = useStore((state) => state.status)
  const approvals = useStore((state) => state.approvals)
  const jobs = useStore((state) => state.jobs)
  const rooms = useStore((state) => state.rooms)
  const roomStatus = useStore((state) => state.roomStatus)
  const automations = useStore((state) => state.automations)
  const projects = useStore((state) => state.projects)
  const setView = useStore((state) => state.setView)
  const panelId = useId()
  const activity = useMemo(() => fullWorkspaceActivity({ threads, status, approvals, jobs, rooms, roomStatus, automations, projects }), [threads, status, approvals, jobs, rooms, roomStatus, automations, projects])
  const needsYou = activity.filter((item) => item.needsYou).length

  return <div className={variant === 'sidebar' ? 'w-full' : undefined}>
    <span role="status" aria-live="polite" aria-atomic="true" className="sr-only">
      {needsYou ? tx("{value0} work items need your attention", { value0: needsYou }) : tx("No work items need your attention")}
    </span>
    <Menu width={360} align={variant === 'sidebar' ? 'start' : 'end'} trigger={(open) => <button
      type="button"
      aria-expanded={open}
      aria-controls={open ? panelId : undefined}
      aria-label={tx("Work activity: {value0} items, {value1} need attention", { value0: activity.length, value1: needsYou })}
      className={clsx('flex h-9 items-center gap-2 rounded-lg px-2.5 text-[13px] text-ink hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent', variant === 'sidebar' && 'w-full')}
    >
      <Icon name="bell" size={15} className={needsYou ? 'text-warn' : 'text-ink-3'} />
      <span className={variant === 'sidebar' ? 'flex-1 text-left' : undefined}>{tx("Work activity")}</span>
      {needsYou > 0
        ? <span className="rounded-md bg-warn/15 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-ink">{needsYou}</span>
        : activity.length > 0 && <span className="text-[12px] tabular-nums text-ink-3">{activity.length}</span>}
      {variant === 'sidebar' && <Icon name="chevronRight" size={12} className="text-ink-3" />}
    </button>}>
      {(close) => <WorkActivityContent id={panelId} activity={activity} navigate={(view) => { setView(view); close() }} />}
    </Menu>
  </div>
}

export function WorkActivityContent({ id, activity, navigate }: { id: string; activity: FullWorkspaceActivity[]; navigate: (view: View) => void }) {
  const tx = useAuthoredText()
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { ref.current?.querySelector<HTMLButtonElement>('button')?.focus() }, [])
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const groups = [
    { id: 'attention', label: tx("Needs you"), rows: activity.filter((item) => item.needsYou) },
    { id: 'progress', label: tx("In progress"), rows: activity.filter((item) => !item.needsYou) },
  ]

  return <div id={id} ref={ref} aria-label={tx("Work activity")}>
    <div className="px-2.5 pb-2 pt-2.5">
      <h2 className="text-[13px] font-semibold text-ink">{tx("Work activity")}</h2>
      <p className="mt-0.5 text-[12px] text-ink-3">{tx("Open an item to continue where it needs you.")}</p>
    </div>
    {activity.length === 0 && <p className="px-2.5 py-4 text-[13px] leading-relaxed text-ink-2">{tx("No active work recorded. Models and API servers can still be loaded.")}</p>}
    {groups.filter((group) => group.rows.length).map((group) => <section key={group.id} aria-labelledby={`${id}-${group.id}`} className="border-t border-line/60 py-1">
      <h3 id={`${id}-${group.id}`} className="flex items-center justify-between px-2.5 py-2 text-[11px] font-medium text-ink-3"><span>{group.label}</span><span className="tabular-nums">{group.rows.length}</span></h3>
      {(expanded[group.id] ? group.rows : group.rows.slice(0, 12)).map((item) => <button
        key={item.key}
        type="button"
        data-activity-key={item.key}
        className="flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2.5 text-left hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent"
        onClick={() => navigate(item.view)}
      >
        <Icon name={item.needsYou ? 'alert' : 'layers'} size={15} className={clsx('mt-0.5 shrink-0', item.needsYou ? 'text-warn' : 'text-ink-3')} />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-medium text-ink" title={item.title}>{item.title}</span>
          <span className="mt-0.5 block truncate text-[11px] text-ink-3">{tx(kindLabels[item.kind])}{item.projectName ? ` · ${item.projectName}` : ''}</span>
          <span className={clsx('mt-1 block text-[12px]', item.needsYou ? 'text-ink-2' : 'text-ink-3')}>{tx(reasonLabels[item.reason])}</span>
        </span>
        <Icon name="chevronRight" size={12} className="mt-1 shrink-0 text-ink-4" />
      </button>)}
      {group.rows.length > 12 && <button type="button" aria-expanded={!!expanded[group.id]} onClick={() => setExpanded((value) => ({ ...value, [group.id]: !value[group.id] }))} className="w-full rounded-lg px-2.5 py-2 text-left text-[12px] text-accent hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent">
        {expanded[group.id] ? tx("Show fewer") : tx("Show all {value0} items", { value0: group.rows.length })}
      </button>}
    </section>)}
    <div className="border-t border-line/60 pt-1">
      <button type="button" onClick={() => navigate({ name: 'commandCentre' })} className="flex w-full items-center justify-between rounded-lg px-2.5 py-2.5 text-left text-[13px] text-ink hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent">
        <span>{tx("Open full activity")}</span><Icon name="arrowRight" size={14} className="text-ink-3" />
      </button>
    </div>
  </div>
}
