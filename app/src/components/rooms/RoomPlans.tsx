import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import clsx from 'clsx'
import { useState } from 'react'
import { Icon } from '../../lib/icons'
import { api } from '../../lib/ipc'
import type { Job, Room, RoomEventItem, RoomMergeResult, Ticket } from '../../lib/types'
import {
  roomConflictBlockers,
  roomSupervisorPhase,
  ticketPhaseLabel,
  Avatar,
} from './helpers'
import { roomEventSnippet } from './RoomTimeline'

interface RoomHandoffQueueItem {
  event: RoomEventItem
  status: 'pending' | 'answered'
  answer?: RoomEventItem
}

function latestRoomEvent(events: RoomEventItem[], predicate: (event: RoomEventItem) => boolean): RoomEventItem | undefined {
  return [...events].reverse().find(predicate)
}

function buildRoomHandoffQueue(events: RoomEventItem[], limit = 6): RoomHandoffQueueItem[] {
  return [...events]
    .reverse()
    .filter((event) => (event.kind === 'ask' || event.kind === 'assign') && event.target && event.target !== 'all')
    .slice(0, limit)
    .map((event) => {
      const answer = events.find((candidate) =>
        candidate.createdAt > event.createdAt &&
        candidate.seat === event.target &&
        !['yield', 'notice', 'error'].includes(candidate.kind),
      )
      const status: RoomHandoffQueueItem['status'] = answer ? 'answered' : 'pending'
      return { event, answer, status }
    })
    .sort((a, b) => Number(a.status === 'answered') - Number(b.status === 'answered') || b.event.createdAt - a.event.createdAt)
}

function missingEvidenceTickets(tickets: Ticket[]): Ticket[] {
  return tickets.filter((ticket) => {
    const required = ticket.requiredEvidence ?? []
    if (!required.length) return false
    const meaningful = (ticket.evidence ?? []).filter((entry) => !/^Reserved for (?:build|review) worker\b/i.test(entry.summary.trim()))
    return meaningful.length < required.length
  })
}

function isFinalBlockEvent(event: RoomEventItem): boolean {
  return event.kind === 'check' && event.seat === 'moderator' && /^Final blocked:/i.test(event.text)
}

function roomPlanSnapshot({
  room,
  events,
  leftToWork,
  activeWorkers,
  ticketJobsByTicket,
  conflictBlockers,
  totalTickets,
}: {
  room: Room
  events: RoomEventItem[]
  leftToWork: Ticket[]
  activeWorkers: Ticket[]
  ticketJobsByTicket: Map<string, Job>
  conflictBlockers: ReturnType<typeof roomConflictBlockers>
  totalTickets: number
}) {
  const active = activeWorkers[0]
  const nextTicket = leftToWork[0]
  const latestQuestion = latestRoomEvent(events, (event) => event.kind === 'ask' && event.target !== 'all')
  const latestDecision = latestRoomEvent(events, (event) => ['final', 'vote', 'claim', 'check', 'object'].includes(event.kind) && !isFinalBlockEvent(event))
  const handoff = buildRoomHandoffQueue(events, 1).find((item) => item.status === 'pending')
  const phase = roomSupervisorPhase(leftToWork, activeWorkers, ticketJobsByTicket, totalTickets)
  const nextAction = conflictBlockers.length
    ? `Resolve ${conflictBlockers.length} conflict blocker${conflictBlockers.length === 1 ? '' : 's'}`
    : active
      ? `Track ${ticketPhaseLabel(active, ticketJobsByTicket.get(active.id)).toLowerCase()} for "${active.title}"`
      : handoff
        ? `${handoff.event.target} answers ${handoff.event.seat}`
        : nextTicket
          ? `Work "${nextTicket.title}"`
          : totalTickets > 0
            ? 'Ready for final gate'
            : 'Create the first executable ticket'

  return {
    objective: (room.problem || latestRoomEvent(events, (event) => event.kind === 'task')?.text || 'No room objective yet.').trim(),
    phase,
    nextAction,
    openQuestion: latestQuestion ? `${latestQuestion.seat} to ${latestQuestion.target}: ${roomEventSnippet(latestQuestion.text, 150)}` : '',
    latestDecision: latestDecision ? `${latestDecision.seat}: ${roomEventSnippet(latestDecision.text, 160)}` : '',
  }
}

export function RoomLivePlanPanel({
  room,
  events,
  leftToWork,
  activeWorkers,
  ticketJobsByTicket,
  conflictBlockers,
  totalTickets,
}: {
  room: Room
  events: RoomEventItem[]
  leftToWork: Ticket[]
  activeWorkers: Ticket[]
  ticketJobsByTicket: Map<string, Job>
  conflictBlockers: ReturnType<typeof roomConflictBlockers>
  totalTickets: number
}) {
  const tx = useAuthoredText()
  const plan = roomPlanSnapshot({ room, events, leftToWork, activeWorkers, ticketJobsByTicket, conflictBlockers, totalTickets })
  return (
    <div className="rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
      <div className="mb-2 flex items-center gap-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-ink-5 text-ink-2">
          <Icon name="list" size={14} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-semibold text-ink">{tx("Live room plan")}</div>
          <div className="truncate text-[12px] text-ink-4">{roomEventSnippet(plan.objective, 130)}</div>
        </div>
      </div>
      <div className="grid gap-2 md:grid-cols-2">
        <PlanCell label={tx("Phase")} value={plan.phase} accent />
        <PlanCell label={tx("Next action")} value={plan.nextAction} accent={!!leftToWork.length || !!activeWorkers.length || !!conflictBlockers.length} />
        <PlanCell label={tx("Open question")} value={plan.openQuestion || 'No direct asks pending.'} />
        <PlanCell label={tx("Latest decision")} value={plan.latestDecision || 'No durable decision yet.'} />
      </div>
    </div>
  )
}

function PlanCell({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  const tx = useAuthoredText()
  return (
    <div className="min-w-0 rounded-lg border border-line bg-card-2 px-2.5 py-2">
      <div className="text-[11px] font-semibold uppercase tracking-[0.06em] text-ink-4">{label}</div>
      <div className={clsx('mt-0.5 line-clamp-2 text-[13px] leading-snug', accent ? 'text-accent' : 'text-ink-2')}>{value}</div>
    </div>
  )
}

export function RoomHandoffQueuePanel({ events }: { events: RoomEventItem[] }) {
  const tx = useAuthoredText()
  const queue = buildRoomHandoffQueue(events)
  if (!queue.length) return null
  const pending = queue.filter((item) => item.status === 'pending').length
  return (
    <div className="border-b border-line px-3 py-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("Handoff queue")}</span>
        <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', pending ? 'bg-accent-soft text-accent' : 'bg-ink-5 text-ink-3')}>{pending} {tx("pending")}</span>
      </div>
      <div className="space-y-1.5">
        {queue.slice(0, 5).map(({ event, status, answer }) => (
          <div key={event.id} className="rounded-lg border border-line bg-card px-2 py-1.5">
            <div className="flex min-w-0 items-center gap-1.5 text-[12px]">
              <Avatar name={event.seat} size={16} />
              <span className="truncate font-medium text-ink-2">{event.seat}</span>
              <Icon name="arrowRight" size={10} className="shrink-0 text-ink-4" />
              <span className="truncate text-ink-3">{event.target}</span>
              <span className={clsx('ml-auto shrink-0 rounded px-1 py-0.5 text-[11px]', status === 'pending' ? 'bg-accent-soft text-accent' : 'bg-positive/12 text-positive')}>
                {status === 'pending' ? tx("Open") : tx("Answered")}
              </span>
            </div>
            <div className="mt-1 line-clamp-2 text-[12px] leading-snug text-ink-3">{roomEventSnippet(event.text, 130)}</div>
            {answer && <div className="mt-1 truncate border-t border-line pt-1 text-[12px] text-ink-4">{tx("Answered by ")}{answer.seat}: {roomEventSnippet(answer.text, 90)}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}

export function RoomFinalGatePanel({ tickets, leftToWork, conflictBlockers, room }: { tickets: Ticket[]; leftToWork: Ticket[]; conflictBlockers: ReturnType<typeof roomConflictBlockers>; room: Room }) {
  const tx = useAuthoredText()
  const [merging, setMerging] = useState(false)
  const [mergeResult, setMergeResult] = useState<RoomMergeResult | null>(null)
  const mergeWorktree = async () => {
    setMerging(true)
    try {
      setMergeResult(await api.mergeRoomWorktree(room.id))
    } catch {
      setMergeResult(null) // failures already surface as an IPC error toast
    } finally {
      setMerging(false)
    }
  }
  if (!tickets.length && !room.finalText && !conflictBlockers.length) return null
  const missingEvidence = missingEvidenceTickets(tickets)
  const reviewTickets = tickets.filter((ticket) => ticket.status === 'review')
  const complete = tickets.length > 0 && leftToWork.length === 0 && missingEvidence.length === 0 && reviewTickets.length === 0 && conflictBlockers.length === 0
  const merged = mergeResult?.status === 'merged' || mergeResult?.status === 'nothing'
  const rows = [
    { label: tx("Tickets"), value: tickets.length ? `${tickets.length - leftToWork.length}/${tickets.length} complete` : 'No tickets yet', ok: tickets.length > 0 && leftToWork.length === 0 },
    { label: tx("Evidence"), value: missingEvidence.length ? `${missingEvidence.length} missing required evidence` : 'Evidence requirements met', ok: missingEvidence.length === 0 },
    { label: tx("Review"), value: reviewTickets.length ? `${reviewTickets.length} waiting for review` : 'No review queue', ok: reviewTickets.length === 0 },
    { label: tx("Blockers"), value: conflictBlockers.length ? `${conflictBlockers.length} conflict blocker${conflictBlockers.length === 1 ? '' : 's'}` : 'No conflict blockers', ok: conflictBlockers.length === 0 },
  ]
  return (
    <div className={clsx('rounded-xl border px-3.5 py-3 shadow-card', complete ? 'border-positive/30 bg-positive/8' : 'border-line bg-card')}>
      <div className="mb-2 flex items-center gap-2">
        <span className={clsx('flex h-7 w-7 items-center justify-center rounded-lg', complete ? 'bg-positive/12 text-positive' : 'bg-ink-5 text-ink-2')}>
          <Icon name={complete ? 'check' : 'shield'} size={14} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-semibold text-ink">{tx("Final readiness")}</div>
          <div className="truncate text-[12px] text-ink-4">{complete ? tx("The room has enough signal to finalize.") : tx("The room still has gates to satisfy.")}</div>
        </div>
        <button
          onClick={() => void mergeWorktree()}
          disabled={!complete || merging || merged}
          title={complete ? tx("Merge the room worktree branches back into the project branch") : tx("Final gates must be green before merging")}
          className="flex h-7 shrink-0 items-center gap-1.5 rounded-lg border border-line px-2.5 text-[13px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-45"
        >
          <Icon name="branch" size={12} />
          {merging ? tx("Merging…") : merged ? tx("Merged") : tx("Merge worktree")}
        </button>
      </div>
      <div className="grid gap-1.5 sm:grid-cols-4">
        {rows.map((row) => (
          <div key={row.label} className="rounded-lg border border-line bg-card-2 px-2 py-1.5">
            <div className="flex items-center gap-1 text-[11px] font-semibold uppercase tracking-[0.06em] text-ink-4">
              <Icon name={row.ok ? 'check' : 'clock'} size={10} className={row.ok ? 'text-positive' : 'text-ink-4'} />
              {row.label}
            </div>
            <div className={clsx('mt-0.5 line-clamp-2 text-[12px] leading-snug', row.ok ? 'text-positive' : 'text-ink-3')}>{row.value}</div>
          </div>
        ))}
      </div>
      {mergeResult && (
        <div className={clsx('mt-2 rounded-lg border px-2.5 py-1.5 text-[13px] leading-snug', mergeResult.ok ? 'border-positive/30 bg-positive/8 text-positive' : 'border-danger/30 bg-danger/8 text-ink-2')}>
          {mergeResult.ok && (
            <span>{mergeResult.status === 'nothing' ? tx("No room branches to merge.") : tx("Merged {value0} into {value1}.", { value0: mergeResult.merged.join(', '), value1: mergeResult.baseBranch })}</span>
          )}
          {!mergeResult.ok && mergeResult.status === 'gated' && (
            <span>{tx("Cannot merge yet: ")}{mergeResult.blockers.join(' ')}</span>
          )}
          {!mergeResult.ok && mergeResult.status === 'conflict' && mergeResult.conflict && (
            <div>
              <div className="font-medium text-danger">{tx("Merge conflict on ")}{mergeResult.conflict.branch}{tx(". The merge was aborted and a ticket was opened for the builder seat.")}</div>
              {mergeResult.conflict.files.length > 0 && (
                <ul className="mt-1 list-inside list-disc text-ink-3">
                  {mergeResult.conflict.files.slice(0, 8).map((file) => <li key={file} className="truncate">{file}</li>)}
                  {mergeResult.conflict.files.length > 8 && <li>+{mergeResult.conflict.files.length - 8} {tx("more")}</li>}
                </ul>
              )}
              {mergeResult.remaining.length > 0 && <div className="mt-1 text-ink-4">{tx("Not merged: ")}{mergeResult.remaining.join(', ')}</div>}
            </div>
          )}
          {!mergeResult.ok && (mergeResult.status === 'blocked' || mergeResult.status === 'error') && (
            <span className="text-danger">{mergeResult.message}</span>
          )}
        </div>
      )}
    </div>
  )
}

const ROOM_MODE_PRESETS: Array<{
  id: string
  label: string
  detail: string
  icon: 'warning' | 'branch' | 'message' | 'check' | 'search'
  patch: Pick<Room, 'collaborationMode' | 'deliberationEffort'> & { workerBudget?: Room['workerBudget'] }
}> = [
  { id: 'bug_bash', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Bug bash") }, detail: 'Triage, isolate, fix, verify.', icon: 'warning', patch: { collaborationMode: 'balanced', deliberationEffort: 'normal', workerBudget: { maxActive: 2, maxStarts: 6 } } },
  { id: 'build_review', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Build and review") }, detail: 'Planner, builder, reviewer flow.', icon: 'branch', patch: { collaborationMode: 'frontier_plans_local_builds', deliberationEffort: 'normal', workerBudget: { maxActive: 2, maxStarts: 4 } } },
  { id: 'planning', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Planning council") }, detail: 'Explore options, then ticket.', icon: 'message', patch: { collaborationMode: 'frontier_council', deliberationEffort: 'deep', workerBudget: { maxActive: 1, maxStarts: 2 } } },
  { id: 'code_review', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Code review") }, detail: 'Evidence and risk first.', icon: 'check', patch: { collaborationMode: 'frontier_council', deliberationEffort: 'normal', workerBudget: { maxActive: 1, maxStarts: 3 } } },
  { id: 'research', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Research room") }, detail: 'Facts, sources, narrow tickets.', icon: 'search', patch: { collaborationMode: 'balanced', deliberationEffort: 'deep', workerBudget: { maxActive: 1, maxStarts: 2 } } },
  { id: 'supervised', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Supervised build") }, detail: 'Supervisor assigns, workers auto-start.', icon: 'branch', patch: { collaborationMode: 'supervisor', deliberationEffort: 'normal', workerBudget: { maxActive: 2, maxStarts: 4 } } },
]

export function RoomModePresets({ room, running, onApply }: { room: Room; running: boolean; onApply: (patch: Partial<Room>) => void }) {
  const tx = useAuthoredText()
  return (
    <div className="mt-3">
      <div className="mb-1.5 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("Room modes")}</div>
      <div className="grid grid-cols-2 gap-1.5">
        {ROOM_MODE_PRESETS.map((preset) => {
          const active = room.collaborationMode === preset.patch.collaborationMode && room.deliberationEffort === preset.patch.deliberationEffort
          return (
            <button
              key={preset.id}
              disabled={running}
              onClick={() => onApply(preset.patch)}
              className={clsx('min-w-0 rounded-lg border px-2 py-1.5 text-left transition-colors disabled:opacity-55', active ? 'border-line-strong bg-card text-ink' : 'border-line bg-card-2 text-ink-3 hover:border-line-strong hover:text-ink')}
            >
              <div className="flex min-w-0 items-center gap-1.5 text-[13px] font-medium">
                <Icon name={preset.icon} size={12} className="shrink-0" />
                <span className="truncate">{preset.label}</span>
              </div>
              <div className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-ink-4">{preset.detail}</div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
