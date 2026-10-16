import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { basename, relativeTime } from '../../lib/format'
import { Icon, type IconName } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import { Button, Input } from '../ui'
import { allowedProviders, projectDefaultModel } from '../../lib/policy'
import { accountLabel, modelDisplayName, providerAccountChoices, providerModelGroups } from '../../lib/provider-accounts'
import { buildRoomAgentTree, type RoomAgentTreeNode } from '../../lib/room-agent-tree'
import type { AgentPreset, AgentRoleProfileId, FanOutBudget, Job, Project, Provider, Room, RoomCollaborationMode, RoomEventItem, RoomMode, RoomSeatLane, RoomState, Seat, Ticket } from '../../lib/types'
import {
  ACTIVE_ROOM_STATES,
  FAN_OUT_OPTIONS,
  FILE_TOOL_VERB,
  ROLE_PROFILES,
  ROOM_COLLABORATION_OPTIONS,
  ROOM_MODE_OPTIONS,
  ROOM_STATE_LABEL,
  Avatar,
  availableSeatName,
  presetSeats,
  seatForRole,
  isActiveRunState,
  modelChoiceValue,
  parseModelChoice,
  pickProvider,
  roomJobDetail,
  roomJobTitle,
  roomSupervisorPhase,
  seatModelChoice,
  ticketFileRefsForDisplay,
  ticketPhaseLabel,
  ticketStatusLabel,
  ticketWorkerLabel,
} from './helpers'
import type { RoomConflictBlocker } from './helpers'

export function RoomSupervisorPanel({
  room,
  roomState,
  roomJob,
  leftToWork,
  totalTickets,
  activeWorkers,
  ticketJobsByTicket,
  conflictBlockers,
  budgetLabel,
  running,
  starting,
  canWake,
  onWake,
  onStop,
}: {
  room: Room
  roomState: RoomState
  roomJob: Job | null
  leftToWork: Ticket[]
  totalTickets: number
  activeWorkers: Ticket[]
  ticketJobsByTicket: Map<string, Job>
  conflictBlockers: RoomConflictBlocker[]
  budgetLabel: string
  running: boolean
  starting: boolean
  canWake: boolean
  onWake: () => void
  onStop: () => void
}) {
  const blocked = conflictBlockers.length > 0 || roomState === 'waiting_user' || roomState === 'waiting_external' || roomJob?.status === 'waiting_approval' || roomJob?.status === 'stalled'
  const stateTone = roomState === 'failed' || roomState === 'error' || roomJob?.status === 'failed'
    ? 'text-danger'
    : blocked
      ? 'text-warn'
      : running
        ? 'text-accent'
        : roomState === 'complete'
          ? 'text-positive'
          : 'text-ink-3'
  return (
    <div className="rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
      <div className="flex items-start gap-3">
        <span className={clsx('mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-5', stateTone)}>
          <Icon name={blocked ? 'warning' : running ? 'play' : 'layers'} size={16} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <div className="text-[13px] font-semibold text-ink">Room supervisor</div>
                <span className={clsx('rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium', stateTone)}>{ROOM_STATE_LABEL[roomState]}</span>
              </div>
              <div className="mt-0.5 truncate text-[13px] text-ink-3">{roomJobTitle(roomJob)}</div>
              {roomJobDetail(roomJob) && <div className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-ink-4">{roomJobDetail(roomJob)}</div>}
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              {!running && canWake && (
                <Button size="xs" variant="default" icon="play" onClick={onWake} disabled={starting || room.seats.length === 0}>
                  {starting ? 'Waking...' : 'Wake'}
                </Button>
              )}
              {running && (
                <Button size="xs" variant="danger" icon="stop" onClick={onStop}>
                  Stop
                </Button>
              )}
            </div>
          </div>

          <div className="mt-3 grid grid-cols-4 gap-2 text-[12px]">
            <SupervisorMetric label="Phase" value={roomSupervisorPhase(leftToWork, activeWorkers, ticketJobsByTicket, totalTickets)} accent={running} />
            <SupervisorMetric label="Left" value={`${leftToWork.length}/${totalTickets}`} accent={leftToWork.length > 0} />
            <SupervisorMetric label="Workers" value={String(activeWorkers.length)} accent={activeWorkers.length > 0} />
            <SupervisorMetric label="Budget" value={budgetLabel} />
          </div>

          {activeWorkers.length > 0 && (
            <div className="mt-3 border-t border-line pt-2">
              <div className="mb-1 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Active workers</div>
              <div className="space-y-1">
                {activeWorkers.slice(0, 3).map((ticket) => (
                  <div key={ticket.id} className="flex min-w-0 items-center gap-1.5 text-[13px]">
                    <Icon name={ticket.runState === 'queued' ? 'clock' : 'play'} size={11} className="shrink-0 text-accent" />
                    <span className="min-w-0 flex-1 truncate text-ink-2">{ticket.title}</span>
                    <span className="shrink-0 rounded bg-ink-5 px-1 py-0.5 text-[11px] text-ink-3">{ticketPhaseLabel(ticket, ticketJobsByTicket.get(ticket.id))}</span>
                    <span className="shrink-0 text-[12px] text-ink-4">{ticketWorkerLabel(ticket)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-3 border-t border-line pt-2">
            <div className="mb-1 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Conflict blockers</div>
            {conflictBlockers.length > 0 ? (
              <div className="space-y-1">
                {conflictBlockers.slice(0, 3).map((blocker) => (
                  <div key={`${blocker.ticketId}:${blocker.text}`} className="flex min-w-0 gap-1.5 text-[13px] leading-snug text-warn">
                    <Icon name="warning" size={11} className="mt-0.5 shrink-0" />
                    <span className="min-w-0"><span className="font-medium">{blocker.ticketTitle}</span>: {blocker.text}</span>
                  </div>
                ))}
                {conflictBlockers.length > 3 && <div className="text-[12px] text-ink-4">+{conflictBlockers.length - 3} more</div>}
              </div>
            ) : (
              <div className="text-[13px] text-ink-4">No conflict blockers.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function SupervisorMetric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-lg bg-card-2 px-2 py-1.5">
      <div className="text-[11px] uppercase tracking-[0.06em] text-ink-4">{label}</div>
      <div className={clsx('mt-0.5 truncate text-[13px] font-semibold tabular-nums', accent ? 'text-accent' : 'text-ink')} title={value}>{value}</div>
    </div>
  )
}

export function RoomTicketQueue({ tickets, total, blockers, ticketJobsByTicket }: { tickets: Ticket[]; total: number; blockers: RoomConflictBlocker[]; ticketJobsByTicket: Map<string, Job> }) {
  const blockersByTicket = new Map<string, string[]>()
  for (const blocker of blockers) blockersByTicket.set(blocker.ticketId, [...(blockersByTicket.get(blocker.ticketId) ?? []), blocker.text])
  if (total === 0) return <div className="mt-3 rounded-lg border border-dashed border-line px-2.5 py-4 text-center text-[13px] text-ink-3">No room tickets yet.</div>
  if (tickets.length === 0) return <div className="mt-3 rounded-lg border border-line bg-card px-2.5 py-3 text-center text-[13px] text-positive">All room tickets complete.</div>
  return (
    <div className="mt-3">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Left to work</span>
        <span className="rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-4">{tickets.length}</span>
      </div>
      <div className="space-y-1.5">
        {tickets.slice(0, 6).map((ticket) => <RoomTicketQueueItem key={ticket.id} ticket={ticket} job={ticketJobsByTicket.get(ticket.id)} blockers={blockersByTicket.get(ticket.id) ?? []} />)}
      </div>
      {tickets.length > 6 && <div className="mt-1.5 text-[12px] text-ink-4">+{tickets.length - 6} more ticket{tickets.length - 6 === 1 ? '' : 's'}</div>}
    </div>
  )
}

function RoomTicketQueueItem({ ticket, job, blockers }: { ticket: Ticket; job?: Job; blockers: string[] }) {
  const worker = ticketWorkerLabel(ticket)
  const refs = ticketFileRefsForDisplay(ticket).slice(0, 2)
  const required = ticket.requiredEvidence ?? []
  const evidence = ticket.evidence ?? []
  const caps = ticket.allowedCapabilities ?? []
  const statusTone = ticket.status === 'needs_clarification'
    ? 'text-danger'
    : ticket.status === 'review'
      ? 'text-positive'
      : ticket.status === 'in_progress' || isActiveRunState(ticket.runState)
        ? 'text-accent'
        : 'text-ink-3'
  return (
    <div className="rounded-lg border border-line bg-card px-2 py-1.5">
      <div className="flex min-w-0 items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium text-ink-2">{ticket.title}</div>
          <div className="mt-0.5 flex min-w-0 items-center gap-1.5 text-[12px]">
            <span className="shrink-0 rounded bg-ink-5 px-1 py-0.5 text-[11px] font-medium text-ink-3">{ticketPhaseLabel(ticket, job)}</span>
            <span className={clsx('shrink-0 capitalize', statusTone)}>{ticketStatusLabel(ticket.status)}</span>
            {worker && <span className="min-w-0 truncate text-ink-4">{worker}</span>}
          </div>
        </div>
        {blockers.length > 0 && <span className="shrink-0 rounded bg-warn/12 px-1.5 py-0.5 text-[11px] font-medium text-warn">{blockers.length}</span>}
      </div>
      <div className="mt-1 flex items-center gap-1.5 text-[11px] text-ink-4">
        <span className="capitalize">risk {ticket.risk ?? 'low'}</span>
        {caps.length ? <span className="truncate">caps {caps.slice(0, 2).join(', ')}{caps.length > 2 ? ` +${caps.length - 2}` : ''}</span> : null}
        {required.length ? <span className="ml-auto tabular-nums">ev {evidence.length}/{required.length}</span> : null}
      </div>
      {refs.length > 0 && (
        <div className="mt-1 flex min-w-0 flex-wrap gap-1">
          {refs.map((ref) => (
            <span key={ref} className="flex max-w-full items-center gap-1 rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-3">
              <FileGlyph path={ref} size={10} />
              <span className="truncate">{basename(ref)}</span>
            </span>
          ))}
        </div>
      )}
      {blockers.length > 0 && (
        <div className="mt-1 space-y-0.5 border-t border-line pt-1 text-[12px] leading-snug text-warn">
          {blockers.slice(0, 2).map((blocker) => <div key={blocker} className="line-clamp-2">{blocker}</div>)}
        </div>
      )}
    </div>
  )
}

export function RoomFlowTicker({ room, events, live, running }: { room: Room; events: RoomEventItem[]; live?: { seat: string; text: string }; running: boolean }) {
  const tick = events.reduce((max, event) => Math.max(max, event.tick ?? 0), 0)
  const latest = [...events].reverse().find((event) => event.seat !== 'you')
  const activeSeat = live?.seat ?? (running ? latest?.seat : null)
  const sleeping = room.state === 'waiting_external' || room.state === 'waiting_user'
  const label = live ? `${live.seat} is speaking` : running ? 'Room is flowing' : sleeping ? 'Room sleeping' : room.state === 'complete' ? 'Room complete' : 'Room idle'
  return (
    <div className="rounded-xl border border-line bg-card px-3.5 py-2.5 shadow-card">
      <div className="flex items-center gap-2">
        <span className={clsx('h-2 w-2 rounded-full', running ? 'animate-pulse bg-accent' : sleeping ? 'bg-warn' : room.state === 'complete' ? 'bg-positive' : 'bg-ink-4')} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[13px] font-medium text-ink">
            <span>{label}</span>
            <span className="rounded bg-ink-5 px-1.5 py-0.5 font-mono text-[12px] text-ink-3">tick {tick}</span>
          </div>
          <div className="mt-1 flex items-center gap-1.5 overflow-hidden">
            {room.seats.map((seat) => (
              <span key={seat.id} className={clsx('flex min-w-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[12px]', activeSeat === seat.name ? 'border-line-strong bg-ink-5 text-ink' : 'border-line bg-card-2 text-ink-3')}>
                <span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', activeSeat === seat.name ? 'bg-accent' : 'bg-ink-4')} />
                <span className="truncate">{seat.name}</span>
              </span>
            ))}
          </div>
        </div>
        {latest && <div className="hidden max-w-[160px] truncate text-right text-[12px] text-ink-4 sm:block">{latest.kind} · {latest.seat}</div>}
      </div>
    </div>
  )
}

export function RoomStatePill({ state }: { state: RoomState }) {
  const active = ACTIVE_ROOM_STATES.has(state)
  const danger = state === 'failed' || state === 'error'
  const done = state === 'complete'
  return (
    <span className={clsx('ml-2 flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[12px] font-medium',
      danger ? 'bg-danger/10 text-danger' : done ? 'bg-positive/12 text-positive' : active ? 'bg-accent-soft text-accent' : 'bg-ink-5 text-ink-3')}
    >
      <span className={clsx('h-1.5 w-1.5 rounded-full', active && 'animate-pulse', danger ? 'bg-danger' : done ? 'bg-positive' : active ? 'bg-accent' : 'bg-ink-4')} />
      {ROOM_STATE_LABEL[state]}
    </span>
  )
}

export function RoomStat({ icon, label, value, accent }: { icon: 'cpu' | 'list' | 'file' | 'message'; label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-card px-2 py-1.5">
      <div className="flex items-center gap-1.5 text-[12px] text-ink-4">
        <Icon name={icon} size={11} />
        <span>{label}</span>
      </div>
      <div className={clsx('mt-0.5 text-[13px] font-semibold', accent ? 'text-accent' : 'text-ink')}>{value}</div>
    </div>
  )
}

const AGENT_NODE_STATUS_LABEL: Record<RoomAgentTreeNode['status'], string> = {
  idle: 'Idle',
  active: 'Active',
  done: 'Done',
  failed: 'Failed',
}

function AgentStatusDot({ status }: { status: RoomAgentTreeNode['status'] }) {
  return (
    <span
      className={clsx(
        'h-1.5 w-1.5 shrink-0 rounded-full',
        status === 'failed' ? 'bg-danger' : status === 'done' ? 'bg-positive' : status === 'active' ? 'bg-accent' : 'bg-ink-4',
      )}
      title={AGENT_NODE_STATUS_LABEL[status]}
    />
  )
}

function RoomAgentDelegates({ node }: { node: RoomAgentTreeNode | undefined }) {
  if (!node?.children.length) return null
  return (
    <div className="mt-2 border-t border-line pt-1.5">
      <div className="mb-1 flex items-center gap-1.5 text-[12px] font-medium uppercase tracking-wide text-ink-3">
        <Icon name="branch" size={11} />
        <span>Delegates</span>
        <span className="ml-auto rounded bg-ink-5 px-1 text-[11px] text-ink-4">{node.children.length}</span>
      </div>
      <div className="space-y-1">
        {node.children.map((child) => <RoomAgentDelegateRow key={child.id} node={child} />)}
      </div>
    </div>
  )
}

function RoomAgentDelegateRow({ node }: { node: RoomAgentTreeNode }) {
  return (
    <div className="min-w-0">
      <div className="flex min-w-0 items-start gap-1.5 rounded-md bg-card-2 px-1.5 py-1">
        <AgentStatusDot status={node.status} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <span title={node.name} className="truncate text-[12px] font-medium text-ink-2">{node.name}</span>
            {node.children.length > 0 && <span className="shrink-0 rounded bg-ink-5 px-1 text-[11px] tabular-nums text-ink-4">{node.children.length}</span>}
            <span className="shrink-0 text-[11px] text-ink-4">{AGENT_NODE_STATUS_LABEL[node.status]}</span>
          </div>
          <div title={node.detail} className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-ink-3">{node.detail}</div>
        </div>
      </div>
      {node.children.length > 0 && (
        <div className="ml-2 mt-1 space-y-1 border-l border-line pl-2">
          {node.children.map((child) => <RoomAgentDelegateRow key={child.id} node={child} />)}
        </div>
      )}
    </div>
  )
}

function roomSeatFocus(room: Room, tickets: Ticket[], seat: Seat, node: RoomAgentTreeNode | undefined): { label: string; detail: string; icon: IconName; tone: 'active' | 'blocked' | 'done' | 'idle' } {
  const heartbeat = room.seatHeartbeats?.[seat.id]
  if (heartbeat?.missedSince) {
    return { label: 'Stalled', detail: `missed heartbeat ${relativeTime(heartbeat.missedSince)}`, icon: 'warning', tone: 'blocked' }
  }

  const pendingHandoff = [...room.events].reverse().find((event) => {
    if (!(event.kind === 'ask' || event.kind === 'assign') || event.target !== seat.name) return false
    return !room.events.some((candidate) =>
      candidate.createdAt > event.createdAt &&
      candidate.seat === seat.name &&
      !['yield', 'notice', 'error'].includes(candidate.kind),
    )
  })
  if (pendingHandoff) {
    return { label: 'Answering handoff', detail: `${pendingHandoff.seat} is waiting`, icon: 'message', tone: 'active' }
  }

  const claimed = tickets.find((t) => t.assignee === seat.name && t.projectId === room.projectId && t.roomId === room.id && (t.status === 'in_progress' || t.status === 'review' || isActiveRunState(t.runState)))
  if (claimed) {
    const running = isActiveRunState(claimed.runState)
    return { label: running ? 'Worker running' : claimed.status === 'review' ? 'Reviewing' : 'Working ticket', detail: claimed.title, icon: running ? 'play' : 'check', tone: 'active' }
  }

  if (node?.children.some((child) => child.status === 'active')) {
    return { label: 'Delegated', detail: `${node.children.filter((child) => child.status === 'active').length} active delegate${node.children.filter((child) => child.status === 'active').length === 1 ? '' : 's'}`, icon: 'branch', tone: 'active' }
  }

  const last = [...room.events].reverse().find((event) => event.seat === seat.name && event.kind !== 'yield')
  if (last) {
    const label = last.kind === 'tool' ? `Used ${last.tool ?? 'tool'}` : last.kind === 'claim' ? 'Claimed work' : last.kind === 'final' ? 'Concluded' : `${last.kind[0]?.toUpperCase() ?? ''}${last.kind.slice(1)}`
    return { label, detail: last.text.replace(/\s+/g, ' ').trim().slice(0, 90), icon: last.kind === 'tool' ? 'grid' : last.kind === 'final' ? 'check' : 'spark', tone: last.kind === 'final' ? 'done' : 'idle' }
  }

  return { label: 'Available', detail: seat.lane ? `${seat.lane} lane` : 'Ready for a handoff', icon: 'clock', tone: 'idle' }
}

function SeatFocusLine({ focus }: { focus: ReturnType<typeof roomSeatFocus> }) {
  return (
    <div className={clsx(
      'mt-1 flex min-w-0 items-center gap-1.5 rounded-md px-1.5 py-1 text-[12px]',
      focus.tone === 'blocked' ? 'bg-warn/10 text-warn' : focus.tone === 'active' ? 'bg-accent-soft text-accent' : focus.tone === 'done' ? 'bg-positive/12 text-positive' : 'bg-ink-5 text-ink-3',
    )}>
      <Icon name={focus.icon} size={11} className="shrink-0" />
      <span className="shrink-0 font-medium">{focus.label}</span>
      <span className="min-w-0 truncate opacity-80">{focus.detail}</span>
    </div>
  )
}

export function RoomAgentsSection({
  room,
  providers,
  tickets,
  running,
  adding,
  setAdding,
  setSeats,
}: {
  room: Room
  providers: Provider[]
  tickets: Ticket[]
  running: boolean
  adding: boolean
  setAdding: Dispatch<SetStateAction<boolean>>
  setSeats: (seats: Seat[]) => void
}) {
  const modelGroups = providers.flatMap((provider) => providerModelGroups(provider).map((group) => ({ provider, group })))
  const agentTree = useMemo(() => buildRoomAgentTree(room), [room])
  const agentTreeBySeat = useMemo(() => new Map(agentTree.map((node) => [node.seatName, node])), [agentTree])
  const updateSeatModel = (seat: Seat, value: string) => {
    const picked = parseModelChoice(value)
    if (!picked) return
    setSeats(room.seats.map((candidate) => candidate.id === seat.id ? { ...candidate, ...picked } : candidate))
  }
  const removeSeat = (seatId: string) => setSeats(room.seats.filter((seat) => seat.id !== seatId))

  return (
    <div className="border-b border-line px-3 py-3">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <div>
          <div className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Room agents</div>
          <div className="mt-0.5 text-[12px] leading-snug text-ink-4">Pick the model each local or cloud agent uses.</div>
        </div>
        <button onClick={() => setAdding((v) => !v)} disabled={running} className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-45" aria-label="Add agent"><Icon name="plus" size={16} /></button>
      </div>
      <div className="space-y-1.5">
        {room.seats.map((seat) => {
          const claimed = tickets.find((t) => t.assignee === seat.name && t.projectId === room.projectId && t.roomId === room.id && (t.status === 'in_progress' || t.status === 'review'))
          const recentFiles = [...new Set([...room.events].reverse().filter((ev) => ev.seat === seat.name && ev.tool && ev.path && FILE_TOOL_VERB[ev.tool]).map((ev) => ev.path!))].slice(0, 2)
          const seatProvider = providers.find((p) => p.id === seat.providerId)
          const choiceValue = seatModelChoice(seat, providers)
          const treeNode = agentTreeBySeat.get(seat.name)
          const focus = roomSeatFocus(room, tickets, seat, treeNode)
          return (
            <div key={seat.id} className="group rounded-lg border border-line bg-card px-2.5 py-2">
              <div className="flex items-start gap-2">
                <Avatar name={seat.name} size={24} />
                <div className="min-w-0 flex-1">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="truncate text-[13px] font-medium text-ink">{seat.name}</span>
                    {seat.roleProfileId && <span className="shrink-0 rounded bg-ink-5 px-1.5 py-0.5 text-[11px] uppercase tracking-wide text-ink-3">{seat.roleProfileId}</span>}
                  </div>
                  <div className="mt-0.5 truncate text-[12px] text-ink-3">{seat.lane ? `${seat.lane} · ` : ''}{seat.role}</div>
                </div>
                <button onClick={() => removeSeat(seat.id)} disabled={running} className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-ink-3 opacity-0 transition-opacity hover:text-danger disabled:opacity-30 group-hover:opacity-100" aria-label="Remove agent"><Icon name="x" size={14} /></button>
              </div>
              <SeatFocusLine focus={focus} />
              <select
                value={choiceValue}
                disabled={running || modelGroups.length === 0}
                onChange={(e) => updateSeatModel(seat, e.target.value)}
                className="no-drag mt-2 h-8 w-full cursor-pointer rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-55"
              >
                {!choiceValue && <option value="">Choose model</option>}
                {modelGroups.map(({ provider, group }) => (
                  <optgroup key={group.id} label={group.detail ? `${group.label} · ${group.detail}` : group.label}>
                    {group.models.map((model) => (
                      <option key={`${group.id}:${model}`} value={modelChoiceValue(provider.id, group.accountId, model)}>
                        {modelDisplayName(provider, model)}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
              {seatProvider && <div className="mt-1 truncate text-[12px] text-ink-4">{seatProvider.name}</div>}
              {claimed && <div className="mt-1 flex items-center gap-1 truncate text-[12px] text-accent"><Icon name="play" size={9} />{claimed.title}</div>}
              {recentFiles.length > 0 && (
                <div className="mt-1 space-y-0.5 border-t border-line pt-1">
                  {recentFiles.map((path) => <div key={path} className="flex items-center gap-1.5 truncate text-[12px] text-ink-3"><FileGlyph path={path} size={11} /><span className="truncate">{basename(path)}</span></div>)}
                </div>
              )}
              <RoomAgentDelegates node={treeNode} />
            </div>
          )
        })}
        {adding && <AddSeat room={room} onAdd={(seat) => { setSeats([...room.seats, seat]); setAdding(false) }} onCancel={() => setAdding(false)} />}
        {room.seats.length === 0 && !adding && <div className="rounded-lg border border-dashed border-line px-2.5 py-5 text-center text-[13px] text-ink-3">No agents yet. Add one or use the preset below.</div>}
      </div>
    </div>
  )
}

// #52/#54 — the center "build your room" canvas. Drag a role into the room (or
// click) to add an agent with a fitting model, then pick how they collaborate.
// Shown before the huddle has a transcript, so room-building happens in the
// middle of the screen rather than buried in the sidebar.
export function RoomCanvas({ room, providers, project, onSetSeats, onSetCollaboration, onSetMode }: {
  room: Room
  providers: Provider[]
  project: Project | null | undefined
  onSetSeats: (seats: Seat[]) => void
  onSetCollaboration: (mode: RoomCollaborationMode) => void
  onSetMode: (mode: RoomMode) => void
}) {
  const [dragOver, setDragOver] = useState(false)
  const noProviders = allowedProviders(providers, project).length === 0
  const activeMode = room.collaborationMode ?? 'frontier_plans_local_builds'
  const addRole = (profileId: AgentRoleProfileId) => {
    const seat = seatForRole(profileId, room.seats, providers, project)
    if (seat) onSetSeats([...room.seats, seat])
  }
  const removeSeat = (seatId: string) => onSetSeats(room.seats.filter((s) => s.id !== seatId))

  return (
    <div className="mx-auto w-full max-w-[640px] py-8 fade-in">
      <h2 className="text-[16px] font-semibold text-ink">Build your room</h2>
      <p className="mt-1 text-[13px] text-ink-3">Drag a role into the room — or click to add. Each agent picks a fitting model: builders and operators run local, planners and reviewers run on a frontier model.</p>

      <div className="mt-4 flex flex-wrap gap-2">
        {ROLE_PROFILES.map((p) => (
          <button
            key={p.id}
            draggable={!noProviders}
            onDragStart={(e) => { e.dataTransfer.setData('text/role', p.id); e.dataTransfer.effectAllowed = 'copy' }}
            onClick={() => addRole(p.id)}
            disabled={noProviders}
            className="flex cursor-grab items-center gap-1.5 rounded-full border border-line bg-card px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink active:cursor-grabbing disabled:opacity-50"
            title={`Add a ${p.label}`}
          >
            <Icon name={p.lane === 'development' || p.lane === 'operations' ? 'cpu' : 'globe'} size={13} className="text-ink-3" />
            {p.label}
            <Icon name="plus" size={12} className="text-ink-4" />
          </button>
        ))}
        <button onClick={() => onSetSeats(presetSeats(providers, project))} disabled={noProviders} className="flex items-center gap-1.5 rounded-full border border-dashed border-line px-3 py-1.5 text-[13px] text-ink-3 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-50" title="Add a planner, builder and reviewer">
          <Icon name="spark" size={13} />Recommended trio
        </button>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false) }}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); const role = e.dataTransfer.getData('text/role') as AgentRoleProfileId; if (role) addRole(role) }}
        className={clsx('mt-3 rounded-2xl border-2 border-dashed p-4 transition-colors', dragOver ? 'border-accent bg-accent-soft/40' : 'border-line')}
      >
        {room.seats.length === 0 ? (
          <div className="py-8 text-center text-[13px] text-ink-3">{noProviders ? 'No models configured — connect one in Settings first.' : 'Drop agents here to staff the room.'}</div>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {room.seats.map((seat) => {
              const prov = providers.find((p) => p.id === seat.providerId)
              return (
                <div key={seat.id} className="group relative flex items-center gap-2 rounded-xl border border-line bg-card-2 px-2.5 py-2">
                  <Avatar name={seat.name} size={26} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13px] font-medium text-ink">{seat.name}</div>
                    <div className="truncate text-[12px] text-ink-3">{seat.role}</div>
                    <div className="truncate text-[11px] text-ink-4">{prov ? modelDisplayName(prov, seat.model) : seat.model}</div>
                  </div>
                  <button onClick={() => removeSeat(seat.id)} className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded text-ink-3 opacity-0 transition-opacity hover:bg-ink-5 hover:text-ink group-hover:opacity-100" aria-label="Remove agent"><Icon name="x" size={12} /></button>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {room.seats.length > 0 && (
        <>
          <div className="mb-2 mt-6 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">What this room is for</div>
          <div className="grid gap-2 sm:grid-cols-3">
            {ROOM_MODE_OPTIONS.map((opt) => (
              <button key={opt.id} onClick={() => onSetMode(opt.id)} className={clsx('rounded-xl border px-3 py-2.5 text-left transition-colors', room.mode === opt.id ? 'border-line-strong bg-card' : 'border-line bg-card-2 hover:border-line-strong')}>
                <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink">{room.mode === opt.id && <Icon name="check" size={13} className="text-accent" />}{opt.label}</div>
                <div className="mt-0.5 text-[12px] leading-snug text-ink-3">{opt.detail}</div>
              </button>
            ))}
          </div>
          <div className="mb-2 mt-6 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">How they work together</div>
          <div className="grid gap-2 sm:grid-cols-2">
            {ROOM_COLLABORATION_OPTIONS.map((opt) => (
              <button key={opt.id} onClick={() => onSetCollaboration(opt.id)} className={clsx('rounded-xl border px-3 py-2.5 text-left transition-colors', activeMode === opt.id ? 'border-line-strong bg-card' : 'border-line bg-card-2 hover:border-line-strong')}>
                <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink">{activeMode === opt.id && <Icon name="check" size={13} className="text-accent" />}{opt.label}</div>
                <div className="mt-0.5 text-[12px] leading-snug text-ink-3">{opt.detail}</div>
              </button>
            ))}
          </div>
          <p className="mt-4 text-center text-[13px] text-ink-3">Pose a problem below to start the huddle.</p>
        </>
      )}
    </div>
  )
}

function AddSeat({ room, onAdd, onCancel }: { room: Room; onAdd: (s: Seat) => void; onCancel: () => void }) {
  const settings = useStore((s) => s.settings)!
  const project = useStore((s) => s.projects.find((p) => p.id === room.projectId)) ?? null
  const agentPresets = (settings.agentPresets ?? []).filter((preset) => !preset.archivedAt && !preset.paused)
  const pickerProviders = allowedProviders(settings.providers, project)
  const def = projectDefaultModel(project, settings.providers)
  // Name seats after their role (planner, planner-2, …) so the chosen role is
  // visible in the name, and suggest a fitting model per role (#47): builders and
  // operators run local; planners, reviewers and researchers run on a frontier
  // (cloud) model.
  const roleSeatName = (base: string) => {
    const used = new Set(room.seats.map((s) => s.name.toLowerCase()))
    if (!used.has(base.toLowerCase())) return base
    for (let i = 2; i < 99; i++) if (!used.has(`${base}-${i}`)) return `${base}-${i}`
    return base
  }
  const suggestModelFor = (laneFor: RoomSeatLane) =>
    pickProvider(pickerProviders, laneFor === 'development' || laneFor === 'operations' ? 'local' : 'cloud', def)
  const [roleProfileId, setRoleProfileId] = useState<AgentRoleProfileId>('builder')
  const [name, setName] = useState(() => roleSeatName('builder'))
  const [nameTouched, setNameTouched] = useState(false)
  const [role, setRole] = useState('Engineer')
  const [lane, setLane] = useState<RoomSeatLane>('development')
  const [fanOutBudget, setFanOutBudget] = useState<FanOutBudget>('single')
  // Seed the form with the model that fits the default 'builder' role (local).
  const initialPick = suggestModelFor('development')
  const [providerId, setProviderId] = useState(initialPick.provider?.id || def.providerId || pickerProviders[0]?.id || '')
  const provider = pickerProviders.find((p) => p.id === providerId) || pickerProviders[0]
  const [providerAccountId, setProviderAccountId] = useState<string | null>(initialPick.provider ? initialPick.providerAccountId : (project?.defaultProviderAccountId ?? provider?.defaultAccountId ?? null))
  const accountChoices = provider ? providerAccountChoices(provider) : []
  const selectedAccount = accountChoices.find((account) => account.id === (providerAccountId ?? null)) ?? accountChoices[0]
  const modelChoices = selectedAccount?.models?.length ? selectedAccount.models : (provider?.models ?? [])
  const [model, setModel] = useState(initialPick.model || def.model || modelChoices[0] || '')
  const selectedAccountLabel = provider ? accountLabel(provider, providerAccountId) : null
  const noAllowedProviders = pickerProviders.length === 0
  const currentModelChoice = provider && model ? modelChoiceValue(provider.id, providerAccountId, model) : ''

  useEffect(() => {
    if (!provider) return
    const account = providerAccountChoices(provider).find((choice) => choice.id === (providerAccountId ?? null)) ?? providerAccountChoices(provider)[0]
    if ((providerAccountId ?? null) !== (account?.id ?? null)) setProviderAccountId(account?.id ?? null)
    const models = account?.models?.length ? account.models : provider.models
    if (models.length && !models.includes(model)) setModel(models[0])
  }, [provider?.id, providerAccountId, model, pickerProviders.length]) // eslint-disable-line

  const add = () => {
    if (!provider || !model) return
    const n = (name.trim() || role.toLowerCase()).replace(/\s+/g, '-')
    onAdd({ id: crypto.randomUUID(), name: n, role: role.trim() || 'Agent', lane, roleProfileId, fanOutBudget, providerId: provider.id, providerAccountId: providerAccountId ?? null, model })
  }
  const pickModel = (value: string) => {
    const picked = parseModelChoice(value)
    if (!picked) return
    setProviderId(picked.providerId)
    setProviderAccountId(picked.providerAccountId)
    setModel(picked.model)
  }
  const applyAgentPreset = (presetId: string) => {
    const preset = agentPresets.find((candidate) => candidate.id === presetId)
    if (!preset) return
    setName(availableSeatName(preset.name, room.seats))
    setNameTouched(true)
    setRole(preset.role)
    setLane(preset.lane)
    setProviderId(preset.providerId)
    setProviderAccountId(preset.providerAccountId ?? null)
    setModel(preset.model)
  }

  return (
    <div className="mb-1.5 rounded-lg border border-line bg-card-2 p-2.5">
      {agentPresets.length > 0 && (
        <select
          defaultValue=""
          onChange={(e) => applyAgentPreset(e.target.value)}
          className="no-drag mb-2 h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent"
        >
          <option value="" disabled>Use saved agent...</option>
          {agentPresets.map((preset: AgentPreset) => <option key={preset.id} value={preset.id}>{preset.name} · {preset.role}</option>)}
        </select>
      )}
      <select
        value={roleProfileId}
        onChange={(e) => {
          const next = ROLE_PROFILES.find((profile) => profile.id === e.target.value) ?? ROLE_PROFILES[1]
          setRoleProfileId(next.id)
          setRole(next.role)
          setLane(next.lane)
          // Reflect the role in the name unless the user typed their own.
          if (!nameTouched) setName(roleSeatName(next.name))
          // Suggest a model that fits the role (local for builders, cloud for planners…).
          const pick = suggestModelFor(next.lane)
          if (pick.provider && pick.model) { setProviderId(pick.provider.id); setProviderAccountId(pick.providerAccountId); setModel(pick.model) }
        }}
        className="no-drag mb-2 h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent"
      >
        {ROLE_PROFILES.map((profile) => <option key={profile.id} value={profile.id}>{profile.label} room role</option>)}
      </select>
      <Input value={name} onChange={(e) => { setNameTouched(true); setName(e.target.value) }} placeholder="Name (e.g. architect)" className="mb-2 h-8" />
      <Input value={role} onChange={(e) => setRole(e.target.value)} placeholder="Role (e.g. Architect)" className="mb-2 h-8" />
      <div className="mb-2 grid grid-cols-2 gap-2">
        <select value={lane} onChange={(e) => setLane(e.target.value as RoomSeatLane)} className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent">
          <option value="planning">Planning</option>
          <option value="development">Development</option>
          <option value="review">Review</option>
          <option value="research">Research</option>
          <option value="operations">Operations</option>
        </select>
        <select value={fanOutBudget} onChange={(e) => setFanOutBudget(e.target.value as FanOutBudget)} className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent">
          {FAN_OUT_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
        </select>
      </div>
      {noAllowedProviders ? (
        <div className="mb-2 rounded-lg border border-warn/30 bg-warn/10 px-2.5 py-2 text-[13px] leading-snug text-warn">
          This project is local-only, but no local providers are configured.
        </div>
      ) : (
        <div className="mb-2">
          <select value={currentModelChoice} onChange={(e) => pickModel(e.target.value)} className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent">
            {project?.localOnly && <option value="" disabled>Local models only</option>}
            {pickerProviders.flatMap((p) => providerModelGroups(p).map((group) => (
              <optgroup key={group.id} label={group.detail ? `${group.label} · ${group.detail}` : group.label}>
                {group.models.map((m) => (
                  <option key={`${group.id}:${m}`} value={modelChoiceValue(p.id, group.accountId, m)}>{modelDisplayName(p, m)}</option>
                ))}
              </optgroup>
            )))}
          </select>
          <div className="mt-1 truncate text-[12px] text-ink-4">{provider?.name}{selectedAccountLabel ? ` · ${selectedAccountLabel}` : ''}</div>
        </div>
      )}
      <div className="flex gap-2">
        <Button variant="primary" size="sm" className="flex-1 justify-center" onClick={add} disabled={noAllowedProviders || !model}>Add agent</Button>
        <Button variant="ghost" size="sm" onClick={onCancel}>Cancel</Button>
      </div>
    </div>
  )
}
