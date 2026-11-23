import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import clsx from 'clsx'
import { basename } from '../../lib/format'
import { Icon } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import { Button } from '../ui'
import { allowedProviders } from '../../lib/policy'
import { modelDisplayName } from '../../lib/provider-accounts'
import type { AgentRoleProfileId, Job, Project, Provider, Room, RoomCollaborationMode, RoomEventItem, RoomMode, RoomState, Seat, Ticket } from '../../lib/types'
import { ACTIVE_ROOM_STATES, ROLE_PROFILES, ROOM_COLLABORATION_OPTIONS, ROOM_MODE_OPTIONS, ROOM_STATE_LABEL, Avatar, presetSeats, seatForRole, isActiveRunState, roomJobDetail, roomJobTitle, roomSupervisorPhase, ticketFileRefsForDisplay, ticketPhaseLabel, ticketStatusLabel, ticketWorkerLabel } from './helpers'
import type { RoomConflictBlocker } from './helpers'

export { RoomAgentsSection } from './RoomAgentsSection'

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
  const tx = useAuthoredText()
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
                <div className="text-[13px] font-semibold text-ink">{tx("Room supervisor")}</div>
                <span className={clsx('rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium', stateTone)}>{ROOM_STATE_LABEL[roomState]}</span>
              </div>
              <div className="mt-0.5 truncate text-[13px] text-ink-3">{roomJobTitle(roomJob)}</div>
              {roomJobDetail(roomJob) && <div className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-ink-4">{roomJobDetail(roomJob)}</div>}
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              {!running && canWake && (
                <Button size="xs" variant="default" icon="play" onClick={onWake} disabled={starting || room.seats.length === 0}>
                  {starting ? tx("Waking...") : tx("Wake")}
                </Button>
              )}
              {running && (
                <Button size="xs" variant="danger" icon="stop" onClick={onStop}>
                  {tx("Stop")}</Button>
              )}
            </div>
          </div>

          <div className="mt-3 grid grid-cols-4 gap-2 text-[12px]">
            <SupervisorMetric label={tx("Phase")} value={roomSupervisorPhase(leftToWork, activeWorkers, ticketJobsByTicket, totalTickets)} accent={running} />
            <SupervisorMetric label={tx("Left")} value={`${leftToWork.length}/${totalTickets}`} accent={leftToWork.length > 0} />
            <SupervisorMetric label={tx("Workers")} value={String(activeWorkers.length)} accent={activeWorkers.length > 0} />
            <SupervisorMetric label={tx("Budget")} value={budgetLabel} />
          </div>

          {activeWorkers.length > 0 && (
            <div className="mt-3 border-t border-line pt-2">
              <div className="mb-1 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("Active workers")}</div>
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
            <div className="mb-1 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("Conflict blockers")}</div>
            {conflictBlockers.length > 0 ? (
              <div className="space-y-1">
                {conflictBlockers.slice(0, 3).map((blocker) => (
                  <div key={`${blocker.ticketId}:${blocker.text}`} className="flex min-w-0 gap-1.5 text-[13px] leading-snug text-warn">
                    <Icon name="warning" size={11} className="mt-0.5 shrink-0" />
                    <span className="min-w-0"><span className="font-medium">{blocker.ticketTitle}</span>: {blocker.text}</span>
                  </div>
                ))}
                {conflictBlockers.length > 3 && <div className="text-[12px] text-ink-4">+{conflictBlockers.length - 3} {tx("more")}</div>}
              </div>
            ) : (
              <div className="text-[13px] text-ink-4">{tx("No conflict blockers.")}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function SupervisorMetric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  const tx = useAuthoredText()
  return (
    <div className="rounded-lg bg-card-2 px-2 py-1.5">
      <div className="text-[11px] uppercase tracking-[0.06em] text-ink-4">{label}</div>
      <div className={clsx('mt-0.5 truncate text-[13px] font-semibold tabular-nums', accent ? 'text-accent' : 'text-ink')} title={value}>{value}</div>
    </div>
  )
}

export function RoomTicketQueue({ tickets, total, blockers, ticketJobsByTicket }: { tickets: Ticket[]; total: number; blockers: RoomConflictBlocker[]; ticketJobsByTicket: Map<string, Job> }) {
  const tx = useAuthoredText()
  const blockersByTicket = new Map<string, string[]>()
  for (const blocker of blockers) blockersByTicket.set(blocker.ticketId, [...(blockersByTicket.get(blocker.ticketId) ?? []), blocker.text])
  if (total === 0) return <div className="mt-3 rounded-lg border border-dashed border-line px-2.5 py-4 text-center text-[13px] text-ink-3">{tx("No room tickets yet.")}</div>
  if (tickets.length === 0) return <div className="mt-3 rounded-lg border border-line bg-card px-2.5 py-3 text-center text-[13px] text-positive">{tx("All room tickets complete.")}</div>
  return (
    <div className="mt-3">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("Left to work")}</span>
        <span className="rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-4">{tickets.length}</span>
      </div>
      <div className="space-y-1.5">
        {tickets.slice(0, 6).map((ticket) => <RoomTicketQueueItem key={ticket.id} ticket={ticket} job={ticketJobsByTicket.get(ticket.id)} blockers={blockersByTicket.get(ticket.id) ?? []} />)}
      </div>
      {tickets.length > 6 && <div className="mt-1.5 text-[12px] text-ink-4">+{tx("More tickets: {count}", { count: tickets.length - 6 })}</div>}
    </div>
  )
}

function RoomTicketQueueItem({ ticket, job, blockers }: { ticket: Ticket; job?: Job; blockers: string[] }) {
  const tx = useAuthoredText()
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
        <span className="capitalize">{tx("risk ")}{ticket.risk ?? tx("low")}</span>
        {caps.length ? <span className="truncate">{tx("caps ")}{caps.slice(0, 2).join(', ')}{caps.length > 2 ? ` +${caps.length - 2}` : ''}</span> : null}
        {required.length ? <span className="ml-auto tabular-nums">{tx("ev ")}{evidence.length}/{required.length}</span> : null}
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
  const tx = useAuthoredText()
  const tick = events.reduce((max, event) => Math.max(max, event.tick ?? 0), 0)
  const latest = [...events].reverse().find((event) => event.seat !== 'you')
  const activeSeat = live?.seat ?? (running ? latest?.seat : null)
  const sleeping = room.state === 'waiting_external' || room.state === 'waiting_user'
  const label = live ? tx("{value0} is speaking", { value0: String(live.seat) }) : running ? tx("Room is flowing") : sleeping ? tx("Room sleeping") : room.state === 'complete' ? tx("Room complete") : tx("Room idle")
  return (
    <div className="rounded-xl border border-line bg-card px-3.5 py-2.5 shadow-card">
      <div className="flex items-center gap-2">
        <span className={clsx('h-2 w-2 rounded-full', running ? 'animate-pulse bg-accent' : sleeping ? 'bg-warn' : room.state === 'complete' ? 'bg-positive' : 'bg-ink-4')} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[13px] font-medium text-ink">
            <span>{label}</span>
            <span className="rounded bg-ink-5 px-1.5 py-0.5 font-mono text-[12px] text-ink-3">{tx("tick ")}{tick}</span>
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
  const tx = useAuthoredText()
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
  const tx = useAuthoredText()
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
  const tx = useAuthoredText()
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
      <h2 className="text-[16px] font-semibold text-ink">{tx("Build your room")}</h2>
      <p className="mt-1 text-[13px] text-ink-3">{tx("Drag a role into the room — or click to add. Each agent picks a fitting model: builders and operators run local, planners and reviewers run on a frontier model.")}</p>

      <div className="mt-4 flex flex-wrap gap-2">
        {ROLE_PROFILES.map((p) => (
          <button
            key={p.id}
            draggable={!noProviders}
            onDragStart={(e) => { e.dataTransfer.setData('text/role', p.id); e.dataTransfer.effectAllowed = 'copy' }}
            onClick={() => addRole(p.id)}
            disabled={noProviders}
            className="flex cursor-grab items-center gap-1.5 rounded-full border border-line bg-card px-3 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink active:cursor-grabbing disabled:opacity-50"
            title={tx("Add a {value0}", { value0: p.label })}
          >
            <Icon name={p.lane === 'development' || p.lane === 'operations' ? 'cpu' : 'globe'} size={13} className="text-ink-3" />
            {p.label}
            <Icon name="plus" size={12} className="text-ink-4" />
          </button>
        ))}
        <button onClick={() => onSetSeats(presetSeats(providers, project))} disabled={noProviders} className="flex items-center gap-1.5 rounded-full border border-dashed border-line px-3 py-1.5 text-[13px] text-ink-3 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-50" title={tx("Add a planner, builder and reviewer")}>
          <Icon name="spark" size={13} />{tx("Recommended trio")}</button>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false) }}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); const role = e.dataTransfer.getData('text/role') as AgentRoleProfileId; if (role) addRole(role) }}
        className={clsx('mt-3 rounded-2xl border-2 border-dashed p-4 transition-colors', dragOver ? 'border-accent bg-accent-soft/40' : 'border-line')}
      >
        {room.seats.length === 0 ? (
          <div className="py-8 text-center text-[13px] text-ink-3">{noProviders ? tx("No models configured — connect one in Settings first.") : tx("Drop agents here to staff the room.")}</div>
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
                  <button onClick={() => removeSeat(seat.id)} className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded text-ink-3 opacity-0 transition-opacity hover:bg-ink-5 hover:text-ink group-hover:opacity-100" aria-label={tx("Remove agent")}><Icon name="x" size={12} /></button>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {room.seats.length > 0 && (
        <>
          <div className="mb-2 mt-6 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("What this room is for")}</div>
          <div className="grid gap-2 sm:grid-cols-3">
            {ROOM_MODE_OPTIONS.map((opt) => (
              <button key={opt.id} onClick={() => onSetMode(opt.id)} className={clsx('rounded-xl border px-3 py-2.5 text-left transition-colors', room.mode === opt.id ? 'border-line-strong bg-card' : 'border-line bg-card-2 hover:border-line-strong')}>
                <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink">{room.mode === opt.id && <Icon name="check" size={13} className="text-accent" />}{opt.label}</div>
                <div className="mt-0.5 text-[12px] leading-snug text-ink-3">{opt.detail}</div>
              </button>
            ))}
          </div>
          <div className="mb-2 mt-6 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("How they work together")}</div>
          <div className="grid gap-2 sm:grid-cols-2">
            {ROOM_COLLABORATION_OPTIONS.map((opt) => (
              <button key={opt.id} onClick={() => onSetCollaboration(opt.id)} className={clsx('rounded-xl border px-3 py-2.5 text-left transition-colors', activeMode === opt.id ? 'border-line-strong bg-card' : 'border-line bg-card-2 hover:border-line-strong')}>
                <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink">{activeMode === opt.id && <Icon name="check" size={13} className="text-accent" />}{opt.label}</div>
                <div className="mt-0.5 text-[12px] leading-snug text-ink-3">{opt.detail}</div>
              </button>
            ))}
          </div>
          <p className="mt-4 text-center text-[13px] text-ink-3">{tx("Pose a problem below to start the huddle.")}</p>
        </>
      )}
    </div>
  )
}
