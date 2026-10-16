import { useEffect, useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import { AttachmentChips } from '../AttachmentChips'
import { appendAttachments, filesToAttachments } from '../../lib/attachments'
import { allowedProviders } from '../../lib/policy'
import { speakText } from '../../lib/voice'
import type { Attachment, Job, Project, Room, RoomCollaborationMode, RoomDeliberationEffort, RoomMemoryEntry, RoomState, Ticket } from '../../lib/types'
import {
  ACTIVE_ROOM_JOB_STATES,
  ACTIVE_ROOM_STATES,
  DEFAULT_ROOM_COLLABORATION_MODE,
  DEFAULT_ROOM_DELIBERATION_EFFORT,
  ROOM_COLLABORATION_OPTIONS,
  ROOM_DELIBERATION_OPTIONS,
  ROOM_STATE_LABEL,
  collaborationOption,
  deliberationBudgetLabel,
  deliberationOption,
  isActiveRunState,
  latestRoomJob,
  presetSeats,
  roomConflictBlockers,
  roomLeftToWork,
  roomModeDefaultEffort,
  roomWorkerBudget,
  roomWorkerBudgetLabel,
} from './helpers'
import { RoomAgentsSection, RoomCanvas, RoomFlowTicker, RoomStatePill, RoomStat, RoomSupervisorPanel, RoomTicketQueue } from './RoomPanels'
import { RoomGitBanner } from './RoomGitBanner'
import { RoomTimeline, RoomHandoffStrip } from './RoomTimeline'
import { RoomFinalGatePanel, RoomHandoffQueuePanel, RoomLivePlanPanel, RoomModePresets } from './RoomPlans'

export function RoomView({ id }: { id: string }) {
  const room = useStore((s) => s.rooms.find((r) => r.id === id))
  const project = useStore((s) => s.projects.find((p) => p.id === room?.projectId))
  const settings = useStore((s) => s.settings)!
  const status = useStore((s) => s.roomStatus[id])
  const live = useStore((s) => s.roomLive[id])
  const tickets = useStore((s) => s.tickets)
  const jobs = useStore((s) => s.jobs)
  const setRoomSeats = useStore((s) => s.setRoomSeats)
  const updateRoom = useStore((s) => s.updateRoom)
  const steerRoom = useStore((s) => s.steerRoom)
  const mutateRoomMemory = useStore((s) => s.mutateRoomMemory)
  const startRoom = useStore((s) => s.startRoom)
  const cancelRoom = useStore((s) => s.cancelRoom)
  const openProject = useStore((s) => s.openProject)
  const archiveItem = useStore((s) => s.archiveItem)
  const rightPanelHidden = useStore((s) => s.rightPanelHidden)

  const [prompt, setPrompt] = useState('')
  const [roomAttachments, setRoomAttachments] = useState<Attachment[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const addFiles = async (files: FileList | File[]) => {
    const next = await filesToAttachments(files)
    if (next.length) setRoomAttachments((prev) => appendAttachments(prev, next))
  }
  const [starting, setStarting] = useState(false)
  const [adding, setAdding] = useState(false)
  const [memoryDraft, setMemoryDraft] = useState('')
  const [memoryKind, setMemoryKind] = useState<RoomMemoryEntry['kind']>('decision')
  const endRef = useRef<HTMLDivElement>(null)
  const lastAutoReadRef = useRef('')
  const roomState: RoomState = status ?? room?.state ?? 'idle'

  const events = room?.events ?? []
  const sig = events.map((e) => e.id).join('|') + (live ? live.seat + live.text.length : '')
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }) }, [sig])
  useEffect(() => {
    if (!room || !settings?.voice.autoReadRoomUpdates || !settings.voice.enabled || !settings.voice.ttsEnabled) return
    const latest = [...(room.events ?? [])].reverse().find((event) => ['final', 'error', 'ask'].includes(event.kind))
    if (!latest || latest.id === lastAutoReadRef.current) return
    lastAutoReadRef.current = latest.id
    speakText(`${room.name}. ${latest.kind === 'ask' ? 'Needs attention.' : latest.kind}. ${latest.text}`, settings.voice, { summarize: true })
  }, [room?.events?.length, settings?.voice.autoReadRoomUpdates]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!room) return null
  const roomTasks = tickets.filter((t) => t.roomId === room.id)
  const leftToWork = roomLeftToWork(roomTasks)
  const activeTicketWorkers = roomTasks.filter((t) => isActiveRunState(t.runState))
  const conflictBlockers = roomConflictBlockers(roomTasks)
  const roomJobs = jobs.filter((job) => job.roomId === room.id)
  const roomRunJob = latestRoomJob(roomJobs.filter((job) => job.kind === 'room'))
  const running = ACTIVE_ROOM_STATES.has(roomState)
  const ticketJobsByTicket = (() => {
    const byTicket = new Map<string, Job>()
    for (const job of roomJobs.filter((candidate) => candidate.kind === 'ticket' && candidate.ticketId)) {
      const previous = byTicket.get(job.ticketId!)
      const nextRank = Number(ACTIVE_ROOM_JOB_STATES.has(job.status))
      const previousRank = previous ? Number(ACTIVE_ROOM_JOB_STATES.has(previous.status)) : -1
      if (!previous || nextRank > previousRank || (nextRank === previousRank && job.updatedAt > previous.updatedAt)) {
        byTicket.set(job.ticketId!, job)
      }
    }
    return byTicket
  })()
  const filesTouched = [...new Set(events.filter((e) => e.path).map((e) => e.path!))]
  const pendingQuestions = events.filter((e) => e.kind === 'ask' && e.target && e.target !== 'all')
  const collaboration = collaborationOption(room.collaborationMode)
  const deliberation = deliberationOption(room.deliberationEffort)
  const roomMemory = room.memory ?? []
  const roomProviders = allowedProviders(settings.providers, project)
  const workerBudget = roomWorkerBudget(room)
  const patchWorkerBudget = (patch: Partial<NonNullable<Room['workerBudget']>>) => {
    updateRoom(id, { workerBudget: { ...workerBudget, ...patch } })
  }

  const submit = async () => {
    const p = prompt.trim()
    if (!p || starting || room.seats.length === 0) return
    if (running) {
      await steerRoom(id, p)
      setPrompt('')
      setRoomAttachments([])
      return
    }
    setStarting(true)
    try {
      await startRoom(id, p, roomAttachments.length ? roomAttachments : undefined)
      setPrompt('')
      setRoomAttachments([])
    } finally {
      setStarting(false)
    }
  }
  const wakeRoom = async () => {
    const p = (prompt.trim() || room.problem || '').trim()
    if (!p || running || starting || room.seats.length === 0) return
    setStarting(true)
    try {
      await startRoom(id, p, roomAttachments.length ? roomAttachments : undefined)
      if (prompt.trim()) setPrompt('')
      setRoomAttachments([])
    } finally {
      setStarting(false)
    }
  }
  const applyPreset = () => {
    if (running) return
    const seats = presetSeats(settings.providers, project)
    if (seats.length) {
      updateRoom(id, { collaborationMode: DEFAULT_ROOM_COLLABORATION_MODE, deliberationEffort: room.deliberationEffort ?? DEFAULT_ROOM_DELIBERATION_EFFORT, seats })
    }
  }
  const addMemory = () => {
    if (running) return
    const text = memoryDraft.trim()
    if (!text) return
    void mutateRoomMemory(id, { action: 'add', kind: memoryKind, text }).then(() => setMemoryDraft(''))
  }
  const removeMemory = (entryId: string) => {
    if (!running) void mutateRoomMemory(id, { action: 'remove', entryId })
  }
  const speakRoomSummary = () => {
    const latest = [...events].reverse().find((event) => event.text.trim())
    const summary = [
      `${room.name}. State: ${ROOM_STATE_LABEL[roomState]}.`,
      `${room.seats.length} seat${room.seats.length === 1 ? '' : 's'}. ${leftToWork.length} ticket${leftToWork.length === 1 ? '' : 's'} left to work out of ${roomTasks.length}.`,
      activeTicketWorkers.length ? `${activeTicketWorkers.length} active ticket worker${activeTicketWorkers.length === 1 ? '' : 's'}.` : 'No active ticket workers.',
      conflictBlockers.length ? `${conflictBlockers.length} conflict blocker${conflictBlockers.length === 1 ? '' : 's'} visible.` : 'No conflict blockers.',
      pendingQuestions.length ? `${pendingQuestions.length} pending question${pendingQuestions.length === 1 ? '' : 's'} need attention.` : 'No pending questions.',
      filesTouched.length ? `${filesTouched.length} file${filesTouched.length === 1 ? '' : 's'} touched.` : 'No files touched yet.',
      room.finalText ? `Final: ${room.finalText}` : latest ? `Latest update from ${latest.seat}: ${latest.text}` : 'No room transcript yet.',
    ].join(' ')
    speakText(summary, settings.voice, { summarize: true })
  }

  return (
    <div className="flex h-full">
      {/* main: transcript */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b border-line px-5 py-2.5">
          <button onClick={() => room && openProject(room.projectId)} className="flex items-center gap-1.5 text-[13px] text-ink-3 transition-colors hover:text-ink"><Icon name="back" size={14} />{project?.name ?? 'Project'}</button>
          <span className="text-ink-4">/</span>
          <span className="text-[13px] font-medium text-ink">{room.name}</span>
          <RoomStatePill state={roomState} />
          <button onClick={speakRoomSummary} className="ml-auto flex h-7 items-center gap-1.5 rounded-lg border border-line px-2.5 text-[13px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label="Read room summary"><Icon name="speaker" size={14} />Tell me</button>
          <button onClick={() => archiveItem('room', id)} className="flex h-7 items-center gap-1.5 rounded-lg border border-line px-2.5 text-[13px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label="Archive room"><Icon name="archive" size={14} />Archive</button>
        </div>

        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 py-5">
          <div className="mx-auto w-full max-w-[920px] space-y-4">
            <RoomGitBanner
              project={project}
              running={running || starting}
              onReady={() => undefined}
            />
            <RoomSupervisorPanel
              room={room}
              roomState={roomState}
              roomJob={roomRunJob}
              leftToWork={leftToWork}
              totalTickets={roomTasks.length}
              activeWorkers={activeTicketWorkers}
              ticketJobsByTicket={ticketJobsByTicket}
              conflictBlockers={conflictBlockers}
              budgetLabel={roomWorkerBudgetLabel(room, activeTicketWorkers.length)}
              running={running}
              starting={starting}
              canWake={!!(prompt.trim() || room.problem) && room.seats.length > 0}
              onWake={() => void wakeRoom()}
              onStop={() => cancelRoom(id)}
            />
            <RoomFlowTicker room={room} events={events} live={live} running={running} />
            <RoomLivePlanPanel
              room={room}
              events={events}
              leftToWork={leftToWork}
              activeWorkers={activeTicketWorkers}
              ticketJobsByTicket={ticketJobsByTicket}
              conflictBlockers={conflictBlockers}
              totalTickets={roomTasks.length}
            />
            <RoomHandoffStrip events={events} live={live} />
            <RoomFinalGatePanel tickets={roomTasks} leftToWork={leftToWork} conflictBlockers={conflictBlockers} room={room} />
            {events.length === 0 && !live && !running && (
              <RoomCanvas
                room={room}
                providers={roomProviders}
                project={project}
                onSetSeats={(seats) => void setRoomSeats(id, seats)}
                onSetCollaboration={(mode) => void updateRoom(id, { collaborationMode: mode })}
                onSetMode={(mode) => void updateRoom(id, { mode, deliberationEffort: roomModeDefaultEffort(mode) })}
              />
            )}
            <RoomTimeline events={events} live={live} />
            <div ref={endRef} className="h-px" />
          </div>
        </div>

        {/* composer */}
        <div className="shrink-0 px-5 pb-5">
          <div className="mx-auto w-full max-w-[920px]">
            {roomAttachments.length > 0 && (
              <AttachmentChips items={roomAttachments} onRemove={(aid) => setRoomAttachments((prev) => prev.filter((a) => a.id !== aid))} className="mb-2 px-1" />
            )}
            <input ref={fileInputRef} type="file" multiple className="hidden" onChange={(e) => { if (e.target.files) void addFiles(e.target.files); e.target.value = '' }} />
            <form onSubmit={(e) => { e.preventDefault(); void submit() }} className="flex items-end gap-2 rounded-[18px] border border-line bg-card p-2 shadow-card focus-within:border-line-strong">
              <button type="button" onClick={() => fileInputRef.current?.click()} disabled={room.seats.length === 0} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-40" title="Attach documents or images" aria-label="Attach"><Icon name="attach" size={17} /></button>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void submit() } }}
                onPaste={(e) => { const files = Array.from(e.clipboardData.files); if (files.length) { e.preventDefault(); void addFiles(files) } }}
                rows={1}
                placeholder={room.seats.length === 0 ? 'Add at least one agent first…' : running ? 'Steer the running room… try @reviewer, @builder, or @planner' : 'Pose a problem for the room…'}
                className="scroll-thin max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[14px] text-ink outline-none placeholder:text-ink-3"
              />
              {running ? (
                <div className="flex shrink-0 items-center gap-1">
                  <button
                    type="button"
                    onMouseDown={(e) => { e.preventDefault(); void submit() }}
                    onClick={() => void submit()}
                    disabled={!prompt.trim()}
                    className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-card-2 text-ink-2 transition-all hover:border-line-strong hover:text-ink disabled:opacity-40"
                    aria-label="Steer room"
                    title="Send steering to the running room"
                  >
                    <Icon name="send" size={15} strokeWidth={2} />
                  </button>
                  <button type="button" onClick={() => cancelRoom(id)} className="flex h-9 w-9 items-center justify-center rounded-full bg-accent text-accent-ink transition-all hover:bg-accent-strong active:scale-95" aria-label="Stop"><Icon name="stop" size={16} /></button>
                </div>
              ) : (
                <button
                  type="button"
                  onMouseDown={(e) => { e.preventDefault(); void submit() }}
                  onClick={() => void submit()}
                  disabled={!prompt.trim() || room.seats.length === 0 || starting}
                  className="flex h-9 w-9 items-center justify-center rounded-full bg-accent text-accent-ink transition-all hover:bg-accent-strong disabled:bg-ink-4 disabled:text-bg"
                  aria-label="Start"
                >
                  <Icon name="send" size={16} strokeWidth={2} />
                </button>
              )}
            </form>
          </div>
        </div>
      </div>

      {/* right: seats */}
      {!rightPanelHidden && <aside className="scroll-thin flex w-[320px] shrink-0 flex-col overflow-y-auto border-l border-line bg-bg xl:w-[360px]">
        <div className="border-b border-line px-3 py-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Room health</span>
            <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-3">{ROOM_STATE_LABEL[roomState]}</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            <RoomStat icon="cpu" label="Seats" value={String(room.seats.length)} />
            <RoomStat icon="list" label="Tasks" value={`${leftToWork.length}/${roomTasks.length}`} accent={leftToWork.length > 0} />
            <RoomStat icon="file" label="Files" value={String(filesTouched.length)} />
            <RoomStat icon="message" label="Asks" value={String(pendingQuestions.length)} accent={pendingQuestions.length > 0} />
          </div>
          <RoomTicketQueue tickets={leftToWork} total={roomTasks.length} blockers={conflictBlockers} ticketJobsByTicket={ticketJobsByTicket} />
        </div>

        <RoomHandoffQueuePanel events={events} />

        <RoomAgentsSection
          room={room}
          providers={roomProviders}
          tickets={tickets}
          running={running}
          adding={adding}
          setAdding={setAdding}
          setSeats={(seats) => void setRoomSeats(id, seats)}
        />

        <div className="border-b border-line px-3 py-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Collaboration</span>
            <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-3">{roomMemory.length} memories</span>
          </div>
          <select
            value={room.collaborationMode ?? DEFAULT_ROOM_COLLABORATION_MODE}
            disabled={running}
            onChange={(e) => updateRoom(id, { collaborationMode: e.target.value as RoomCollaborationMode })}
            className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2 text-[13px] text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-60"
          >
            {ROOM_COLLABORATION_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
          </select>
          <p className="mt-1.5 text-[12px] leading-snug text-ink-3">{collaboration.detail}</p>
          <RoomModePresets
            room={room}
            running={running}
            onApply={(patch) => {
              const seats = room.seats.length ? room.seats : presetSeats(settings.providers, project)
              void updateRoom(id, { ...patch, seats })
            }}
          />
          <div className="mt-3 flex items-center justify-between gap-2">
            <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Deliberation</span>
            <span className="text-[12px] text-ink-4">{deliberationBudgetLabel(room.deliberationEffort)}</span>
          </div>
          <select
            value={room.deliberationEffort ?? DEFAULT_ROOM_DELIBERATION_EFFORT}
            disabled={running}
            onChange={(e) => updateRoom(id, { deliberationEffort: e.target.value as RoomDeliberationEffort })}
            className="no-drag mt-1.5 h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2 text-[13px] text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-60"
          >
            {ROOM_DELIBERATION_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
          </select>
          <p className="mt-1.5 text-[12px] leading-snug text-ink-3">{deliberation.detail} Separate from model reasoning effort.</p>
          <div className="mt-3 flex items-center justify-between gap-2">
            <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Worker budget</span>
            <span className="text-[12px] text-ink-4">{roomWorkerBudgetLabel(room, activeTicketWorkers.length)}</span>
          </div>
          <div className="mt-1.5 grid grid-cols-2 gap-1.5">
            <label className="min-w-0">
              <span className="mb-1 block text-[12px] text-ink-4">Concurrent</span>
              <select
                value={workerBudget.maxActive}
                disabled={running}
                onChange={(e) => patchWorkerBudget({ maxActive: Number(e.target.value), maxStarts: Math.max(Number(e.target.value), workerBudget.maxStarts) })}
                className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2 text-[13px] text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-60"
              >
                {[1, 2, 3, 4, 5, 6, 7, 8].map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </label>
            <label className="min-w-0">
              <span className="mb-1 block text-[12px] text-ink-4">Starts/run</span>
              <select
                value={workerBudget.maxStarts}
                disabled={running}
                onChange={(e) => patchWorkerBudget({ maxStarts: Number(e.target.value) })}
                className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2 text-[13px] text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-60"
              >
                {[1, 2, 3, 4, 6, 8, 12, 16, 24, 32].filter((value) => value >= workerBudget.maxActive).map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </label>
          </div>
          <Button variant="default" size="sm" icon="spark" className="mt-2 w-full justify-center" disabled={running} onClick={applyPreset}>
            Planner + builder + reviewer
          </Button>
        </div>

        <div className="border-b border-line px-3 py-3">
          <div className="mb-2 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Room memory</div>
          {room.memorySummary ? <p className="mb-2 line-clamp-3 rounded-lg border border-line bg-card px-2 py-1.5 text-[12px] leading-snug text-ink-3">{room.memorySummary}</p> : null}
          <div className="space-y-1.5">
            {roomMemory.slice(0, 3).map((entry) => (
              <div key={entry.id} className="group rounded-lg border border-line bg-card px-2 py-1.5">
                <div className="flex items-start gap-1.5">
                  <span className="mt-0.5 rounded bg-ink-5 px-1 text-[11px] font-medium uppercase tracking-wide text-ink-3">{entry.kind}</span>
                  <div className="min-w-0 flex-1 line-clamp-2 text-[12px] leading-snug text-ink-2">{entry.text}</div>
                  {entry.source === 'manual' && (
                    <button onClick={() => removeMemory(entry.id)} disabled={running} className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-ink-4 opacity-0 transition-opacity hover:bg-danger/10 hover:text-danger group-hover:opacity-100 disabled:pointer-events-none" aria-label="Remove memory"><Icon name="x" size={12} /></button>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-2 flex gap-1.5">
            <select value={memoryKind} disabled={running} onChange={(e) => setMemoryKind(e.target.value as RoomMemoryEntry['kind'])} className="no-drag h-7 w-[92px] cursor-pointer rounded-lg border border-line bg-card px-1.5 text-[12px] text-ink outline-none focus:border-accent disabled:opacity-45">
              <option value="decision">Decision</option>
              <option value="preference">Preference</option>
              <option value="finding">Finding</option>
              <option value="summary">Summary</option>
            </select>
            <button onClick={addMemory} disabled={running || !memoryDraft.trim()} className="flex h-7 w-7 items-center justify-center rounded-lg border border-line bg-card text-ink-2 transition-colors hover:border-line-strong disabled:opacity-45" aria-label="Add memory"><Icon name="plus" size={14} /></button>
          </div>
          <textarea
            value={memoryDraft}
            disabled={running}
            onChange={(e) => setMemoryDraft(e.target.value)}
            rows={2}
            placeholder="Add a durable room note..."
            className="scroll-thin mt-1.5 w-full resize-none rounded-lg border border-line bg-card px-2 py-1.5 text-[13px] leading-snug text-ink outline-none placeholder:text-ink-4 focus:border-accent disabled:opacity-45"
          />
        </div>

      </aside>}
    </div>
  )
}
