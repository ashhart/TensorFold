
import type { RoomSupervisorMetadataPatch } from '../../agent/jobs'

import { buildUserContent, contentText } from '../../agent/content'
import { normalizeAttachments } from '../../agent/attachment-policy'
import type { Attachment, Job, Provider, Room, RoomStreamEvent, Seat, Ticket } from '../../agent/types'

import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey } from './workspace-mutation-guard'

import type { WorkspaceRuntimeIpcDeps } from './workspace-runtime-ipc'
import { reserveRoomStartup } from './workspace-startup-policy'
import { createRoomTools, type RoomToolScope } from './room-tools-runtime'
import { captureAgentReasoningEffort } from '../../../shared/reasoning-effort'

export function createRoomRunRuntime(ctx: WorkspaceRuntimeIpcDeps) {
  const {
    path,
    store,
    workspaceMutationGuard,
    requireSecurityRole,
    persistRooms,
    roomAborters,
    ensureRoomWorktree,
    emitRoom,
    patchRoomSupervisorJob,
    finishRoomSupervisorMission,
    finishJob,
    ensureRoomSupervisorMission,
    roomWorkerBudget,
    claimJob,
    patchJob,
    buildRoomSupervisorJobMetadata,
    ticketsForRoom,
    runtimeProvider,
    inferSeatLane,
    appendRoomSupervisorEvent,
    ticketNeedsReviewAfterBuilder,
    roomTicketConflictBlockers,
    parkRoomIfWorkRemains,
    modelRouter,
    modelRouteAutoFailoverEnabled,
    effectiveContextLimit,
    roomContextLimit,
    auditScopedModelRoute,
    runtimeProviderWithFreshModels,
    assertProviderEndpoint,
    modelRouteFailoverText,
    roomRouteHandoff,
    modelRouteResetPolicy,
    roomFinalGateBlockers,
    runRoom,
  } = ctx
async function startRoomRun(roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }> {
  requireSecurityRole('operator', 'start rooms')
  const room = store.rooms.find((r) => r.id === roomId)
  if (!room) throw new Error('Room not found')
  if (room.archivedAt) throw new Error('Restore this archived room before starting it.')
  // Don't start a second run for a room that's already running — it would overwrite the live
  // aborter, making both the first run uncancellable and the second un-stoppable.
  if (roomAborters.has(roomId)) return { ok: false, room }
  const project = store.projects.find((p) => p.id === room.projectId)
  if (!project) throw new Error('Project not found')
  if (project.archivedAt) throw new Error('Restore the archived project before starting this room.')
  // Rooms broadcast a single text problem to every seat (many are CLI/text-only), so fold
  // attachments into text: documents become labelled blocks; images are noted (vision-capable
  // seats can't receive pixels through the room channel).
  const safeAttachments = normalizeAttachments(attachments)
  const cleanProblem = contentText(buildUserContent(String(problem || '').trim(), safeAttachments)).trim()
  if (!cleanProblem) throw new Error('Add a room objective before starting.')
  if (room.seats.length < 1) throw new Error('Add at least one agent to the room first.')
  const reasoningEffort = store.settings.reasoningEffort
  const seatEfforts = new Map(room.seats.map((seat) => [seat.id, captureAgentReasoningEffort(seat, reasoningEffort)]))
  const configuredWorkspace = room.workspace ?? project.workspace ?? null
  if (configuredWorkspace?.kind === 'ssh') {
    throw new Error('Local room execution does not support SSH workspaces; run this room on its remote Orchestrator.')
  }
  const initialPaths = [configuredWorkspace?.path, room.workspaceCwd, room.rootCwd, room.scratchpadCwd, project.cwd]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
  const initialWorkspaceKeys = initialPaths.map((value: string) => lexicalLocalWorkspaceKey(value))
  const reservation = reserveRoomStartup(roomId, roomAborters, workspaceMutationGuard, initialWorkspaceKeys)
  if (!reservation) return { ok: false, room }
  const { controller, runLease } = reservation
  const timeout = setTimeout(() => controller.abort(), 30 * 60_000)
  roomAborters.set(roomId, controller)
  let startupComplete = false
  try {
  const roomWorkspace = await ensureRoomWorktree(project, room)
  const finalWorkspaceKey = await canonicalLocalWorkspaceKey(roomWorkspace)
  runLease.addWorkspaceKeys?.([finalWorkspaceKey])
  if (room.archivedAt || project.archivedAt) throw new Error('Room or project was archived while the room was starting.')
  if (room.workspaceCwd !== roomWorkspace) {
    room.workspaceCwd = roomWorkspace
    persistRooms()
  }
  const roomProject = { ...project, cwd: roomWorkspace }

  room.problem = cleanProblem
  room.events = []
  room.finalText = ''
  room.state = 'queued'
  room.updatedAt = Date.now()
  persistRooms()
  emitRoom({ roomId, type: 'status', state: 'queued' })
  const supervisorMission = ensureRoomSupervisorMission(room, project, roomWorkspace, cleanProblem)
  const supervisorRunId = `room:${room.id}:${Date.now()}`
  const configuredWorkerBudget = roomWorkerBudget(room)
  let job: Job | null = claimJob({
    kind: 'room',
    owner: room.id,
    projectId: room.projectId,
    missionId: supervisorMission.id,
    roomId: room.id,
    currentThreadId: null,
    metadata: buildRoomSupervisorJobMetadata({
      room,
      project,
      tickets: ticketsForRoom(room),
      patch: { missionId: supervisorMission.id, workerBudget: configuredWorkerBudget, workerStarts: 0 },
    }),
  })
  job = patchJob(job, { currentRunId: supervisorRunId })

  let lastCodeChangeAt = 0
  let greenEvidenceAt = 0
  const providerForSeat = (seat: Seat): Provider | undefined => {
    const provider = store.settings.providers.find((p) => p.id === seat.providerId)
    return provider ? runtimeProvider(provider, seat.providerAccountId) : undefined
  }
  if (roomProject.localOnly) {
    const cloudSeat = room.seats.find((seat) => providerForSeat(seat)?.kind !== 'local')
    if (cloudSeat) {
      room.state = 'failed'
      persistRooms()
      emitRoom({ roomId: room.id, type: 'error', message: `Project is local-only, but seat "${cloudSeat.name}" is not using a local provider.`, fatal: true })
      emitRoom({ roomId: room.id, type: 'status', state: 'failed' })
      emitRoom({ roomId: room.id, type: 'done' })
      clearTimeout(timeout)
      roomAborters.delete(room.id)
      job = patchRoomSupervisorJob(job, room, project)
      finishRoomSupervisorMission(job, 'failed', `Project is local-only, but seat "${cloudSeat.name}" is not using a local provider.`)
      finishJob(job, 'failed', `Project is local-only, but seat "${cloudSeat.name}" is not using a local provider.`)
      return { ok: false }
    }
  }
  const roomTickets = (): Ticket[] => ticketsForRoom(room)
  // Resolve a ticket from free-form LLM tool input. Precedence: exact id, then exact (trimmed,
  // case-insensitive) title equality, then a UNIQUE case-insensitive substring. A substring that
  // matches multiple tickets is AMBIGUOUS — never silently pick the first (store-insertion-order)
  // match, or claim/complete/start could act on the wrong ticket (e.g. "auth" hitting both
  // "Add auth" and "Add auth tests"). On ambiguity, `error` is set and `ticket` is undefined.
  const resolveTicket = (args: any): { ticket?: Ticket; error?: string } => {
    const board = roomTickets()
    const id = typeof args?.id === 'string' ? args.id : ''
    if (id) {
      const byId = board.find((t) => t.id === id)
      if (byId) return { ticket: byId }
    }
    const rawTitle = typeof args?.title === 'string' ? args.title.trim() : ''
    if (!rawTitle) return { ticket: undefined }
    const wanted = rawTitle.toLowerCase()
    const exact = board.filter((t) => t.title.trim().toLowerCase() === wanted)
    if (exact.length === 1) return { ticket: exact[0] }
    if (exact.length > 1) {
      return { error: `"${rawTitle}" matches ${exact.length} tickets with that exact title; reference the ticket by id instead.` }
    }
    const substr = board.filter((t) => t.title.toLowerCase().includes(wanted))
    if (substr.length === 1) return { ticket: substr[0] }
    if (substr.length > 1) {
      return { error: `"${rawTitle}" is ambiguous — it matches ${substr.length} tickets (${substr.map((t) => `"${t.title}"`).join(', ')}); use a more specific title or the ticket id.` }
    }
    return { ticket: undefined }
  }
  const findTicket = (args: any): Ticket | undefined => resolveTicket(args).ticket
  // Mirrors tickets.ts isReservationEvidence / meaningfulTicketEvidence (which are module-private
  // there): a "Reserved for build/review worker." note placed by start_ticket_work is a placeholder,
  // not real evidence, so it must not satisfy a ticket's required-evidence gate.
  const isReservationEvidenceLocal = (item: { kind?: string; summary?: string; ref?: string | null }): boolean =>
    /\breserved for (?:build|review) worker\b/i.test(`${item.kind ?? ''} ${item.summary ?? ''} ${item.ref ?? ''}`)
  const hasMeaningfulEvidence = (tk: Ticket): boolean =>
    (tk.evidence ?? []).some((item) => !isReservationEvidenceLocal(item))
  const workerBudget = configuredWorkerBudget
  let workerStarts = 0
  const activeTicketWorkerCount = () => roomTickets().filter((ticket) => ticket.runState === 'queued' || ticket.runState === 'running').length
  const touchSupervisor = (patch: RoomSupervisorMetadataPatch = {}) => {
    job = patchRoomSupervisorJob(job, room, project, { workerBudget, workerStarts, ...patch })
  }
  const selectAutomaticReviewSeat = (builderSeat: string): Seat | null => {
    const candidates = room.seats
      .map((seat, index) => {
        if (seat.name === builderSeat) return null
        if (!providerForSeat(seat)) return null
        const reviewRank = inferSeatLane(seat) === 'review' || seat.roleProfileId === 'reviewer' ? 0 : 1
        return { seat, index, reviewRank }
      })
      .filter((item): item is { seat: Seat; index: number; reviewRank: number } => !!item)
      .sort((a, b) => a.reviewRank - b.reviewRank || a.index - b.index)
    return candidates[0]?.seat ?? null
  }
  const blockAutomaticReview = (tk: Ticket, builderSeat: string, paths: string[], summary: string) => {
    touchSupervisor({ phase: { phase: 'review', status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat: builderSeat, paths, summary } })
    appendRoomSupervisorEvent(room, `Automatic review not started for "${tk.title}": ${summary}`, {
      tool: 'auto_review_scheduler',
      args: { ticketId: tk.id, phase: 'review', builderSeat, paths },
    })
  }
  const scheduleAutomaticReviewAfterBuild = async (tk: Ticket, builderSeat: string, paths: string[]) => {
    if (!ticketNeedsReviewAfterBuilder(tk)) return
    const reviewSeat = selectAutomaticReviewSeat(builderSeat)
    if (!reviewSeat) {
      blockAutomaticReview(tk, builderSeat, paths, 'No suitable different room seat is available for review.')
      return
    }
    const blockers: string[] = roomTicketConflictBlockers({ ticket: tk, tickets: roomTickets(), phase: 'review', seat: reviewSeat.name, paths })
    if (blockers.length) {
      blockAutomaticReview(tk, reviewSeat.name, paths, blockers.join(' '))
      return
    }
    const activeWorkers = activeTicketWorkerCount()
    if (activeWorkers >= workerBudget.maxActive) {
      blockAutomaticReview(tk, reviewSeat.name, paths, `Worker budget busy: ${activeWorkers}/${workerBudget.maxActive} active.`)
      return
    }
    if (workerStarts >= workerBudget.maxStarts) {
      blockAutomaticReview(tk, reviewSeat.name, paths, `Worker budget exhausted: ${workerStarts}/${workerBudget.maxStarts} started.`)
      return
    }
    appendRoomSupervisorEvent(room, `Automatic review queued for "${tk.title}" with ${reviewSeat.name}.`, {
      tool: 'auto_review_scheduler',
      args: { ticketId: tk.id, phase: 'review', builderSeat, reviewSeat: reviewSeat.name, paths },
    })
    try {
      await roomTools.start_ticket_work({ id: tk.id, phase: 'review', paths }, reviewSeat.name)
    } catch (err: any) {
      blockAutomaticReview(tk, reviewSeat.name, paths, String(err?.message ?? err))
    }
  }
  const emitRoomProgress = (ev: RoomStreamEvent) => {
    if (ev.type === 'status') {
      room.state = ev.state
      room.updatedAt = Date.now()
      persistRooms()
      touchSupervisor()
    } else if (ev.type === 'turn') {
      touchSupervisor({ currentTurnSeat: ev.seat })
    } else if (ev.type === 'event') {
      room.updatedAt = Date.now()
      persistRooms()
      touchSupervisor({ lastEventKind: ev.item.kind, lastEventSeat: ev.item.seat, lastEventAt: ev.item.createdAt })
    } else if (ev.type === 'memory') {
      room.updatedAt = Date.now()
      persistRooms()
      touchSupervisor()
    } else if (ev.type === 'final' || ev.type === 'done' || ev.type === 'error') {
      touchSupervisor()
    }
    emitRoom(ev)
  }
  touchSupervisor()
  const roomTools: Record<string, (args: any, seat: string) => Promise<string>> = createRoomTools(ctx, {
    reasoningEffort,
    seatEfforts,
    roomTickets,
    room,
    resolveTicket,
    findTicket,
    hasMeaningfulEvidence,
    touchSupervisor,
    activeTicketWorkerCount,
    workerBudget,
    get workerStarts() { return workerStarts },
    set workerStarts(value: RoomToolScope['workerStarts']) { workerStarts = value },
    get lastCodeChangeAt() { return lastCodeChangeAt },
    set lastCodeChangeAt(value: RoomToolScope['lastCodeChangeAt']) { lastCodeChangeAt = value },
    get greenEvidenceAt() { return greenEvidenceAt },
    set greenEvidenceAt(value: RoomToolScope['greenEvidenceAt']) { greenEvidenceAt = value },
    scheduleAutomaticReviewAfterBuild,
    roomProject,
    roomWorkspace,
    controller,
  })

  // Supervisor mode: an ASSIGN auto-starts the addressed seat's ticket worker.
  // The ticket is resolved from the ASSIGN text (a verbatim ticket title inside the
  // text wins; otherwise the text minus the seat name is used as a title query), then
  // routed through the standard start_ticket_work path so budget/conflict gates apply.
  const autoStartAssignedTicket = async ({ target, text }: { seat: string; target: string; text: string }): Promise<string | null> => {
    const board = roomTickets().filter((t) => t.status !== 'completed')
    const hay = text.toLowerCase()
    const inText = board.filter((t) => t.title.trim() && hay.includes(t.title.trim().toLowerCase()))
    let tk = inText.length === 1 ? inText[0] : undefined
    if (!tk && inText.length > 1) {
      const matches = inText.slice(0, 4).map((t) => `"${t.title}"`).join(', ')
      return `ASSIGN auto-start skipped: the text matches ${inText.length} open tickets (${matches}). Name one verbatim to start it.`
    }
    if (!tk && inText.length === 0) {
      const remainder = text.replace(new RegExp(`\\b${target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'ig'), '').trim()
      const resolved = resolveTicket({ title: remainder })
      if (resolved.error) return `ASSIGN auto-start skipped: ${resolved.error}`
      tk = resolved.ticket
    }
    if (!tk) return `ASSIGN to ${target} noted, but no open ticket title matched. Create the ticket with create_ticket first, then assign it.`
    if (tk.assignee && tk.assignee !== target) return `ASSIGN auto-start skipped: "${tk.title}" is already claimed by ${tk.assignee}.`
    return roomTools.start_ticket_work({ id: tk.id, phase: 'build' }, target)
  }

  runRoom({
    room, project: roomProject, providers: store.settings.providers, problem: cleanProblem,
    temperature: store.settings.temperature, reasoningEffort,
    reasoningEffortForSeat: (seat) => seatEfforts.get(seat.id) ?? captureAgentReasoningEffort(seat, reasoningEffort),
    emit: emitRoomProgress, signal: controller.signal, tools: roomTools,
    onAssign: autoStartAssignedTicket,
    providerForSeat,
    recoverSeatModel: async (seat, error) => {
      const latestBaseProvider = store.settings.providers.find((p) => p.id === seat.providerId)
      if (!latestBaseProvider) return null
      const currentAccountId = seat.providerAccountId ?? null
      const routed = modelRouter.routeAfterError({
        role: 'room_seat',
        provider: latestBaseProvider,
        accountId: currentAccountId,
        model: seat.model,
        error,
        allowFailover: modelRouteAutoFailoverEnabled(latestBaseProvider, currentAccountId),
        localOnly: roomProject.localOnly === true,
        contextLimit: effectiveContextLimit(latestBaseProvider, seat.model, currentAccountId),
        reasoningEffort: seatEfforts.get(seat.id) ?? captureAgentReasoningEffort(seat, reasoningEffort),
        budget: room.workerBudget?.maxActive ?? null,
        source: 'room_seat_recovery',
      })
      if (routed.signal) {
        for (const audit of routed.audit) {
          await auditScopedModelRoute('room', audit, routed.signal, {
            roomId: room.id,
            seatId: seat.id,
            seatName: seat.name,
            jobId: job?.id ?? null,
            missionId: supervisorMission.id,
          })
        }
      }
      if (!routed.signal || !routed.route.selected || routed.route.selectedAccountId === currentAccountId) return null
      const providerAccountId = routed.route.selectedAccountId
      const provider = await runtimeProviderWithFreshModels(latestBaseProvider, providerAccountId)
      assertProviderEndpoint(provider)
      const liveSeat = room.seats.find((candidate) => candidate.id === seat.id)
      if (liveSeat) {
        liveSeat.providerId = latestBaseProvider.id
        liveSeat.providerAccountId = providerAccountId
        liveSeat.model = seat.model
      }
      room.updatedAt = Date.now()
      persistRooms()
      const failoverText = modelRouteFailoverText(latestBaseProvider, routed.route.audit, routed.signal)
      const notice = `${failoverText}\n${roomRouteHandoff(room, seat, failoverText)}`
      job = patchJob(job, {
        metadata: {
          modelRoute: {
            action: 'room_failover',
            seatId: seat.id,
            seatName: seat.name,
            providerId: latestBaseProvider.id,
            model: seat.model,
            preferredProviderAccountId: currentAccountId,
            failoverProviderAccountId: providerAccountId,
            resetPolicy: modelRouteResetPolicy(latestBaseProvider, currentAccountId),
            resetAt: routed.signal.resetAt,
            reason: routed.route.audit.reason,
          },
        },
      })
      touchSupervisor()
      return { provider, providerAccountId, model: seat.model, notice }
    },
    contextLimitForSeat: (seat) => {
      if (!store.settings.autoCompact) return 0
      const provider = providerForSeat(seat)
      if (!provider) return 0
      // Task #41: in a mixed-model room every seat compacts to the lowest seat's
      // limit so no model silently overflows. Fall back to this seat's own limit
      // if the floor can't be computed (e.g. another seat is misconfigured).
      const floor = roomContextLimit(
        room.seats
          .map((s) => ({ provider: providerForSeat(s), model: s.model }))
          .filter((entry): entry is { provider: Provider; model: string } => !!entry.provider),
        {
          cloudLimit: store.settings.contextLimit,
          localLimit: store.settings.localContextLimit,
          contextLimitExplicit: store.settings.contextLimitExplicit,
          localContextLimitExplicit: store.settings.localContextLimitExplicit,
          override: store.settings.contextLimitOverride,
          engineModelProfiles: store.settings.engineModelProfiles,
        },
      )
      return floor > 0 ? floor : effectiveContextLimit(provider, seat.model)
    },
    finalGate: () => {
      return roomFinalGateBlockers({ tickets: roomTickets(), lastCodeChangeAt, greenEvidenceAt })
    },
  })
    .catch((err) => emitRoomProgress({ roomId, type: 'error', message: String(err?.message ?? err), fatal: true }))
    .finally(() => {
      clearTimeout(timeout)
      roomAborters.delete(roomId)
      runLease.release()
      const failed = room.state === 'failed' || room.state === 'error'
      const parked = !controller.signal.aborted && !failed && parkRoomIfWorkRemains(job, room, project)
      if (!parked) {
        touchSupervisor({ currentTurnSeat: null })
        const outcome = controller.signal.aborted ? 'paused' : failed ? 'failed' : 'completed'
        const reason = failed ? 'Room ended with an error.' : controller.signal.aborted ? 'Room paused after the active run stopped.' : undefined
        finishRoomSupervisorMission(job, outcome, reason)
        finishJob(job, outcome, reason)
      }
      room.updatedAt = Date.now()
      persistRooms()
    })

  startupComplete = true
  return { ok: true, room }
  } finally {
    if (!startupComplete) {
      clearTimeout(timeout)
      if (roomAborters.get(roomId) === controller) roomAborters.delete(roomId)
      runLease.release()
    }
  }
}
  return { startRoomRun }
}
