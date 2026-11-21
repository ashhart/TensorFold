
import type { RoomSupervisorMetadataPatch } from '../../agent/jobs'

import type { Project, ReasoningEffortSetting, Room, Ticket, Thread } from '../../agent/types'

import { resolveDelegationRoute } from '../../agent/delegation-route'
import { createSharedMemory, MEMORY_MAX_BLOCK_CHARS, MEMORY_MAX_BLOCKS, renderSharedMemory, writeBlock } from '../../agent/shared-room-memory'

import type { WorkspaceRuntimeIpcDeps } from './workspace-runtime-ipc'
import type { RoomWorkerBudget } from '../../agent/types'
import { pickTicketUserPatch, ticketHasActiveRun } from './room-ticket-patch-policy'

export interface RoomToolScope {
  reasoningEffort: ReasoningEffortSetting
  seatEfforts: ReadonlyMap<string, ReasoningEffortSetting>
  roomTickets: () => Ticket[]
  room: Room
  resolveTicket: (args: any) => { ticket?: Ticket; error?: string; }
  findTicket: (args: any) => Ticket | undefined
  hasMeaningfulEvidence: (tk: Ticket) => boolean
  touchSupervisor: (patch?: RoomSupervisorMetadataPatch) => void
  activeTicketWorkerCount: () => number
  workerBudget: RoomWorkerBudget
  workerStarts: number
  lastCodeChangeAt: number
  greenEvidenceAt: number
  scheduleAutomaticReviewAfterBuild: (tk: Ticket, builderSeat: string, paths: string[]) => Promise<void>
  roomProject: Project
  roomWorkspace: string
  controller: AbortController
}

export function createRoomTools(ctx: WorkspaceRuntimeIpcDeps, runtimeScope: RoomToolScope): Record<string, (args: any, seat: string) => Promise<string>> {
  const {
    path,
    randomUUID,
    store,
    auditEvent,
    auditPreview,
    persistRooms,
    persistTickets,
    persistSubAgents,
    emitTicket,
    publishRemote,
    activeJob,
    aborters,
    isTicketStarting,
    runtimeProvider,
    appendRoomSupervisorEvent,
    roomTicketConflictBlockers,
    summarizeRoomLeftToWork,
    describeTicketForRoom,
    mkTicket,
    normalizeTicketCapabilities,
    normalizeStringList,
    ticketEvidenceFromArgs,
    ticketFileRefs,
    startTicketWork,
    createSubAgentFromLaunch,
    parentForRoomSeat,
    persistSubAgents: persistSubAgentsFromCtx,
    startSubAgent,
    runReadOnlySubAgent,
    executeTool,
  } = ctx
  const persistSubAgentsSafe = persistSubAgentsFromCtx ?? persistSubAgents
  return {
    list_tickets: async () => {
      const board = runtimeScope.roomTickets()
      return board.length ? board.map(describeTicketForRoom).join('\n') : 'The board is empty.'
    },
    left_to_work: async () => summarizeRoomLeftToWork(runtimeScope.roomTickets()),
    create_ticket: async (args) => {
      const tk = mkTicket(runtimeScope.room.projectId, {
        roomId: runtimeScope.room.id,
        title: args?.title || 'Untitled',
        body: args?.body || '',
        priority: args?.priority,
        effort: args?.effort,
        type: args?.type || 'task',
        status: 'ready',
        tags: args?.tags,
        risk: args?.risk,
        allowedCapabilities: args?.allowedCapabilities,
        requiredEvidence: args?.requiredEvidence,
        acceptanceCriteria: args?.acceptanceCriteria,
        evidence: args?.evidence,
      })
      store.tickets.unshift(tk); persistTickets(); emitTicket(tk)
      return `Created ticket "${tk.title}" with ${tk.risk ?? 'low'} risk, allowed capabilities ${normalizeTicketCapabilities(tk.allowedCapabilities, ['read']).join(', ')}, and ${tk.requiredEvidence?.length ?? 0} required evidence item(s).`
    },
    claim_ticket: async (args, seat) => {
      const resolved = runtimeScope.resolveTicket(args)
      if (resolved.error) return resolved.error
      const tk = resolved.ticket
      if (!tk) return `No ticket matches "${args?.title ?? args?.id ?? ''}".`
      if (tk.assignee && tk.assignee !== seat) return `Already claimed by ${tk.assignee}. Pick another ticket.`
      tk.assignee = seat; tk.status = 'in_progress'; tk.updatedAt = Date.now(); persistTickets(); emitTicket(tk)
      const required = normalizeStringList(tk.requiredEvidence, [])
      const caps = normalizeTicketCapabilities(tk.allowedCapabilities, ['read']).join(', ')
      return `You claimed "${tk.title}". Allowed capabilities: ${caps}. Required evidence: ${required.length ? required.join('; ') : 'none'}.`
    },
    release_ticket: async (args, seat) => {
      const selector = typeof args?.id === 'string' && args.id ? { id: args.id } : args
      const { ticket, error } = runtimeScope.resolveTicket(selector)
      if (error) return error
      const room = runtimeScope.room
      if (!ticket || ticket.archivedAt || ticket.roomId !== room.id || ticket.projectId !== room.projectId) {
        return 'No matching ticket.'
      }
      const name = seat.trim().toLowerCase()
      const configured = room.seats.filter((candidate) => candidate.name.trim().toLowerCase() === name)
      if (configured.length !== 1 || configured[0].name !== seat) return 'The requesting seat is not uniquely configured.'
      if (!ticket.assignee) return 'This ticket has no claim to release.'
      if (ticket.assignee !== seat) return `Only ${ticket.assignee} can release "${ticket.title}".`
      if (ticket.status === 'completed') return 'A completed ticket cannot be released.'
      const activeThread = !!ticket.threadId && aborters?.has?.(ticket.threadId)
      if (ticketHasActiveRun(ticket, activeThread || !!isTicketStarting?.(ticket.id))) {
        return 'Stop the active ticket run before releasing its claim.'
      }
      Object.assign(ticket, pickTicketUserPatch({ status: 'ready' }), { updatedAt: Date.now() })
      persistTickets()
      emitTicket(ticket)
      return `Released "${ticket.title}" to Ready.`
    },
    add_ticket_evidence: async (args, seat) => {
      const tk = runtimeScope.findTicket(args)
      if (!tk) return 'No matching ticket.'
      if (tk.assignee && tk.assignee !== seat) return `Only ${tk.assignee} can add evidence to "${tk.title}".`
      const evidence = ticketEvidenceFromArgs(args, seat)
      if (!evidence.length) return 'No evidence summary was provided.'
      tk.assignee = tk.assignee ?? seat
      tk.evidence = [...(tk.evidence ?? []), ...evidence]
      tk.updatedAt = Date.now()
      persistTickets(); emitTicket(tk)
      return `Added ${evidence.length} evidence item(s) to "${tk.title}".`
    },
    complete_ticket: async (args, seat) => {
      const resolved = runtimeScope.resolveTicket(args)
      if (resolved.error) return resolved.error
      const tk = resolved.ticket
      if (!tk) return 'No matching ticket.'
      if (tk.assignee && tk.assignee !== seat) return `Only ${tk.assignee} can complete "${tk.title}".`
      const evidence = ticketEvidenceFromArgs(args, seat)
      if (evidence.length) tk.evidence = [...(tk.evidence ?? []), ...evidence]
      const required = normalizeStringList(tk.requiredEvidence, [])
      // Only MEANINGFUL evidence satisfies the required-evidence gate — reservation/placeholder notes
      // appended by start_ticket_work ("Reserved for build/review worker.") must not count, or a
      // ticket can be flipped to 'completed' with zero real evidence (the room final gate uses the
      // same meaningful filter, so this keeps the per-ticket gate consistent with it).
      if (required.length && !runtimeScope.hasMeaningfulEvidence(tk)) return `Cannot complete "${tk.title}" until evidence is attached: ${required.join('; ')}.`
      tk.assignee = seat; tk.status = 'completed'; tk.updatedAt = Date.now(); persistTickets(); emitTicket(tk)
      return `Completed "${tk.title}" with ${tk.evidence?.length ?? 0} evidence item(s).`
    },
    start_ticket_work: async (args, seat) => {
      const resolved = runtimeScope.resolveTicket(args)
      if (resolved.error) return resolved.error
      const tk = resolved.ticket
      if (!tk) return `No ticket matches "${args?.title ?? args?.id ?? ''}".`
      const phase = args?.phase === 'review' ? 'review' : 'build'
      const declaredPaths: string[] = normalizeStringList(args?.paths, [])
      const mergeDecision = typeof args?.mergeDecision === 'string' ? args.mergeDecision.trim() : ''
      const blockers: string[] = roomTicketConflictBlockers({ ticket: tk, tickets: runtimeScope.roomTickets(), phase, seat, paths: declaredPaths })
      const overlapBlockers = blockers.filter((blocker) => /\boverlaps with\b/i.test(blocker))
      const hardBlockers = blockers.filter((blocker) => !overlapBlockers.includes(blocker))
      const effectiveBlockers = mergeDecision && overlapBlockers.length ? hardBlockers : blockers
      if (blockers.length) {
        if (effectiveBlockers.length) {
          runtimeScope.touchSupervisor({ phase: { phase, status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: effectiveBlockers.join(' ') } })
          return `Blocked by room supervisor: ${effectiveBlockers.join(' ')}`
        }
        appendRoomSupervisorEvent(runtimeScope.room, `Merge decision accepted for "${tk.title}": ${mergeDecision}`, {
          tool: 'merge_decision',
          args: { ticketId: tk.id, phase, seat, paths: declaredPaths, blockers: overlapBlockers },
        })
        tk.evidence = [
          ...(tk.evidence ?? []),
          {
            id: randomUUID(),
            kind: 'note',
            summary: `Supervisor merge decision: ${mergeDecision}. Waived overlap blocker(s): ${overlapBlockers.join(' ')}`,
            ref: declaredPaths[0] ?? null,
            createdBy: 'room-supervisor',
            createdAt: Date.now(),
          },
        ]
      }
      const activeWorkers = runtimeScope.activeTicketWorkerCount()
      if (activeWorkers >= runtimeScope.workerBudget.maxActive) {
        runtimeScope.touchSupervisor({ phase: { phase, status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: `Worker budget busy: ${activeWorkers}/${runtimeScope.workerBudget.maxActive} active.` } })
        return `Room worker budget busy: ${activeWorkers}/${runtimeScope.workerBudget.maxActive} active ticket workers. Use left_to_work and wait for a worker result.`
      }
      if (runtimeScope.workerStarts >= runtimeScope.workerBudget.maxStarts) {
        runtimeScope.touchSupervisor({ phase: { phase, status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: `Worker budget exhausted: ${runtimeScope.workerStarts}/${runtimeScope.workerBudget.maxStarts} started.` } })
        return `Room worker budget exhausted: ${runtimeScope.workerStarts}/${runtimeScope.workerBudget.maxStarts} ticket workers started in this room run.`
      }
      const seatConfig = runtimeScope.room.seats.find((candidate) => candidate.name === seat)
      if (!seatConfig) return 'Cannot start worker: room seat is no longer configured.'
      // Snapshot before the pre-launch mutation so a failed launch can restore the ticket
      // instead of wedging it (runState 'queued' blocks retries via roomTicketConflictBlockers,
      // counts against the worker budget, and keeps the final gate open).
      const prevAssignee = tk.assignee
      const prevStatus = tk.status
      runtimeScope.workerStarts++
      // Only a BUILD-phase start takes ownership of the ticket. A review-phase start must NOT
      // overwrite the builder's assignee with the reviewer — doing so erases the builder's claim and
      // makes a later build-phase conflict check (roomTicketConflictBlockers) report the ticket as
      // claimed by the reviewer, blocking the original builder from their own ticket. The reviewer's
      // identity is tracked separately via the review evidence (createdBy), not tk.assignee.
      if (phase === 'build') tk.assignee = seat
      tk.status = 'in_progress'
      tk.runState = 'queued'
      if (declaredPaths.length) {
        tk.evidence = [
          ...(tk.evidence ?? []),
          ...declaredPaths.map((ref) => ({ id: randomUUID(), kind: 'file' as const, summary: `Reserved for ${phase} worker.`, ref, createdBy: seat, createdAt: Date.now() })),
        ]
      }
      tk.updatedAt = Date.now()
      persistTickets(); emitTicket(tk)
      runtimeScope.touchSupervisor({ phase: { phase, status: 'queued', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: 'Ticket worker queued.' } })
      const knownPaths = ticketFileRefs(tk, declaredPaths)
      let workerThreadId: string | null = null
      let thread: Thread
      try {
        thread = await startTicketWork({
          ticketId: tk.id,
          opts: { phase, providerId: seatConfig.providerId, providerAccountId: seatConfig.providerAccountId ?? null, model: seatConfig.model },
          source: {
            roomId: runtimeScope.room.id,
            seat,
            reasoningEffort: runtimeScope.seatEfforts.get(seatConfig.id) ?? runtimeScope.reasoningEffort,
            onSettled: ({ codeChanged, testsPassed, failed, jobId }) => {
              if (codeChanged) runtimeScope.lastCodeChangeAt = Date.now()
              if (testsPassed) runtimeScope.greenEvidenceAt = Date.now()
              runtimeScope.touchSupervisor({
                phase: {
                  phase,
                  status: tk.status === 'needs_clarification' ? 'failed' : 'completed',
                  ticketId: tk.id,
                  ticketTitle: tk.title,
                  seat,
                  threadId: workerThreadId,
                  jobId: jobId ?? activeJob('ticket', tk.id)?.id ?? null,
                  paths: knownPaths,
                  summary: tk.status === 'needs_clarification' ? 'Ticket worker needs clarification.' : 'Ticket worker finished.',
                },
              })
              if (phase === 'build' && !failed) void runtimeScope.scheduleAutomaticReviewAfterBuild(tk, seat, knownPaths)
            },
          },
        })
      } catch (launchError: any) {
        // Launch failed: unwind the pre-launch mutation so the ticket is retryable instead
        // of wedged as 'queued' (which blocks retries, burns budget, and holds the gate).
        runtimeScope.workerStarts--
        tk.runState = null
        tk.status = prevStatus
        tk.assignee = prevAssignee
        tk.updatedAt = Date.now()
        persistTickets(); emitTicket(tk)
        runtimeScope.touchSupervisor({ phase: { phase, status: 'failed', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: `Worker launch failed: ${launchError?.message ?? launchError}` } })
        return `Cannot start worker: launch failed: ${launchError?.message ?? launchError}`
      }
      workerThreadId = thread.id
      runtimeScope.touchSupervisor({
        phase: {
          phase,
          status: 'running',
          ticketId: tk.id,
          ticketTitle: tk.title,
          seat,
          threadId: thread.id,
          jobId: activeJob('ticket', tk.id)?.id ?? null,
          paths: knownPaths,
          summary: 'Ticket worker running.',
        },
      })
      return `Room Supervisor started ${phase} worker thread ${thread.id} for "${tk.title}"${knownPaths.length ? ` on ${knownPaths.slice(0, 4).join(', ')}` : ''}.`
    },
    delegate: async (args, seat) => {
      const s = runtimeScope.room.seats.find((x) => x.name === seat)
      if (!s) return 'Cannot delegate: seat misconfigured.'
      // Optional cross-model route: targetSeat adopts that seat's provider/account/model;
      // explicit providerId/accountId/model override it; absent fields inherit this seat.
      // A local-only project may not delegate out to a cloud provider (same guard the
      // thread runtime and room seat config enforce).
      const routed = resolveDelegationRoute(store.settings.providers, {
        providerId: s.providerId,
        providerAccountId: s.providerAccountId,
        model: s.model,
        reasoningEffort: runtimeScope.seatEfforts.get(s.id) ?? runtimeScope.reasoningEffort,
      }, args ?? {}, runtimeScope.room.seats.map((seat) => ({ ...seat,
        reasoningEffort: runtimeScope.seatEfforts.get(seat.id) ?? runtimeScope.reasoningEffort,
      })), runtimeScope.roomProject.localOnly === true)
      if (!routed.route) return `Cannot delegate: ${routed.error}.`
      const routeAccountId = routed.route.providerAccountId ?? null
      const routeProvider = store.settings.providers.find((p) => p.id === routed.route.providerId)
      const prov = routeProvider ? runtimeProvider(routeProvider, routed.route.providerAccountId) : undefined
      if (!prov) return s.providerId === routed.route.providerId ? 'Cannot delegate: seat misconfigured.' : `Cannot delegate: provider "${routed.route.providerId}" is not configured.`
      const task = String(args?.task || '').trim() || 'Focused room subtask'
      const name = typeof args?.name === 'string' ? args.name.trim() : ''
      const runId = `subagent-${randomUUID()}`
      const created = createSubAgentFromLaunch(store.subAgents, {
        source: 'room_delegate',
        parent: parentForRoomSeat(runtimeScope.room, s),
        task,
        name,
        providerId: routed.route.providerId,
        providerAccountId: routeAccountId,
        model: routed.route.model,
        runId,
        cwd: runtimeScope.roomWorkspace,
      })
      store.subAgents = created.agents
      persistSubAgentsSafe()
      if (args && typeof args === 'object') {
        args.id = created.agent.id
        args.name = created.agent.name
        args.task = task
        args.parentSeat = seat
        args.providerId = routed.route.providerId
        args.providerAccountId = routeAccountId
        args.model = routed.route.model
        if (routed.notice) args.notice = routed.notice
        args.status = 'running'
      }
      const started = startSubAgent(store.subAgents, created.agent.id, runId)
      store.subAgents = started.agents
      persistSubAgentsSafe()
      const launched = await runReadOnlySubAgent(created.agent, prov, routed.route.model, {
        task,
        parentLabel: `room seat "${seat}" in room "${runtimeScope.room.name}"`,
        cwd: runtimeScope.roomWorkspace,
        projectId: runtimeScope.room.projectId,
        providerAccountId: routeAccountId,
        signal: runtimeScope.controller.signal,
        reasoningEffort: routed.route.reasoningEffort ?? runtimeScope.reasoningEffort,
        project: runtimeScope.roomProject,
        maxSteps: 10,
        maxWallMs: 150_000,
      })
      if (args && typeof args === 'object') {
        args.status = launched.agent.status
        args.result = launched.agent.result
        args.error = launched.agent.error
      }
      return routed.notice ? `${routed.notice}\n${launched.output}` : launched.output
    },
    // Room tools are read-only by default. Mutating tools need a future room-level approval UI.
    // memory_write is room state (the shared scratchpad), not workspace mutation, so it is allowed.
    memory_write: async (args, seat) => {
      const name = typeof args?.name === 'string' ? args.name.trim() : ''
      const content = typeof args?.content === 'string' ? args.content.trim() : ''
      if (!name || !content) return 'memory_write needs a "name" (decisions, plan, blockers, or any label) and "content".'
      if (!runtimeScope.room.sharedMemory) runtimeScope.room.sharedMemory = createSharedMemory()
      const mode = args?.mode === 'append' || args?.mode === 'append-bullet' ? args.mode : 'replace'
      const expectedRevision = typeof args?.expectedRevision === 'number' ? args.expectedRevision : undefined
      const result = writeBlock(runtimeScope.room.sharedMemory, { name, content, author: seat, mode, expectedRevision })
      if (!result.ok) {
        if (result.reason === 'conflict') return `Memory conflict on "${name}": current revision is ${result.currentRevision}. Re-read the block and retry.`
        if (result.reason === 'content-too-large') return `Memory block "${name}" is too large: content is capped at ${MEMORY_MAX_BLOCK_CHARS} characters. Summarize it and retry.`
        if (result.reason === 'block-limit') return `Shared memory is full (${MEMORY_MAX_BLOCKS} blocks). Update an existing block instead of adding a new one.`
        return 'Memory write needs a non-empty name.'
      }
      runtimeScope.room.updatedAt = Date.now()
      persistRooms()
      // Live push so the renderer reflects the write without waiting for a full snapshot.
      if (result.changed) publishRemote?.({ kind: 'room', data: runtimeScope.room })
      return result.changed
        ? `Memory "${result.block.name}" updated (rev ${result.block.revision}).\n${renderSharedMemory(runtimeScope.room.sharedMemory)}`
        : `Memory "${result.block.name}" already says that (rev ${result.block.revision}).`
    },
    read_file: async (args) => { const r = await executeTool('read_file', args, { cwd: runtimeScope.roomWorkspace }); return r.output.slice(0, 4000) },
    list_dir: async (args) => { const r = await executeTool('list_dir', args, { cwd: runtimeScope.roomWorkspace }); return r.output.slice(0, 2000) },
    search: async (args) => { const r = await executeTool('search', args, { cwd: runtimeScope.roomWorkspace }); return r.output.slice(0, 2000) },
    write_file: async (args) => {
      await auditEvent('room.tool_blocked', { roomId: runtimeScope.room.id, tool: 'write_file', path: auditPreview(args?.path) })
      return 'Blocked: room agents are read-only until a room-level approval gate is available. Create/claim a ticket and run the work in an approved thread.'
    },
    edit_file: async (args) => {
      await auditEvent('room.tool_blocked', { roomId: runtimeScope.room.id, tool: 'edit_file', path: auditPreview(args?.path) })
      return 'Blocked: room agents are read-only until a room-level approval gate is available. Create/claim a ticket and run the work in an approved thread.'
    },
    run_command: async (args) => {
      await auditEvent('room.tool_blocked', { roomId: runtimeScope.room.id, tool: 'run_command', command: auditPreview(args?.command) })
      return 'Blocked: room agents cannot run shell commands unattended. Ask the user to run an approved thread or terminal command.'
    },
  }
}
