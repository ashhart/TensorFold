import type { Project, Room, ConnectorEvent, CreateMissionInput, CreatePrLoopInput, Mission, MissionCheckpoint, MissionJournalEvent, MissionPendingApproval, MissionStatus } from '../types'
import { mockState, uid } from './state'
import { simulate, simulateRoom, thread } from './fixtures-core'
import { autoEmit, mkAuto, mockJob, mt, replaceMockJob, ticketEmit } from './fixtures-data'
import { activeOnly, createMockLocalPrLoop, mockArchive, mockMissionInspection, mockMissionSnapshot, updateMockMission } from './runtime'
import type { TensorFoldStudioApi } from '../ipc'

export function createMockProjectsApi(): Pick<TensorFoldStudioApi, 'getMissionSnapshot' | 'inspectMission' | 'createMission' | 'createLocalPrLoop' | 'setMissionStatus' | 'recordMissionHeartbeat' | 'addMissionCheckpoint' | 'appendMissionJournal' | 'requestMissionApproval' | 'resolveMissionApproval' | 'getProjects' | 'createProject' | 'updateProject' | 'setProjectExecutionTarget' | 'renameProject' | 'deleteProject' | 'onProjectUpdate' | 'onThreadUpdate' | 'getRooms' | 'createRoom' | 'updateRoom' | 'renameRoom' | 'deleteRoom' | 'onRoomUpdate' | 'startRoom' | 'steerRoom' | 'mutateRoomMemory' | 'cancelRoom' | 'mergeRoomWorktree' | 'onRoomEvent' | 'getTickets' | 'createTicket' | 'updateTicket' | 'deleteTicket' | 'generateTickets' | 'workTicket' | 'onTicketUpdate' | 'getJobs' | 'retryJob' | 'resumeJob' | 'deadLetterJob' | 'onJobUpdate' | 'getAutomations' | 'createAutomation' | 'updateAutomation' | 'deleteAutomation' | 'runAutomation' | 'onAutomationUpdate' | 'onMission' | 'onIncident'> {

  return {
    getMissionSnapshot: async () => mockMissionSnapshot(),
    inspectMission: async (id: string) => mockMissionInspection(id),
    createMission: async (input: CreateMissionInput) => {
      const now = Date.now()
      const mission: Mission = {
        id: uid(), title: input.title || 'Untitled mission', objective: input.objective || '', status: 'queued', runBudgetClass: input.runBudgetClass || 'mission',
        stopCondition: input.stopCondition || '', repeatMode: input.repeatMode || 'until_condition', contextMode: input.contextMode || 'build_on_previous',
        roleProfileId: input.roleProfileId || 'operator', fanOutBudget: input.fanOutBudget || 'single',
        maxRuns: input.maxRuns ?? 15, costCapUsd: input.costCapUsd ?? null, expiresAt: input.expiresAt ?? null, destructiveApprovalRequired: input.destructiveApprovalRequired ?? true,
        budget: { executionClass: input.runBudgetClass || 'mission', label: input.runBudgetClass || 'Mission', maxWallSeconds: 3600, heartbeatIntervalSeconds: 30, stallGraceSeconds: 120, requiresHeartbeat: true, startedAt: null, deadlineAt: null },
        heartbeat: { intervalSeconds: 30, stallGraceSeconds: 120, required: true, count: 0, lastBeatAt: null, nextDueAt: null, missedSince: null },
        checkpoints: [], journal: [{ id: uid(), kind: 'created', actor: 'system', message: 'Mission queued.', createdAt: now, status: 'queued' }], scratchMemory: [], pendingApprovals: [],
        threadId: input.threadId ?? null, projectId: input.projectId ?? null, cwd: input.cwd ?? null, providerId: input.providerId ?? mockState.settings.activeProviderId, model: input.model ?? mockState.settings.activeModel,
        createdAt: now, updatedAt: now, startedAt: null, finishedAt: null, metadata: input.metadata,
      }
      mockState.mockMissions = [mission, ...mockState.mockMissions]
      return mission
    },
    createLocalPrLoop: async (input: CreatePrLoopInput) => createMockLocalPrLoop(input),
    setMissionStatus: async (id, status: MissionStatus, message) => updateMockMission(id, (mission) => ({ ...mission, status, updatedAt: Date.now(), startedAt: status === 'running' ? mission.startedAt ?? Date.now() : mission.startedAt, finishedAt: ['completed', 'failed', 'cancelled'].includes(status) ? Date.now() : mission.finishedAt, journal: [...mission.journal, { id: uid(), kind: status === 'running' ? 'started' : status === 'paused' ? 'paused' : status === 'completed' ? 'completed' : status === 'cancelled' ? 'cancelled' : 'status', actor: 'user', message: message || `Mission marked ${status}.`, createdAt: Date.now(), status }] })),
    recordMissionHeartbeat: async (id, message) => updateMockMission(id, (mission) => ({ ...mission, heartbeat: { ...mission.heartbeat, count: mission.heartbeat.count + 1, lastBeatAt: Date.now(), nextDueAt: Date.now() + mission.heartbeat.intervalSeconds * 1000, missedSince: null }, updatedAt: Date.now(), journal: [...mission.journal, { id: uid(), kind: 'heartbeat', actor: 'agent', message: message || 'Heartbeat recorded.', createdAt: Date.now(), status: mission.status }] })),
    addMissionCheckpoint: async (id, checkpoint: Partial<MissionCheckpoint> & { label: string; summary: string }) => updateMockMission(id, (mission) => {
      const nextCheckpoint: MissionCheckpoint = { id: uid(), sequence: mission.checkpoints.length + 1, label: checkpoint.label, summary: checkpoint.summary, status: checkpoint.status || mission.status, createdAt: Date.now(), data: checkpoint.data, artifactRefs: checkpoint.artifactRefs }
      return { ...mission, checkpoints: [...mission.checkpoints, nextCheckpoint], updatedAt: Date.now(), journal: [...mission.journal, { id: uid(), kind: 'checkpoint', actor: 'agent', message: checkpoint.summary, checkpointId: nextCheckpoint.id, createdAt: Date.now(), status: nextCheckpoint.status }] }
    }),
    appendMissionJournal: async (id, event: Partial<MissionJournalEvent> & { message: string }) => updateMockMission(id, (mission) => ({ ...mission, updatedAt: Date.now(), journal: [...mission.journal, { id: uid(), kind: event.kind || 'note', actor: event.actor || 'user', message: event.message, createdAt: Date.now(), status: event.status, checkpointId: event.checkpointId, approvalId: event.approvalId, data: event.data }] })),
    requestMissionApproval: async (id, approval: Partial<MissionPendingApproval> & { callId: string; title: string; summary: string }) => updateMockMission(id, (mission) => ({ ...mission, status: 'waiting_approval', updatedAt: Date.now(), pendingApprovals: [...mission.pendingApprovals, { id: uid(), callId: approval.callId, itemId: approval.itemId, title: approval.title, summary: approval.summary, requestedAt: Date.now(), expiresAt: approval.expiresAt ?? null, data: approval.data }] })),
    resolveMissionApproval: async (id, approvalId, decision) => updateMockMission(id, (mission) => ({ ...mission, status: mission.pendingApprovals.length <= 1 ? 'running' : 'waiting_approval', updatedAt: Date.now(), pendingApprovals: mission.pendingApprovals.filter((approval) => approval.id !== approvalId && approval.callId !== approvalId), journal: [...mission.journal, { id: uid(), kind: 'approval_resolved', actor: 'user', message: `${approvalId}: ${decision}.`, createdAt: Date.now(), status: mission.status, approvalId, data: { decision } }] })),
    getProjects: async () => activeOnly(mockState.projects),
    createProject: async (opts) => {
      const now = Date.now()
      const p: Project = {
        id: opts.id || uid(),
        name: opts.name || 'New project',
        cwd: opts.cwd || '',
        workspace: opts.workspace ?? null,
        memory: opts.memory ?? '',
        team: opts.team ?? '',
        mode: opts.mode || 'solo',
        localOnly: opts.localOnly ?? false,
        executionTarget: opts.executionTarget === 'orchestrator' ? 'orchestrator' : 'local',
        orchestratorId: opts.orchestratorId ?? null,
        defaultProviderId: opts.defaultProviderId ?? null,
        defaultProviderAccountId: opts.defaultProviderAccountId ?? null,
        defaultModel: opts.defaultModel ?? null,
        kanban: opts.kanban ?? { modelMode: 'single' },
        tags: opts.tags ?? [],
        archivedAt: opts.archivedAt ?? null,
        createdAt: opts.createdAt ?? now,
        updatedAt: now,
      }
      const existing = mockState.projects.find((project) => project.id === p.id)
      if (existing) {
        Object.assign(existing, { ...p, createdAt: existing.createdAt })
        return existing
      }
      mockState.projects = [p, ...mockState.projects]
      return p
    },
    updateProject: async (id, patch) => { const p = mockState.projects.find((x) => x.id === id)!; if (p) Object.assign(p, patch); return p },
    setProjectExecutionTarget: async (id, executionTarget, orchestratorId = null, configPatch = {}) => {
      const p = mockState.projects.find((x) => x.id === id)!
      if (p) Object.assign(p, configPatch, { executionTarget, orchestratorId: executionTarget === 'orchestrator' ? orchestratorId : null, updatedAt: Date.now() })
      return p
    },
    renameProject: async (id, name) => { const p = mockState.projects.find((x) => x.id === id); if (p) { p.name = name.trim() || 'New project'; p.updatedAt = Date.now() } },
    deleteProject: async (id) => { mockArchive('project', id) },
    onProjectUpdate: () => () => { },
    onThreadUpdate: () => () => { },
    getRooms: async () => activeOnly(mockState.mockRooms),
    createRoom: async (projectId, name) => {
      const r: Room = {
        id: uid(), projectId, name: name || 'New room', seats: [],
        rootCwd: null, workspaceCwd: null, scratchpadCwd: null,
        collaborationMode: 'frontier_plans_local_builds', deliberationEffort: 'normal', workerBudget: { maxActive: 2, maxStarts: 4 }, memory: [], memorySummary: '',
        problem: '', state: 'created', events: [], finalText: '', createdAt: Date.now(), updatedAt: Date.now(),
      }
      mockState.mockRooms = [r, ...mockState.mockRooms]
      return r
    },
    updateRoom: async (id, patch) => { const r = mockState.mockRooms.find((x) => x.id === id)!; if (r) Object.assign(r, patch); return r },
    renameRoom: async (id, name) => { const r = mockState.mockRooms.find((x) => x.id === id); if (r) { r.name = name.trim() || 'New room'; r.updatedAt = Date.now() } },
    deleteRoom: async (id) => { mockArchive('room', id) },
    onRoomUpdate: () => () => { },
    startRoom: async (roomId, problem) => {
      const room = mockState.mockRooms.find((r) => r.id === roomId)
      if (!room) throw new Error('Room not found')
      const clean = problem.trim()
      if (!clean) throw new Error('Add a room objective before starting.')
      if (!room.seats.length) throw new Error('Add at least one agent to the room first.')
      room.problem = clean
      room.workspaceCwd = room.workspaceCwd || `/Users/example/.vontra/rooms/${room.id}/code`
      room.state = 'queued'
      room.updatedAt = Date.now()
      simulateRoom(roomId, clean)
      return { ok: true, room }
    },
    steerRoom: async (roomId, text) => {
      const room = mockState.mockRooms.find((x) => x.id === roomId)!
      const item = { id: uid(), tick: room.events.length + 1, kind: 'user', seat: 'you', target: 'all', text: text.trim(), createdAt: Date.now() }
      room.events.push(item)
      return item
    },
    mutateRoomMemory: async (roomId, command) => {
      const room = mockState.mockRooms.find((x) => x.id === roomId)!
      if (command.action === 'add') room.memory = [{ id: uid(), kind: command.kind, text: command.text, source: 'manual', createdAt: Date.now(), updatedAt: Date.now() }, ...(room.memory ?? [])]
      else room.memory = (room.memory ?? []).filter((entry) => entry.id !== command.entryId)
      return room
    },
    cancelRoom: async () => { },
    mergeRoomWorktree: async () => ({ ok: true, status: 'nothing', baseBranch: 'main', merged: [], remaining: [], conflict: null, blockers: [], message: '' }),
    onRoomEvent: (cb) => { mockState.roomListener = cb; return () => { mockState.roomListener = null } },
    getTickets: async () => activeOnly(mockState.mockTickets),
    createTicket: async (t) => { const tk = mt(t.projectId, uid(), t.title || 'Untitled', t.body || '', t.type || 'task', t.status || 'todo', t.priority || 'medium', t.effort || 'medium', t.tags || [], t.parentId || null, t.roomId || null); mockState.mockTickets = [tk, ...mockState.mockTickets]; return tk },
    updateTicket: async (id, patch) => { const tk = mockState.mockTickets.find((x) => x.id === id)!; if (tk) Object.assign(tk, patch); return tk },
    deleteTicket: async (id) => { mockArchive('ticket', id) },
    generateTickets: async (projectId, idea) => {
      await new Promise((r) => setTimeout(r, 1000))
      const eid = uid()
      const created = [
        mt(projectId, eid, `Epic: ${idea.slice(0, 52)}`, idea, 'epic', 'todo', 'high', 'large', ['epic'], null),
        mt(projectId, uid(), 'Set up project scaffolding', 'Initialize the repo and base config.', 'subticket', 'ready', 'high', 'small', ['setup'], eid),
        mt(projectId, uid(), 'Implement the core feature', 'Build the main capability described in the idea.', 'subticket', 'ready', 'high', 'large', ['core'], eid),
        mt(projectId, uid(), 'Add tests + a smoke run', 'Cover the core logic and verify end to end.', 'subticket', 'ready', 'medium', 'medium', ['tests'], eid),
      ]
      mockState.mockTickets = [...created, ...mockState.mockTickets]
      return created
    },
    workTicket: async (id, opts) => {
      const tk = mockState.mockTickets.find((t) => t.id === id)
      if (!tk) throw new Error('Ticket not found')
      const proj = tk.projectId ? mockState.projects.find((p) => p.id === tk.projectId) : null
      const t = thread(tk.title, proj?.cwd || '~', 0, 'running', [], 'task', tk.projectId || null)
      mockState.threads = [t, ...mockState.threads]
      tk.threadId = t.id; tk.status = 'in_progress'; tk.runState = 'running'; tk.startedAt = Date.now(); tk.assignee = opts?.model ?? 'gpt-4o'
      mockJob('ticket', tk.id, { projectId: tk.projectId, roomId: tk.roomId ?? null, ticketId: tk.id, threadId: t.id, currentThreadId: t.id, metadata: { title: tk.title, phase: opts?.phase ?? 'build' } })
      ticketEmit({ ...tk })
      simulate(t.id, tk.title)
      setTimeout(() => {
        tk.runState = 'done'; tk.status = opts?.phase === 'review' ? 'completed' : 'review'
        mockJob('ticket', tk.id, { status: 'completed', finishedAt: Date.now(), projectId: tk.projectId, ticketId: tk.id, threadId: t.id, currentThreadId: t.id })
        ticketEmit({ ...tk })
      }, 4200)
      return t
    },
    onTicketUpdate: (cb) => { mockState.ticketListener = cb; return () => { mockState.ticketListener = null } },
    getJobs: async () => mockState.mockJobs,
    retryJob: async (id, reason, nextWakeAt, resumePlan) => {
      const job = mockState.mockJobs.find((candidate) => candidate.id === id)
      if (!job || job.status === 'completed' || job.status === 'cancelled') throw new Error('Job cannot be changed from its current state.')
      return replaceMockJob(id, {
        status: 'queued',
        nextWakeAt: nextWakeAt ?? null,
        blockedReason: reason ?? null,
        resumePlan: resumePlan ?? 'Retry from the last safe checkpoint.',
        leaseOwner: null,
        leaseExpiresAt: null,
        currentRunId: null,
      }, true)
    },
    resumeJob: async (id, resumePlan) => replaceMockJob(id, {
      status: 'queued',
      nextWakeAt: null,
      blockedReason: null,
      resumePlan: resumePlan ?? 'Resume deliberately from the current state.',
      leaseOwner: null,
      leaseExpiresAt: null,
      currentRunId: null,
    }),
    deadLetterJob: async (id, reason) => {
      const job = mockState.mockJobs.find((candidate) => candidate.id === id)
      if (!job || job.status === 'completed' || job.status === 'cancelled') throw new Error('Job cannot be changed from its current state.')
      return replaceMockJob(id, {
        status: 'dead_letter',
        blockedReason: reason || 'Moved to dead letter by user.',
        leaseOwner: null,
        leaseExpiresAt: null,
        finishedAt: Date.now(),
      }, true)
    },
    onJobUpdate: (cb) => { mockState.jobListener = cb; return () => { mockState.jobListener = null } },
    getAutomations: async () => mockState.mockAutomations,
    createAutomation: async (a) => { const auto = mkAuto({ ...a, id: uid() }); mockState.mockAutomations = [auto, ...mockState.mockAutomations]; return auto },
    updateAutomation: async (id, patch) => { const a = mockState.mockAutomations.find((x) => x.id === id)!; if (a) Object.assign(a, patch, { updatedAt: Date.now() }); return a },
    deleteAutomation: async (id) => { mockState.mockAutomations = mockState.mockAutomations.filter((a) => a.id !== id) },
    runAutomation: async (id) => {
      const a = mockState.mockAutomations.find((x) => x.id === id)
      if (!a) return { ok: false }
      const run = { id: uid(), startedAt: Date.now(), finishedAt: null as number | null, status: 'running' as const, output: '' }
      a.runs = [run as any, ...a.runs].slice(0, 20); a.lastRunAt = run.startedAt; autoEmit(a)
      setTimeout(() => {
        const r = a.runs[0] as any
        r.status = 'ok'; r.finishedAt = Date.now()
        r.output = `Demo run of "${a.name}". In the desktop app this runs ${a.mode === 'agent' ? 'a full agent loop' : 'a single completion'} on ${a.model} and delivers to ${a.deliverTo}.`
        if (a.deliverTo !== 'inbox') {
          const target = a.deliverTo === 'telegram' ? 'Telegram' : 'Email'
          const event: ConnectorEvent = {
            id: uid(), connectorId: a.deliverTo, pluginName: target, action: 'automation_result_ready',
            direction: 'outbound', classification: 'approval', priority: 'normal', trustLevel: 'suggest',
            requiresApproval: true, approvalId: `automation-${a.id}-${r.id}`,
            title: `${a.name} result ready`, summary: r.output.slice(0, 500), actor: target,
            threadKey: `automation:${a.id}`, payloadKeys: ['automationId', 'runId', 'output'], createdAt: Date.now(),
          }
          mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
        }
        a.lastStatus = 'ok'; autoEmit(a)
      }, 1600)
      return { ok: true }
    },
    onAutomationUpdate: (cb) => { mockState.autoListener = cb; return () => { mockState.autoListener = null } },
    onMission: () => () => { },
    onIncident: () => () => { },
  }
}
