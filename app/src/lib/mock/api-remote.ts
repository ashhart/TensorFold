import type { Project, Room, CliDefinition, CliSession } from '../types'
import { mockState, uid } from './state'
import { emit, simulate, simulateRoom, thread } from './fixtures-core'
import { mt } from './fixtures-data'
import { mockArchive, mockArchiveSnapshot, mockRemotePairingInfo, mockRemoteStatus, mockRemoteWorkSnapshot, mockRestore } from './runtime'
import type { TensorFoldStudioApi } from '../ipc'

export function createMockRemoteApi(): Pick<TensorFoldStudioApi, 'getRemoteStatus' | 'getMobileBridgeStatus' | 'configureMobileBridge' | 'issueMobilePairing' | 'getRemotePairingInfo' | 'regenerateRemoteToken' | 'issueDevicePairing' | 'listDeviceSessions' | 'revokeDeviceSession' | 'testRemoteClient' | 'getRemoteSnapshot' | 'getRemoteOrchestratorArchives' | 'listRemoteOrchestratorDir' | 'createRemoteOrchestratorProject' | 'updateRemoteOrchestratorProject' | 'deleteRemoteOrchestratorProject' | 'restoreRemoteOrchestratorProject' | 'createRemoteOrchestratorThread' | 'archiveRemoteOrchestratorThread' | 'restoreRemoteOrchestratorThread' | 'startRemoteOrchestratorThread' | 'guideRemoteOrchestratorThread' | 'cancelRemoteOrchestratorThread' | 'approveRemoteOrchestrator' | 'remoteListClis' | 'remoteTerminalStart' | 'remoteTerminalWrite' | 'remoteTerminalResize' | 'remoteTerminalRename' | 'remoteTerminalKill' | 'remoteTerminalList' | 'createRemoteOrchestratorRoom' | 'updateRemoteOrchestratorRoom' | 'renameRemoteOrchestratorRoom' | 'steerRemoteOrchestratorRoom' | 'mutateRemoteOrchestratorRoomMemory' | 'startRemoteOrchestratorRoom' | 'cancelRemoteOrchestratorRoom' | 'deleteRemoteOrchestratorRoom' | 'restoreRemoteOrchestratorRoom' | 'createRemoteOrchestratorTicket' | 'startRemoteOrchestratorTicketExecution' | 'updateRemoteOrchestratorTicket' | 'transitionRemoteOrchestratorTicket' | 'deleteRemoteOrchestratorTicket' | 'restoreRemoteOrchestratorTicket'> {
  const remoteTerminalSessions: CliSession[] = []
  return {
    getRemoteStatus: async () => mockRemoteStatus(),
    getMobileBridgeStatus: async () => ({ enabled: false, listening: false, url: null, host: null, port: null, certificateSha256: null, certificateExpiresAt: null, interfaces: [], clients: 0, lastError: null }),
    configureMobileBridge: async () => { throw new Error('Enable the private mobile connection in the desktop app.') },
    issueMobilePairing: async () => { throw new Error('Pair a phone from the desktop app.') },
    getRemotePairingInfo: async () => mockRemotePairingInfo(),
    regenerateRemoteToken: async () => {
      mockState.settings.remoteControl.authToken = 'demo_' + uid().replace(/-/g, '').slice(0, 24)
      return mockRemotePairingInfo()
    },
    // Per-device pairing (VST-035) — simulated: minted credentials and a demo device list.
    issueDevicePairing: async (label) => {
      const credential = 'pair_' + uid().replace(/-/g, '').slice(0, 20)
      mockState.deviceSessions = [
        ...(mockState.deviceSessions ?? []),
        { id: uid(), label: label?.trim() || 'Paired device', createdAt: Date.now(), lastSeenAt: Date.now(), revoked: false, clientInfo: 'pending redeem (demo)' },
      ]
      return { credential, expiresAt: Date.now() + 10 * 60_000, url: 'http://192.168.1.20:7860' }
    },
    listDeviceSessions: async () => mockState.deviceSessions ?? [
      { id: 'dev-demo-1', label: 'Demo iPhone', createdAt: Date.now() - 86_400_000, lastSeenAt: Date.now() - 120_000, revoked: false, clientInfo: 'Safari · iOS' },
    ],
    revokeDeviceSession: async (sessionId) => {
      mockState.deviceSessions = (mockState.deviceSessions ?? []).map((s) => s.id === sessionId ? { ...s, revoked: true } : s)
      return { revoked: true }
    },
    testRemoteClient: async (input) => {
      const client = {
        ...mockState.settings.remoteClient,
        ...input,
        authToken: input?.authToken === '••••••••' || input?.authToken === '[set]'
          ? mockState.settings.remoteClient.authToken
          : (input?.authToken ?? mockState.settings.remoteClient.authToken),
      }
      if (!client.serverUrl || !client.authToken) {
        const error = 'Server URL and token are required.'
        mockState.settings.remoteClient = { ...client, lastError: error }
        return { ok: false, error }
      }
      const snapshot = {
        serverName: 'TensorFold Orchestrator',
        generatedAt: Date.now(),
        activeThreadIds: mockState.threads.filter((t) => t.status === 'running').map((t) => t.id),
        activeRoomIds: mockState.mockRooms.filter((r) => r.state === 'running').map((r) => r.id),
        activeMissionIds: mockState.mockMissions.filter((m) => m.status === 'running').map((m) => m.id),
        work: mockRemoteWorkSnapshot(),
      }
      mockState.settings.remoteClient = { ...client, lastConnectedAt: Date.now(), lastError: null }
      return { ok: true, snapshot, status: mockRemoteStatus() }
    },
    getRemoteSnapshot: async (hostId) => {
      const host = mockState.settings.remoteOrchestrators.find((item) => item.id === hostId) ?? mockState.settings.remoteOrchestrators[0]
      if (!host) return { ok: false, hostId: null, name: 'Orchestrator', serverUrl: '', error: 'No Orchestrator configured.' }
      const now = Date.now()
      host.lastConnectedAt = now
      host.lastError = null
      host.updatedAt = now
      return {
        ok: true,
        hostId: host.id,
        name: host.name,
        serverUrl: host.serverUrl,
        generatedAt: now,
        activeThreadIds: mockState.threads.filter((thread) => ['running', 'thinking', 'streaming', 'tool'].includes(thread.status || '')).map((thread) => thread.id),
        activeRoomIds: mockState.mockRooms.filter((room) => ['queued', 'running', 'waiting_external', 'waiting_user'].includes(room.state)).map((room) => room.id),
        activeMissionIds: mockState.mockMissions.filter((mission) => ['running', 'queued', 'waiting_external', 'waiting_user'].includes(mission.status)).map((mission) => mission.id),
        work: mockRemoteWorkSnapshot(now),
        totalProjects: mockState.projects.length,
        totalThreads: mockState.threads.length,
        totalRooms: mockState.mockRooms.length,
        totalMissions: mockState.mockMissions.length,
        projects: mockState.projects,
        jobs: mockState.mockJobs,
      }
    },
    getRemoteOrchestratorArchives: async (hostId) => {
      const snapshot = mockArchiveSnapshot()
      return {
        projects: snapshot.projects.map((item) => ({ ...item, executionTarget: 'orchestrator' as const, orchestratorId: hostId ?? null })),
        threads: snapshot.threads.map((item) => ({ ...item, orchestratorHostId: hostId ?? null })),
        rooms: snapshot.rooms.map((item) => ({ ...item, orchestratorHostId: hostId ?? null })),
        tickets: snapshot.tickets.map((item) => ({ ...item, orchestratorHostId: hostId ?? null })),
      }
    },
    listRemoteOrchestratorDir: async (_hostId, dir) => {
      const base = dir || '/Users/example'
      return {
        ok: true,
        path: base,
        parent: base === '/' ? null : '/Users/example',
        entries: ['Projects', 'Documents', 'Desktop', 'Code'].map((name) => ({ name, path: `${base.replace(/\/$/, '')}/${name}`, isDirectory: true })),
      }
    },
    createRemoteOrchestratorProject: async (hostId, project) => {
      const p: Project = {
        id: uid(),
        name: project.name || 'Remote project',
        cwd: project.cwd || '/Users/example/Projects/remote-project',
        memory: '',
        team: '',
        mode: project.mode || 'solo',
        localOnly: project.localOnly ?? false,
        executionTarget: 'orchestrator',
        orchestratorId: hostId ?? null,
        defaultProviderId: project.defaultProviderId ?? null,
        defaultProviderAccountId: project.defaultProviderAccountId ?? null,
        defaultModel: project.defaultModel ?? null,
        kanban: project.kanban ?? { modelMode: 'single' },
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }
      mockState.projects = [p, ...mockState.projects]
      return p
    },
    updateRemoteOrchestratorProject: async (hostId, id, patch) => {
      const project = mockState.projects.find((item) => item.id === id)
      if (!project) throw new Error('Project not found')
      Object.assign(project, patch, { executionTarget: 'orchestrator', orchestratorId: hostId ?? project.orchestratorId ?? null, updatedAt: Date.now() })
      return project
    },
    deleteRemoteOrchestratorProject: async (hostId, id) => {
      const project = mockState.projects.find((item) => item.id === id)
      if (!project) return null
      mockArchive('project', id)
      return { ...project, executionTarget: 'orchestrator' as const, orchestratorId: hostId ?? project.orchestratorId ?? null }
    },
    restoreRemoteOrchestratorProject: async (hostId, id) => {
      const project = mockState.projects.find((item) => item.id === id)
      if (!project) return null
      mockRestore('project', id)
      return { ...project, executionTarget: 'orchestrator' as const, orchestratorId: hostId ?? project.orchestratorId ?? null }
    },
    createRemoteOrchestratorThread: async (_hostId, input) => {
      const proj = input.projectId ? mockState.projects.find((p) => p.id === input.projectId) : null
      const t = thread(input.kind === 'chat' ? 'New chat' : 'New task', input.cwd || proj?.cwd || '/Users/example/Projects/remote-project', 0, 'idle', [], input.kind === 'chat' ? 'chat' : 'task', input.projectId ?? null)
      mockState.threads = [t, ...mockState.threads]
      return t
    },
    archiveRemoteOrchestratorThread: async (hostId, id) => {
      mockArchive('thread', id)
      const item = mockState.threads.find((thread) => thread.id === id)
      return item ? { ...item, orchestratorHostId: hostId ?? null } : null
    },
    restoreRemoteOrchestratorThread: async (hostId, id) => {
      mockRestore('thread', id)
      const item = mockState.threads.find((thread) => thread.id === id)
      return item ? { ...item, orchestratorHostId: hostId ?? null } : null
    },
    startRemoteOrchestratorThread: async (_hostId, payload) => {
      emit({ runId: 'remote-demo', threadId: payload.threadId, type: 'item_add', item: { id: uid(), kind: 'user', text: payload.userText, createdAt: Date.now() } })
      simulate(payload.threadId, payload.userText)
      return { runId: 'remote-demo' }
    },
    guideRemoteOrchestratorThread: async (_hostId, payload) => {
      const id = uid()
      emit({ runId: 'remote-demo', threadId: payload.threadId, type: 'item_add', item: { id, kind: 'cue', title: 'Guide', text: payload.userText, state: 'pending', meta: 'queued', createdAt: Date.now() } })
      return { queued: true, queueDepth: 1 }
    },
    cancelRemoteOrchestratorThread: async () => { },
    approveRemoteOrchestrator: async () => { },
    remoteListClis: async (): Promise<CliDefinition[]> => [
      { id: 'shell', name: 'Shell', short: 'zsh', command: null, resumeCommand: null, color: '#8B8B97', icon: 'terminal', builtin: true },
      { id: 'codex', name: 'Codex', short: 'codex', command: 'codex', resumeCommand: 'codex resume --last', color: '#2FB9B0', icon: 'code', builtin: true },
    ],
    remoteTerminalStart: async (hostId, input) => {
      const now = Date.now()
      const session: CliSession = {
        id: uid(),
        projectId: input.projectId,
        cliId: input.cliId,
        title: input.label || (input.cliId === 'shell' ? 'Shell' : input.cliId),
        status: 'idle',
        createdAt: now,
        cwd: mockState.projects.find((project) => project.id === input.projectId)?.cwd || '/Users/example/Projects/remote-project',
        builtin: true,
        hostLocation: 'orchestrator',
        orchestratorHostId: hostId ?? undefined,
        managedBy: 'local',
      }
      remoteTerminalSessions.unshift(session)
      return session
    },
    remoteTerminalWrite: async (_hostId, sessionId) => {
      const session = remoteTerminalSessions.find((item) => item.id === sessionId)
      if (session) session.status = 'busy'
      if (session) setTimeout(() => { session.status = 'idle' }, 150)
    },
    remoteTerminalResize: async () => { },
    remoteTerminalRename: async (_hostId, sessionId, title) => {
      const session = remoteTerminalSessions.find((item) => item.id === sessionId)
      if (session) session.title = title.trim() || session.title
    },
    remoteTerminalKill: async (_hostId, sessionId) => {
      const idx = remoteTerminalSessions.findIndex((session) => session.id === sessionId)
      if (idx >= 0) remoteTerminalSessions.splice(idx, 1)
    },
    remoteTerminalList: async (_hostId, projectId) => remoteTerminalSessions.filter((session) => session.projectId === projectId),
    createRemoteOrchestratorRoom: async (_hostId, projectId, name) => {
      const r: Room = {
        id: uid(), projectId, name: name || 'New room', seats: [],
        rootCwd: null, workspaceCwd: null, scratchpadCwd: null,
        collaborationMode: 'frontier_plans_local_builds', deliberationEffort: 'normal', workerBudget: { maxActive: 2, maxStarts: 4 }, memory: [], memorySummary: '',
        problem: '', state: 'created', events: [], finalText: '', createdAt: Date.now(), updatedAt: Date.now(),
      }
      mockState.mockRooms = [r, ...mockState.mockRooms]
      return r
    },
    updateRemoteOrchestratorRoom: async (_hostId, id, patch) => { const r = mockState.mockRooms.find((x) => x.id === id)!; if (r) Object.assign(r, patch); return r },
    renameRemoteOrchestratorRoom: async (_hostId, id, name) => { const r = mockState.mockRooms.find((x) => x.id === id)!; r.name = name.trim() || 'New room'; return r },
    steerRemoteOrchestratorRoom: async (_hostId, roomId, text) => {
      const room = mockState.mockRooms.find((x) => x.id === roomId)!
      const item = { id: uid(), tick: room.events.length + 1, kind: 'user', seat: 'you', target: 'all', text: text.trim(), createdAt: Date.now() }
      room.events.push(item)
      return item
    },
    mutateRemoteOrchestratorRoomMemory: async (_hostId, roomId, command) => {
      const room = mockState.mockRooms.find((x) => x.id === roomId)!
      if (command.action === 'add') room.memory = [{ id: uid(), kind: command.kind, text: command.text, source: 'manual', createdAt: Date.now(), updatedAt: Date.now() }, ...(room.memory ?? [])]
      else room.memory = (room.memory ?? []).filter((entry) => entry.id !== command.entryId)
      return room
    },
    startRemoteOrchestratorRoom: async (_hostId, roomId, problem) => {
      const room = mockState.mockRooms.find((r) => r.id === roomId)
      if (!room) throw new Error('Room not found')
      room.problem = problem
      room.state = 'queued'
      simulateRoom(roomId, problem)
      return { ok: true, room }
    },
    cancelRemoteOrchestratorRoom: async () => { },
    deleteRemoteOrchestratorRoom: async (_hostId, id) => {
      const room = mockState.mockRooms.find((x) => x.id === id)
      if (!room) return null
      mockArchive('room', id)
      return room
    },
    restoreRemoteOrchestratorRoom: async (_hostId, id) => {
      const room = mockState.mockRooms.find((x) => x.id === id)
      if (!room) return null
      mockRestore('room', id)
      return room
    },
    createRemoteOrchestratorTicket: async (_hostId, t) => { const tk = mt(t.projectId, uid(), t.title || 'Untitled', t.body || '', t.type || 'task', t.status || 'todo', t.priority || 'medium', t.effort || 'medium', t.tags || [], t.parentId || null, t.roomId || null); mockState.mockTickets = [tk, ...mockState.mockTickets]; return tk },
    startRemoteOrchestratorTicketExecution: async (hostId, id, _userText) => {
      const ticket = mockState.mockTickets.find((item) => item.id === id)!
      const worker = mockState.threads.find((item) => item.id === ticket.threadId) ?? thread(ticket.title, '/tmp', 0, 'running', [], 'task', ticket.projectId)
      if (!mockState.threads.some((item) => item.id === worker.id)) mockState.threads.unshift(worker)
      ticket.threadId = worker.id; ticket.status = 'in_progress'; ticket.runState = 'running'
      ticket.execution = { target: 'remote', hostId: hostId ?? null, threadId: worker.id, runId: 'mock-run', timelineBaseline: 0, startedAt: Date.now(), finishedAt: null, checks: [], blockers: [], reviewVerdict: null }
      return { ticket, thread: worker, runId: 'mock-run' }
    },
    updateRemoteOrchestratorTicket: async (_hostId, id, patch) => { const tk = mockState.mockTickets.find((x) => x.id === id)!; if (tk) Object.assign(tk, patch); return tk },
    transitionRemoteOrchestratorTicket: async (hostId, id, command) => {
      const tk = mockState.mockTickets.find((x) => x.id === id)!
      if (command.action === 'begin') {
        tk.threadId = command.threadId; tk.status = 'in_progress'; tk.runState = 'queued'; tk.assignee = 'remote'
        tk.execution = { target: 'remote', hostId: hostId ?? null, threadId: command.threadId, startedAt: Date.now(), finishedAt: null, checks: [], blockers: [], reviewVerdict: null }
      } else if (command.action === 'sync' && tk.runState === 'queued') {
        tk.status = 'in_progress'; tk.runState = 'running'
      } else if (command.action === 'start_failed') {
        tk.status = 'needs_clarification'; tk.runState = null
      } else if (command.action === 'cancel') {
        tk.status = 'needs_clarification'; tk.runState = 'cancelled'
      }
      return tk
    },
    deleteRemoteOrchestratorTicket: async (_hostId, id) => { mockArchive('ticket', id); return mockState.mockTickets.find((item) => item.id === id) ?? null },
    restoreRemoteOrchestratorTicket: async (_hostId, id) => { mockRestore('ticket', id); return mockState.mockTickets.find((item) => item.id === id) ?? null },
  }
}
