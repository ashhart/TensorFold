import type { AgentEvent, ArchiveKind, ArchiveSnapshot, DiffLine, Settings, StartRunPayload, Thread, TimelineItem, Project, Room, RoomEventItem, RoomMemoryEntry, RoomStreamEvent, Ticket, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitStatus, TokenUsageSnapshot, TokenUsageWindow, TokenUsageWindowKey, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, AppUpdateStatus, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSyncResult, EmailThreadResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, Skill, Command, McpServerConfig, McpServerSnapshot, McpSnapshot, McpToolCallResult, McpToolDefinition, ControllerSnapshot, ControllerStatus, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookDeliveryRecord, WebhookEndpoint, WebhookSnapshot, RemoteControlPairingInfo, RemoteControlStatus, RemoteControlWorkSnapshot, SecurityControlSnapshot, SecurityPurgeResult, CliDefinition, CliSession, CliToolStatus } from '../types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DiscordSource, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, GoogleChatSource, MattermostSource, MatrixSource, SlackMessage, SlackListOptions, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from '../types'
import type { Incident, RunExportBundle } from '../types'
import type { TensorFoldStudioApi } from '../ipc'
import './fixtures-core'
import './fixtures-data'
import { mockState, uid, ago } from './state'
import { cancelSimulation, emit, simulate, simulateRoom, thread } from './fixtures-core'
import { autoEmit, mkAuto, mockContextPressure, mockJob, mockTokenUsage, mt, replaceMockJob, ticketEmit, updateEmit } from './fixtures-data'
import { CODEX_CLI_MODELS, CLAUDE_CLI_MODELS } from '../cli-models'
import { filterMockBrowser, filterMockCalendar, filterMockDocuments, filterMockEmail, filterMockSlack, mockAssistantSnapshot, mockBrowserCurrentTabSource, mockBrowserSourceId, mockBrowserSourceRef, mockBrowserSummary, mockCalendarMutationEvent, mockCalendarSourceRef, mockCalendarSummary, mockDocumentSourceId, mockDocumentSourceRef, mockDocumentSummary, mockEmailSourceRef, mockEmailThread, mockModelRouterSnapshot, mockSlackSourceId, mockSlackSourceRef, mockSlackSummary } from './connectors'
import { activeOnly, createMockLocalPrLoop, mockArchive, mockArchiveSnapshot, mockBlockedCommandReason, mockGitHygieneReport, mockLocalMemoryCounts, mockMcpSnapshot, mockMissionInspection, mockMissionSnapshot, mockRemotePairingInfo, mockRemoteStatus, mockRemoteWorkSnapshot, mockRestore, mockSecurityControls, mockWebhookSnapshot, redactedMockSettings, updateMockMission } from './runtime'
import { createMockKnowledgeApi } from './api-knowledge'
import { createMockEngineApi } from './engine'
import { createMockWorktreeApi } from './worktrees'
import { demoApprovalThread, demoRunStateEvents, mockGitDiff, mockGitStatus } from './workspace-fixtures'

function mockDiscordSource(label: string): DiscordSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Discord server', token: '', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function mockMattermostSource(label: string): MattermostSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Mattermost server', baseUrl: '', token: '', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function mockMatrixSource(label: string): MatrixSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Matrix account', baseUrl: 'https://matrix.org', token: '', allowedRooms: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function mockGoogleChatSource(label: string): GoogleChatSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Google Chat space', webhookUrl: '', accessToken: '', baseUrl: '', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function patchSource<T extends { id: string; updatedAt: number }>(sources: T[], id: string, patch: Partial<T>): T[] {
  return sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source)
}

function isMaskedSecret(value: unknown): boolean {
  return value === '••••••••' || value === '[set]'
}

function preserveMaskedSourceSecrets<T extends { id: string }>(incoming: T[], current: T[], keys: string[]): T[] {
  const currentById = new Map(current.map((source) => [source.id, source as Record<string, unknown>]))
  return incoming.map((source) => {
    const existing = currentById.get(source.id)
    if (!existing) return source
    const next = { ...(source as Record<string, unknown>) }
    for (const key of keys) {
      if (isMaskedSecret(next[key])) next[key] = existing[key] ?? ''
    }
    return next as T
  })
}

export function createMockApi(): TensorFoldStudioApi {
  const { mockEngineReadiness, ...engineApi } = createMockEngineApi()
  // Demo-mode WhatsApp status subscribers, so startWhatsAppPairing can push a "not available in
  // the browser" status that clears the renderer's transient "Connecting…" pairing state.
  let demoRunStatesSent = false
  const whatsappStatusSubs = new Set<(payload: { sourceId: string; paired?: boolean; lastError?: string | null }) => void>()
  const remoteTerminalSessions: CliSession[] = []
  return {
    platform: 'browser',
    getSecurityControls: async () => mockSecurityControls(),
    writeClipboard: async (text: string) => {
      try {
        await navigator.clipboard?.writeText(text)
        return { ok: true }
      } catch {
        return { ok: false }
      }
    },
    purgeLocalMemory: async (): Promise<SecurityPurgeResult> => {
      const counts = mockLocalMemoryCounts()
      mockState.threads = []
      mockState.projects = []
      mockState.mockRooms = []
      mockState.mockTickets = []
      mockState.mockJobs = []
      mockState.mockAutomations = []
      mockState.mockConnectorEvents = []
      mockState.mockMissions = []
      mockState.mockProjectMemory = []
      mockState.mockProjectMemoryCandidates = []
      mockState.mockIncidents = []
      mockState.settings = {
        ...mockState.settings,
        whatsapp: {
          ...mockState.settings.whatsapp,
          sources: mockState.settings.whatsapp.sources.map((source) => ({
            ...source,
            paired: false,
            phoneNumber: null,
            credentialRotatedAt: null,
            lastError: 'Linked-device credentials were removed by the local-state purge; pair again.',
          })),
        },
      }
      return {
        ok: true,
        purgedAt: Date.now(),
        counts,
        removedPaths: ['/Users/example/.vontra/archives', '/Users/example/.vontra/rooms', '/Users/example/.vontra/updates', '/Users/example/.vontra/whatsapp-auth'],
        preservedCredentialsAndSettings: ['App settings and configured provider/cloud credentials (linked WhatsApp device credentials are removed)'],
        preservedResiduals: [],
      }
    },
    // Mirror the Electron path: never hand the renderer real keys; treat a masked
    // key on save as "unchanged" so a mask round-trip can't clobber the stored key.
    getSettings: async () => redactedMockSettings(),
    saveSettings: async (s) => {
      const providers = s.providers.map((p) => {
        const cur = mockState.settings.providers.find((x) => x.id === p.id)
        const masked = p.apiKey === '••••••••' || p.apiKey === '[set]'
        const accounts = p.accounts?.map((account) => {
          const curAccount = cur?.accounts?.find((x) => x.id === account.id)
          const accountMasked = account.apiKey === '••••••••' || account.apiKey === '[set]'
          return {
            ...account,
            apiKey: accountMasked ? (curAccount?.apiKey ?? '') : account.apiKey,
            env: Object.fromEntries(Object.entries(account.env ?? {}).map(([key, value]) => [key, value === '••••••••' || value === '[set]' ? (curAccount?.env?.[key] ?? '') : value])),
          }
        })
        return {
          ...p,
          apiKey: masked ? (cur?.apiKey ?? '') : p.apiKey,
          cliEnv: Object.fromEntries(Object.entries(p.cliEnv ?? {}).map(([key, value]) => [key, value === '••••••••' || value === '[set]' ? (cur?.cliEnv?.[key] ?? '') : value])),
          accounts,
        }
      })
      const botControl = {
        ...s.botControl,
        telegram: {
          ...s.botControl.telegram,
          botToken: s.botControl.telegram.botToken === '••••••••' || s.botControl.telegram.botToken === '[set]'
            ? mockState.settings.botControl.telegram.botToken
            : s.botControl.telegram.botToken,
        },
        whatsapp: {
          ...s.botControl.whatsapp,
          accessToken: s.botControl.whatsapp.accessToken === '••••••••' || s.botControl.whatsapp.accessToken === '[set]'
            ? mockState.settings.botControl.whatsapp.accessToken
            : s.botControl.whatsapp.accessToken,
          verifyToken: s.botControl.whatsapp.verifyToken === '••••••••' || s.botControl.whatsapp.verifyToken === '[set]'
            ? mockState.settings.botControl.whatsapp.verifyToken
            : s.botControl.whatsapp.verifyToken,
          appSecret: s.botControl.whatsapp.appSecret === '••••••••' || s.botControl.whatsapp.appSecret === '[set]'
            ? mockState.settings.botControl.whatsapp.appSecret
            : s.botControl.whatsapp.appSecret,
        },
      }
      const voice = {
        ...s.voice,
        apiKey: s.voice.apiKey === '••••••••' || s.voice.apiKey === '[set]'
          ? mockState.settings.voice.apiKey
          : s.voice.apiKey,
        elevenLabsApiKey: s.voice.elevenLabsApiKey === '••••••••' || s.voice.elevenLabsApiKey === '[set]'
          ? mockState.settings.voice.elevenLabsApiKey
          : s.voice.elevenLabsApiKey,
      }
      const remoteControl = {
        ...s.remoteControl,
        authToken: s.remoteControl.authToken === '••••••••' || s.remoteControl.authToken === '[set]'
          ? mockState.settings.remoteControl.authToken
          : s.remoteControl.authToken,
        lastStartedAt: s.remoteControl.enabled ? (s.remoteControl.lastStartedAt ?? Date.now()) : s.remoteControl.lastStartedAt,
        lastError: null,
      }
      const remoteClient = {
        ...s.remoteClient,
        authToken: s.remoteClient.authToken === '••••••••' || s.remoteClient.authToken === '[set]'
          ? mockState.settings.remoteClient.authToken
          : s.remoteClient.authToken,
      }
      const remoteOrchestrators = s.remoteOrchestrators.map((host) => {
        const current = mockState.settings.remoteOrchestrators.find((item) => item.id === host.id)
        return {
          ...host,
          authToken: host.authToken === '••••••••' || host.authToken === '[set]' ? (current?.authToken ?? mockState.settings.remoteClient.authToken ?? '') : host.authToken,
        }
      })
      const email = { ...s.email, sources: preserveMaskedSourceSecrets(s.email.sources, mockState.settings.email.sources, ['password']) }
      const slack = { ...s.slack, sources: preserveMaskedSourceSecrets(s.slack.sources, mockState.settings.slack.sources, ['token']) }
      const discord = { ...s.discord, sources: preserveMaskedSourceSecrets(s.discord.sources, mockState.settings.discord.sources, ['token']) }
      const mattermost = { ...s.mattermost, sources: preserveMaskedSourceSecrets(s.mattermost.sources, mockState.settings.mattermost.sources, ['token']) }
      const matrix = { ...s.matrix, sources: preserveMaskedSourceSecrets(s.matrix.sources, mockState.settings.matrix.sources, ['token']) }
      const googlechat = { ...s.googlechat, sources: preserveMaskedSourceSecrets(s.googlechat.sources, mockState.settings.googlechat.sources, ['webhookUrl', 'accessToken']) }
      Object.assign(mockState.settings, { ...s, providers, botControl, voice, remoteControl, remoteClient, remoteOrchestrators, email, slack, discord, mattermost, matrix, googlechat })
      return redactedMockSettings()
    },
    revokeApprovalRule: async (id) => {
      mockState.settings.approvalRules = mockState.settings.approvalRules.filter((rule) => rule.id !== id)
      return redactedMockSettings()
    },
    getRemoteStatus: async () => mockRemoteStatus(),
    getMobileBridgeStatus: async () => ({ enabled: false, listening: false, url: null, host: null, port: null, certificateSha256: null, certificateExpiresAt: null, interfaces: [], clients: 0, lastError: null }),
    configureMobileBridge: async () => { throw new Error('Enable the private mobile connection in the desktop app.') },
    issueMobilePairing: async () => { throw new Error('Pair a phone from the desktop app.') },
    // Provider rate-limit status (VST-037) — a demo codex snapshot so the picker/doctor
    // show live pace/ETA in the browser demo (session ~60% mid-window, weekly ~35%). Keyed
    // to whichever codex account is active so the picker's per-account lookup hits.
    getProviderStatus: async () => {
      const now = Date.now()
      const codex = mockState.settings.providers.find((p) => p.transport === 'codex-cli')
      const accountId = codex ? (mockState.settings.activeProviderAccountId ?? codex.defaultAccountId ?? codex.accounts?.[0]?.id ?? null) : null
      const key = `${codex?.id ?? 'codex-cli'}:${accountId ?? 'default'}`
      return {
        version: 1 as const,
        statuses: {
          [key]: {
            providerId: codex?.id ?? 'codex-cli',
            accountId,
            raw: {
              primary: { usedPercent: 62, resetsAt: now + 90 * 60_000, windowDurationMins: 300 },
              secondary: { usedPercent: 34, resetsAt: now + 4 * 24 * 60 * 60_000, windowDurationMins: 7 * 24 * 60 },
              planType: 'plus',
              limitName: 'ChatGPT Plus',
            },
            updatedAt: now,
          },
        },
      }
    },
    onProviderStatus: () => () => {},
    onUsageCleared: () => () => {},
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
    cancelRemoteOrchestratorThread: async () => {},
    approveRemoteOrchestrator: async () => {},
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
    remoteTerminalResize: async () => {},
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
    cancelRemoteOrchestratorRoom: async () => {},
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
    getThreads: async () => activeOnly(mockState.threads),
    getArchives: async () => mockArchiveSnapshot(),
    archiveItem: async (kind, id) => mockArchive(kind, id),
    restoreItem: async (kind, id) => mockRestore(kind, id),
    createThread: async (cwd, kind = 'task', projectId = null) => {
      const proj = projectId ? mockState.projects.find((p) => p.id === projectId) : null
      const t = thread(kind === 'chat' ? 'New chat' : 'New task', cwd || proj?.cwd || mockState.settings.defaultCwd || '~', 0, 'idle', [], kind, projectId)
      mockState.threads = [t, ...mockState.threads]
      return t
    },
    setThreadProject: async (id, projectId) => { const t = mockState.threads.find((x) => x.id === id); if (t) { t.projectId = projectId; const proj = projectId ? mockState.projects.find((p) => p.id === projectId) : null; if (proj?.cwd) t.cwd = proj.cwd } },
    deleteThread: async (id) => { mockArchive('thread', id) },
    renameThread: async (id, title) => { const t = mockState.threads.find((x) => x.id === id); if (t) { t.title = title; t.updatedAt = Date.now() } },
    setThreadEngine: async (id, engine) => { const t = mockState.threads.find((x) => x.id === id); if (t) t.engine = engine === 'native' ? undefined : engine },
    ompStatus: async () => ({ available: true, binary: '/usr/local/bin/omp', version: 'omp v18.1.18 (demo)' }),
    configureThread: async (id, providerId, model, providerAccountId = null) => {
      const t = mockState.threads.find((x) => x.id === id)
      if (t) { t.providerId = providerId; t.providerAccountId = providerAccountId; t.model = model }
    },
    patchThread: async (id, patch) => { const t = mockState.threads.find((x) => x.id === id)!; if (t) Object.assign(t, patch); return t },
    setThreadWorkspace: async (id, workspace, cwd) => {
      const t = mockState.threads.find((x) => x.id === id)!
      if (t) { t.workspace = workspace; t.cwd = workspace?.path ?? cwd ?? t.cwd }
      return t
    },
    appendThreadNotice: async (id, text) => { const t = mockState.threads.find((x) => x.id === id); if (t) t.timeline.push({ id: `notice-${t.timeline.length}`, kind: 'notice', text, createdAt: Date.now() } as any) },
    compactThread: async (id) => {
      const t = mockState.threads.find((x) => x.id === id)
      if (!t) throw new Error('Thread not found')
      const item = { id: uid(), kind: 'notice' as const, text: 'Context compacted. Folded earlier messages into a running summary.', createdAt: Date.now() }
      t.timeline = [...t.timeline, item]
      t.compactions = [...(t.compactions ?? []), { id: uid(), at: Date.now(), trigger: 'manual', messagesFolded: 4, tokensBefore: 9200, tokensAfter: 2100, summaryExcerpt: 'Demo context summary.', kindsDropped: ['assistant', 'tool'] }]
      emit({ runId: 'demo', threadId: id, type: 'item_add', item } as AgentEvent)
      return { ok: true, thread: t }
    },
    startRun: async ({ threadId, userText }: StartRunPayload) => {
      emit({ runId: 'demo', threadId, type: 'item_add', item: { id: uid(), kind: 'user', text: userText, createdAt: Date.now() } })
      simulate(threadId, userText)
      return { runId: 'demo' }
    },
    cueRun: async ({ threadId, userText }) => {
      const id = uid()
      emit({ runId: 'demo', threadId, type: 'item_add', item: { id, kind: 'cue', title: 'Guide', text: userText, state: 'pending', meta: 'queued', createdAt: Date.now() } })
      setTimeout(() => emit({ runId: 'demo', threadId, type: 'item_update', id, patch: { state: 'ok', meta: 'applied' } }), 350)
      return { queued: true, queueDepth: 1 }
    },
    cancelRun: async (threadId) => { cancelSimulation(threadId) },
    approve: async (callId) => {
      const threadId = demoApprovalThread(callId)
      if (!threadId) return
      emit({ runId: 'demo-waiting', threadId, type: 'status', status: 'done' })
      emit({ runId: 'demo-waiting', threadId, type: 'done' })
    },
    pickDirectory: async () => '/Users/example/Projects/web-dashboard',
    getTokenUsage: async () => mockTokenUsage(),
    getContextPressure: async () => mockContextPressure(),
    getModelRouterSnapshot: async () => mockModelRouterSnapshot(),
    getAssistantSnapshot: async () => mockAssistantSnapshot(),
    getUpdateStatus: async () => mockState.mockUpdateStatus,
    checkForUpdates: async () => {
      mockState.mockUpdateStatus = { ...mockState.mockUpdateStatus, state: 'checking', checkedAt: Date.now(), error: null }
      updateEmit(mockState.mockUpdateStatus)
      await new Promise((r) => setTimeout(r, 600))
      mockState.mockUpdateStatus = { ...mockState.mockUpdateStatus, state: 'available', checkedAt: Date.now() }
      updateEmit(mockState.mockUpdateStatus)
      return mockState.mockUpdateStatus
    },
    openUpdateDownload: async () => ({ ok: false }),
    downloadAppUpdate: async () => {
      mockState.mockUpdateStatus = { ...mockState.mockUpdateStatus, automaticInstall: { supported: false, reason: 'Updates can be downloaded and installed in the desktop app.' }, error: 'Browser demo cannot download or install app updates.' }
      updateEmit(mockState.mockUpdateStatus)
      return { ok: false }
    },
    cancelAppUpdateDownload: async () => ({ cancelled: false }),
    installAppUpdate: async () => ({ ok: false }),
    onUpdateStatus: (cb) => { mockState.updateListener = cb; return () => { mockState.updateListener = null } },
	    recordConnectorInteraction: async (input: ConnectorInteractionInput) => {
	      const event: ConnectorEvent = {
	        id: uid(), connectorId: input.connectorId, pluginName: input.pluginName || input.connectorId,
	        action: input.action, direction: input.direction || 'internal', classification: 'communication',
	        priority: 'normal', trustLevel: input.direction === 'outbound' ? 'approve' : 'suggest',
	        requiresApproval: input.direction === 'outbound', approvalId: input.direction === 'outbound' ? `approval-${uid()}` : null,
	        approvalStatus: input.direction === 'outbound' ? 'pending' : null,
	        resolvedAt: null,
	        resolvedBy: null,
	        title: String(input.input?.title || input.input?.subject || input.action),
        summary: String(input.input?.summary || input.input?.body || 'Connector interaction recorded.'),
        actor: String(input.target || input.connectorId), threadKey: String(input.input?.threadKey || input.target || input.connectorId),
        payloadKeys: Object.keys(input.input || {}).slice(0, 10), createdAt: Date.now(),
      }
	      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
	      return event
	    },
	    replyToThread: async (_threadKey: string, text: string) => ({ ok: !!text.trim(), error: text.trim() ? undefined : 'Refusing to send an empty reply.' }),
	    requestChannelApproval: async (_threadKey, _request) => ({ ok: false as const, error: 'In-channel approval needs the desktop app.', fallbackToApp: true }),
	    resolveConnectorApproval: async (eventId, decision) => {
	      const event = mockState.mockConnectorEvents.find((item) => item.id === eventId || item.approvalId === eventId)
	      if (!event) throw new Error('Connector approval not found')
	      event.approvalStatus = decision === 'approve' ? 'approved' : 'denied'
	      event.requiresApproval = false
	      event.resolvedAt = Date.now()
	      event.resolvedBy = 'demo'
      if (decision === 'approve' && event.connectorId === 'email' && ['send_requested', 'reply_requested'].includes(event.action)) {
        event.delivery = {
          transport: 'local_outbox',
          status: 'delivered',
          path: `/Users/example/.vontra/email-outbox/${event.id}.eml`,
          deliveredAt: event.resolvedAt,
          error: null,
        }
      }
	      const resolvedEvent: ConnectorEvent = {
	        id: uid(),
	        connectorId: event.connectorId,
	        pluginName: event.pluginName,
	        action: decision === 'approve' ? 'approval_approved' : 'approval_denied',
	        direction: 'internal',
	        classification: 'approval',
	        priority: 'normal',
	        trustLevel: 'auto_logged',
	        requiresApproval: false,
	        approvalId: null,
	        approvalStatus: null,
	        resolvedAt: null,
	        resolvedBy: null,
	        title: decision === 'approve' ? 'Approval allowed' : 'Approval denied',
	        summary: event.title,
	        actor: 'demo',
	        threadKey: event.threadKey,
	        payloadKeys: ['event_id', 'decision'],
	        createdAt: Date.now(),
	      }
	      mockState.mockConnectorEvents = [
	        resolvedEvent,
	        ...mockState.mockConnectorEvents,
	      ].slice(0, 80)
	      return event
	    },
    getControllerStatus: async (): Promise<ControllerStatus> => ({
      platform: 'browser',
      supported: true,
      enabled: mockState.settings.controller.enabled,
      accessibilityTrusted: true,
      canCapture: mockState.settings.controller.enabled,
      message: mockState.settings.controller.enabled ? 'Controller demo capture is ready.' : 'Controller is disabled.',
    }),
    enableController: async (): Promise<ControllerStatus> => {
      mockState.settings.controller = { ...mockState.settings.controller, enabled: true, lastError: null }
      return {
        platform: 'browser',
        supported: true,
        enabled: true,
        accessibilityTrusted: true,
        canCapture: true,
        message: 'Controller demo capture is ready.',
      }
    },
    captureController: async (): Promise<ControllerSnapshot> => {
      const fullText = [
        'Create an intelligent Gomoku game where the AI can explain why it chose each move.',
        'Goal: implement player turns, computer strategy, restart, and review changed files.',
        'Progress: board rendering complete; heuristic scoring needs refinement.',
      ].join('\n')
      const text = mockState.settings.controller.includeAccessibilityText ? fullText.slice(0, mockState.settings.controller.maxTextChars || 4_000) : ''
      const snapshot: ControllerSnapshot = {
        appName: 'Demo Browser',
        bundleId: 'dev.tensorfold.studio.demo',
        windowTitle: 'Gomoku task window',
        capturedAt: Date.now(),
        platform: 'browser',
        text,
        textCharCount: mockState.settings.controller.includeAccessibilityText ? fullText.length : 0,
        truncated: mockState.settings.controller.includeAccessibilityText && text.length < fullText.length,
        source: 'demo',
        warning: mockState.settings.controller.includeAccessibilityText ? null : 'Text capture is off; only app/window metadata was captured.',
      }
      mockState.settings.controller = { ...mockState.settings.controller, enabled: true, lastCapturedAt: snapshot.capturedAt, lastError: null }
      return snapshot
    },
    getMcpSnapshot: async () => mockMcpSnapshot(),
    saveMcpServer: async (input) => {
      const now = Date.now()
      const existing = input.id ? mockState.mockMcpServers.find((server) => server.id === input.id) : null
      const next: McpServerConfig = {
        id: existing?.id || input.id || uid(),
        name: input.name || existing?.name || 'New MCP server',
        transport: 'stdio',
        command: input.command ?? existing?.command ?? '',
        args: input.args ?? existing?.args ?? [],
        cwd: input.cwd ?? existing?.cwd ?? null,
        env: input.env ?? existing?.env ?? {},
        enabled: input.enabled ?? existing?.enabled ?? true,
        autoConnect: input.autoConnect ?? existing?.autoConnect ?? false,
        approvalMode: input.approvalMode ?? existing?.approvalMode ?? 'ask',
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
      }
      mockState.mockMcpServers = existing ? mockState.mockMcpServers.map((server) => server.id === next.id ? next : server) : [next, ...mockState.mockMcpServers]
      return mockMcpSnapshot()
    },
    deleteMcpServer: async (id) => {
      mockState.mockMcpServers = mockState.mockMcpServers.filter((server) => server.id !== id)
      mockState.mockMcpStopped.delete(id)
      return mockMcpSnapshot()
    },
    connectMcpServer: async (id) => {
      mockState.mockMcpServers = mockState.mockMcpServers.map((server) => server.id === id ? { ...server, enabled: true, updatedAt: Date.now() } : server)
      mockState.mockMcpStopped.delete(id)
      return mockMcpSnapshot()
    },
    disconnectMcpServer: async (id) => {
      mockState.mockMcpStopped.add(id)
      mockState.mockMcpServers = mockState.mockMcpServers.map((server) => server.id === id ? { ...server, updatedAt: Date.now() } : server)
      return mockMcpSnapshot()
    },
    refreshMcpServer: async () => mockMcpSnapshot(),
    testMcpTool: async (callName, args): Promise<McpToolCallResult> => {
      if (callName.endsWith('list_allowed_directories')) return { ok: true, output: '/Users/example/Projects', title: 'List allowed directories', meta: 'Filesystem demo MCP' }
      return { ok: true, output: `Demo MCP tool call ${callName}\n${JSON.stringify(args ?? {}, null, 2)}`, title: callName, meta: 'Demo MCP' }
    },
    addEmailSource: async (dir) => {
      const source: EmailSourceConfig = { id: uid(), label: dir.split('/').filter(Boolean).pop() || 'Email', path: dir, kind: 'eml_dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.email = { ...mockState.settings.email, sources: [source, ...mockState.settings.email.sources] }
      return mockState.settings.email
    },
    updateEmailSource: async (id, patch) => {
      mockState.settings.email = { ...mockState.settings.email, sources: mockState.settings.email.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) }
      return mockState.settings.email
    },
    removeEmailSource: async (id) => {
      mockState.settings.email = { ...mockState.settings.email, sources: mockState.settings.email.sources.filter((source) => source.id !== id) }
      return mockState.settings.email
    },
    syncEmail: async (): Promise<EmailSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.email = { ...mockState.settings.email, lastSyncAt: syncedAt, sources: mockState.settings.email.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const event: ConnectorEvent = { id: uid(), connectorId: 'email', pluginName: 'Email', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'high', trustLevel: 'suggest', requiresApproval: false, approvalId: null, title: 'Demo email sync completed', summary: 'Local email source scanned and recent messages are available to Assistant.', actor: 'Email', threadKey: `email:sync:${syncedAt}`, payloadKeys: ['source', 'message_count'], createdAt: syncedAt }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events: [event], messages: mockState.mockEmailMessages.length, sources: mockState.settings.email.sources, syncedAt }
    },
    listEmail: async (opts) => filterMockEmail(opts),
	    readEmail: async (id) => {
	      const message = mockState.mockEmailMessages.find((m) => m.id === id.replace(/^email:/, ''))
	      if (!message) throw new Error('Email message not found')
	      return message
	    },
	    readEmailThread: async (id): Promise<EmailThreadResult> => {
	      return mockEmailThread(id)
	    },
    summarizeEmail: async (opts) => {
      const messages = filterMockEmail(opts)
      const recent = messages.slice(0, 3)
      const sources = recent.map((message) => ({
        id: `email:${message.sourceId}:${message.id}`,
        connectorId: 'email' as const,
        sourceId: message.sourceId,
        itemId: message.id,
        title: message.subject || '(no subject)',
        path: message.path,
        excerpt: message.snippet,
        createdAt: message.date,
      }))
      return {
        summary: `${messages.length} demo messages, ${messages.filter((m) => m.unread).length} unread. Recent: ${recent.map((m) => `[email:${m.sourceId}:${m.id}] ${m.subject}`).join(', ')}.\n\nSources:\n${sources.length ? sources.map((source) => `- ${source.id} | ${source.title}`).join('\n') : 'none.'}`,
        sources,
      }
    },
    addSlackSource: async (dir) => {
      const source: SlackSourceConfig = { id: uid(), label: dir.split('/').filter(Boolean).pop() || 'Slack export', path: dir, kind: 'export_dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.slack = { ...mockState.settings.slack, sources: [source, ...mockState.settings.slack.sources] }
      return mockState.settings.slack
    },
    updateSlackSource: async (id, patch) => {
      mockState.settings.slack = { ...mockState.settings.slack, sources: mockState.settings.slack.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, kind: 'export_dir', updatedAt: Date.now() } : source) }
      return mockState.settings.slack
    },
    removeSlackSource: async (id) => {
      mockState.settings.slack = { ...mockState.settings.slack, sources: mockState.settings.slack.sources.filter((source) => source.id !== id) }
      return mockState.settings.slack
    },
    syncSlack: async (): Promise<SlackSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.slack = { ...mockState.settings.slack, lastSyncAt: syncedAt, sources: mockState.settings.slack.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const messages = filterMockSlack({ limit: mockState.settings.slack.maxMessages })
      const events: ConnectorEvent[] = messages.slice(0, 6).map((message) => ({ id: uid(), connectorId: 'slack', pluginName: 'Slack', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: `#${message.channelName} ${message.senderName || 'Unknown'}`, summary: message.snippet, actor: `#${message.channelName}`, threadKey: mockSlackSourceId(message), payloadKeys: ['channel', 'message', 'source'], sources: [mockSlackSourceRef(message)], createdAt: syncedAt }))
      mockState.mockConnectorEvents = [...events, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events, messages: messages.length, sources: mockState.settings.slack.sources, syncedAt }
    },
    listSlack: async (opts) => filterMockSlack(opts),
    readSlack: async (id) => {
      const target = id.replace(/^slack:[^:]+:/, '')
      const message = mockState.mockSlackMessages.find((item) => item.id === target || mockSlackSourceId(item) === id)
      if (!message) throw new Error('Slack message not found')
      return message
    },
    summarizeSlack: async (opts) => mockSlackSummary(opts),
    verifySlack: async () => ({ ok: false, error: 'Slack live API is unavailable in browser demo mode.' }),
    listSlackChannels: async () => [],
    sendSlack: async () => { throw new Error('Slack send is unavailable in browser demo mode.') },
    addDiscordSource: async (label) => {
      mockState.settings.discord = { ...mockState.settings.discord, sources: [...mockState.settings.discord.sources, mockDiscordSource(label)] }
      return redactedMockSettings().discord
    },
    removeDiscordSource: async (id) => {
      mockState.settings.discord = { ...mockState.settings.discord, sources: mockState.settings.discord.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().discord
    },
    verifyDiscord: async () => ({ ok: false, error: 'Discord live API is unavailable in browser demo mode.' }),
    listDiscordChannels: async () => [],
    syncDiscord: async () => ({ messages: 0, events: 0 }),
    sendDiscord: async () => { throw new Error('Discord send is unavailable in browser demo mode.') },
    addMattermostSource: async (label) => {
      mockState.settings.mattermost = { ...mockState.settings.mattermost, sources: [...mockState.settings.mattermost.sources, mockMattermostSource(label)] }
      return redactedMockSettings().mattermost
    },
    updateMattermostSource: async (id, patch) => {
      mockState.settings.mattermost = { ...mockState.settings.mattermost, sources: patchSource(mockState.settings.mattermost.sources, id, patch) }
      return redactedMockSettings().mattermost
    },
    removeMattermostSource: async (id) => {
      mockState.settings.mattermost = { ...mockState.settings.mattermost, sources: mockState.settings.mattermost.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().mattermost
    },
    verifyMattermost: async () => ({ ok: false, error: 'Mattermost live API is unavailable in browser demo mode.' }),
    listMattermostChannels: async () => [],
    syncMattermost: async () => ({ messages: 0, events: 0 }),
    sendMattermost: async () => { throw new Error('Mattermost send is unavailable in browser demo mode.') },
    addMatrixSource: async (label) => {
      mockState.settings.matrix = { ...mockState.settings.matrix, sources: [...mockState.settings.matrix.sources, mockMatrixSource(label)] }
      return redactedMockSettings().matrix
    },
    updateMatrixSource: async (id, patch) => {
      mockState.settings.matrix = { ...mockState.settings.matrix, sources: patchSource(mockState.settings.matrix.sources, id, patch) }
      return redactedMockSettings().matrix
    },
    removeMatrixSource: async (id) => {
      mockState.settings.matrix = { ...mockState.settings.matrix, sources: mockState.settings.matrix.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().matrix
    },
    verifyMatrix: async () => ({ ok: false, error: 'Matrix live API is unavailable in browser demo mode.' }),
    listMatrixChannels: async () => [],
    syncMatrix: async () => ({ messages: 0, events: 0 }),
    sendMatrix: async () => { throw new Error('Matrix send is unavailable in browser demo mode.') },
    addGoogleChatSource: async (label) => {
      mockState.settings.googlechat = { ...mockState.settings.googlechat, sources: [...mockState.settings.googlechat.sources, mockGoogleChatSource(label)] }
      return redactedMockSettings().googlechat
    },
    updateGoogleChatSource: async (id, patch) => {
      mockState.settings.googlechat = { ...mockState.settings.googlechat, sources: patchSource(mockState.settings.googlechat.sources, id, patch) }
      return redactedMockSettings().googlechat
    },
    removeGoogleChatSource: async (id) => {
      mockState.settings.googlechat = { ...mockState.settings.googlechat, sources: mockState.settings.googlechat.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().googlechat
    },
    verifyGoogleChat: async () => ({ ok: false, mode: 'webhook' as const, error: 'Google Chat live API is unavailable in browser demo mode.' }),
    listGoogleChatChannels: async () => [],
    syncGoogleChat: async () => ({ messages: 0, events: 0 }),
    sendGoogleChat: async () => { throw new Error('Google Chat send is unavailable in browser demo mode.') },
    sendIMessage: async () => { throw new Error('iMessage send is unavailable in browser demo mode.') },
	    composeEmail: async (input: EmailDraftInput) => {
      const wantsSend = input.mode === 'send'
      const wantsReply = Boolean(input.inReplyTo || input.threadKey)
      const action = wantsSend ? wantsReply ? 'reply_requested' : 'send_requested' : wantsReply ? 'reply_draft_created' : 'draft_created'
      const thread = wantsReply ? mockEmailThread(input.threadKey || input.inReplyTo || '') : null
      const sources = thread?.messages.map(mockEmailSourceRef)
	      const event: ConnectorEvent = { id: uid(), connectorId: 'email', pluginName: 'Email', action, direction: 'outbound', classification: 'approval', priority: wantsSend ? 'high' : 'normal', trustLevel: wantsSend ? 'approve' : 'suggest', requiresApproval: wantsSend, approvalId: wantsSend ? `approval-${uid()}` : null, approvalStatus: wantsSend ? 'pending' : null, resolvedAt: null, resolvedBy: null, title: input.subject, summary: `${wantsSend ? wantsReply ? 'Reply send request' : 'Send request' : wantsReply ? 'Reply draft' : 'Draft'} to ${input.to}: ${input.body.slice(0, 180)}`, actor: input.to, threadKey: thread ? `email:${thread.root.id}` : input.threadKey || `email:draft:${uid()}`, payloadKeys: ['to', 'subject', 'body', ...(input.threadKey ? ['thread_key'] : []), ...(input.inReplyTo ? ['in_reply_to'] : [])], payload: { mode: input.mode ?? 'draft', to: input.to, subject: input.subject, body: input.body, ...(input.threadKey ? { threadKey: input.threadKey } : {}), ...(input.inReplyTo ? { inReplyTo: input.inReplyTo } : {}) }, sources, createdAt: Date.now() }
	      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
	      return event
	    },
    resolveEmailThread: async (input: EmailResolveInput) => {
      const thread = mockEmailThread(input.id)
      const event: ConnectorEvent = { id: uid(), connectorId: 'email', pluginName: 'Email', action: 'thread_resolved', direction: 'internal', classification: 'communication', priority: 'normal', trustLevel: 'auto_logged', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: Date.now(), resolvedBy: 'user', title: thread.root.subject || 'Email thread resolved', summary: `${input.note || 'Marked resolved locally.'} ${thread.summary}`, actor: thread.root.from || 'Email', threadKey: `email:${thread.root.id}`, payloadKeys: ['message_id', ...(input.note ? ['note'] : [])], sources: thread.messages.map(mockEmailSourceRef), createdAt: Date.now() }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    addCalendarSource: async (target) => {
      const source: CalendarSourceConfig = { id: uid(), label: target.split('/').filter(Boolean).pop() || 'Calendar', path: target, kind: target.toLowerCase().endsWith('.ics') ? 'ics_file' : 'ics_dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.calendar = { ...mockState.settings.calendar, sources: [source, ...mockState.settings.calendar.sources] }
      return mockState.settings.calendar
    },
    updateCalendarSource: async (id, patch) => {
      mockState.settings.calendar = { ...mockState.settings.calendar, sources: mockState.settings.calendar.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) }
      return mockState.settings.calendar
    },
    removeCalendarSource: async (id) => {
      mockState.settings.calendar = { ...mockState.settings.calendar, sources: mockState.settings.calendar.sources.filter((source) => source.id !== id) }
      return mockState.settings.calendar
    },
    syncCalendar: async (): Promise<CalendarSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.calendar = { ...mockState.settings.calendar, lastSyncAt: syncedAt, sources: mockState.settings.calendar.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const event: ConnectorEvent = { id: uid(), connectorId: 'calendar', pluginName: 'Calendar', action: 'event_seen', direction: 'inbound', classification: 'schedule', priority: 'normal', trustLevel: 'notify', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: 'Demo calendar sync completed', summary: 'Local .ics source scanned and calendar events are available to Assistant.', actor: 'Calendar', threadKey: `calendar:sync:${syncedAt}`, payloadKeys: ['source', 'event_count'], createdAt: syncedAt }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events: filterMockCalendar({ limit: mockState.settings.calendar.maxEvents }), sources: mockState.settings.calendar.sources, syncedAt }
    },
    listCalendar: async (opts) => filterMockCalendar(opts),
    readCalendar: async (id) => {
      const target = id.replace(/^calendar:[^:]+:/, '')
      const event = mockState.mockCalendarEvents.find((item) => item.id === target)
      if (!event) throw new Error('Calendar event not found')
      return event
    },
    summarizeCalendar: async (opts) => mockCalendarSummary(opts),
    createCalendarEvent: async (input) => {
      const enabledSource = input.sourceId
        ? mockState.settings.calendar.sources.find((source) => source.id === input.sourceId && source.enabled)
        : mockState.settings.calendar.sources.find((source) => source.enabled)
      if (!enabledSource) throw new Error('No enabled calendar source is available for the create request.')
      const event = mockCalendarMutationEvent({ ...input, mode: 'create', sourceId: enabledSource.id })
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    updateCalendarEvent: async (input) => {
      const target = input.id.replace(/^calendar:[^:]+:/, '')
      const existing = mockState.mockCalendarEvents.find((item) => item.id === target)
      if (!existing) throw new Error('Calendar event not found')
      const event = mockCalendarMutationEvent({ ...input, mode: 'update', sourceId: existing.sourceId }, existing)
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    addDocumentSource: async (target) => {
      const source: DocumentSourceConfig = { id: uid(), label: target.split('/').filter(Boolean).pop() || 'Documents', path: target, kind: /\.(md|mdx|txt|rst|csv|tsv|xlsx|json|jsonl|ya?ml|toml|xml|html?|css|jsx?|tsx?|py|rb|go|rs|java|c|cpp|h|hpp|pdf)$/i.test(target) ? 'file' : 'dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.documents = { ...mockState.settings.documents, sources: [source, ...mockState.settings.documents.sources] }
      return mockState.settings.documents
    },
    updateDocumentSource: async (id, patch) => {
      mockState.settings.documents = { ...mockState.settings.documents, sources: mockState.settings.documents.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) }
      return mockState.settings.documents
    },
    removeDocumentSource: async (id) => {
      mockState.settings.documents = { ...mockState.settings.documents, sources: mockState.settings.documents.sources.filter((source) => source.id !== id) }
      return mockState.settings.documents
    },
    syncDocuments: async (): Promise<DocumentSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.documents = { ...mockState.settings.documents, lastSyncAt: syncedAt, sources: mockState.settings.documents.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const docs = filterMockDocuments({ limit: mockState.settings.documents.maxDocuments })
      const events: ConnectorEvent[] = docs.map((doc) => ({ id: uid(), connectorId: 'docs', pluginName: 'Documents', action: 'document_indexed', direction: 'inbound', classification: 'document', priority: 'normal', trustLevel: 'suggest', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: doc.title || doc.relativePath, summary: `${doc.relativePath}: ${doc.excerpt}`, actor: doc.sourceLabel, threadKey: mockDocumentSourceId(doc), payloadKeys: ['path', 'title', 'excerpt'], sources: [mockDocumentSourceRef(doc)], createdAt: syncedAt }))
      mockState.mockConnectorEvents = [...events, ...mockState.mockConnectorEvents].slice(0, 80)
      return { documents: docs, sources: mockState.settings.documents.sources, syncedAt }
    },
    listDocuments: async (opts) => filterMockDocuments(opts),
    readDocument: async (id) => {
      const target = id.replace(/^docs:[^:]+:/, '')
      const doc = mockState.mockDocuments.find((item) => item.id === target || item.relativePath === id || item.path === id)
      if (!doc) throw new Error('Document not found')
      return doc
    },
    summarizeDocuments: async (opts) => mockDocumentSummary(opts),
    addBrowserCurrentTabSource: async (input): Promise<BrowserConnectorSettings> => {
      const parsed = new URL(String(input.url || '').trim())
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Browser source URLs must use http or https.')
      const { source, item } = mockBrowserCurrentTabSource(input)
      mockState.settings.browser = { ...mockState.settings.browser, sources: [source, ...mockState.settings.browser.sources] }
      mockState.mockBrowserItems = [item, ...mockState.mockBrowserItems]
      return mockState.settings.browser
    },
    addBrowserHistorySource: async (target): Promise<BrowserConnectorSettings> => {
      const now = Date.now()
      const source: BrowserSourceConfig = { id: uid(), label: target.split('/').filter(Boolean).pop() || 'Browser history export', kind: 'history_export', enabled: true, createdAt: now, updatedAt: now, lastSyncAt: null, lastError: null, path: target }
      mockState.settings.browser = { ...mockState.settings.browser, sources: [source, ...mockState.settings.browser.sources] }
      return mockState.settings.browser
    },
    updateBrowserSource: async (id, patch): Promise<BrowserConnectorSettings> => {
      mockState.settings.browser = { ...mockState.settings.browser, sources: mockState.settings.browser.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, kind: source.kind, updatedAt: Date.now() } : source) }
      return mockState.settings.browser
    },
    removeBrowserSource: async (id): Promise<BrowserConnectorSettings> => {
      mockState.settings.browser = { ...mockState.settings.browser, sources: mockState.settings.browser.sources.filter((source) => source.id !== id) }
      return mockState.settings.browser
    },
    syncBrowser: async (): Promise<BrowserSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.browser = { ...mockState.settings.browser, lastSyncAt: syncedAt, sources: mockState.settings.browser.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const items = filterMockBrowser({ limit: mockState.settings.browser.maxItems })
      const events: ConnectorEvent[] = items.map((item) => ({ id: uid(), connectorId: 'browser', pluginName: 'Browser', action: 'browser_source_seen', direction: 'inbound', classification: 'document', priority: 'low', trustLevel: 'notify', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: item.title || item.url, summary: item.excerpt || item.url, actor: item.sourceLabel, threadKey: mockBrowserSourceId(item), payloadKeys: ['source', 'url', 'title'], sources: [mockBrowserSourceRef(item)], createdAt: item.capturedAt ?? syncedAt }))
      mockState.mockConnectorEvents = [...events, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events, items, sources: mockState.settings.browser.sources, syncedAt }
    },
    listBrowser: async (opts) => filterMockBrowser(opts),
    readBrowser: async (id) => {
      const target = id.replace(/^browser:[^:]+:/, '')
      const item = mockState.mockBrowserItems.find((candidate) => candidate.id === target || candidate.url === id || mockBrowserSourceId(candidate) === id)
      if (!item) throw new Error('Browser source item not found')
      return item
    },
    summarizeBrowser: async (opts) => mockBrowserSummary(opts),
    // iMessage is a local-only connector; browser demo mode reports it unavailable.
    imessageAccess: async () => ({ state: 'no_macos', detail: 'iMessage local bridge is unavailable in browser demo mode.' }),
    syncImessage: async () => ({ events: [], messages: 0, syncedAt: Date.now() }),
    listImessage: async () => [],
    readImessage: async () => null,
    openFullDiskAccess: async () => {},
    addWhatsAppSource: async () => ({ sources: [], lastSyncAt: null }),
    removeWhatsAppSource: async () => ({ sources: [], lastSyncAt: null }),
    // Browser demo mode can't run the WhatsApp bridge, so pairing reports a clear status back to
    // any onWhatsAppStatus subscriber instead of leaving the UI stuck on "Connecting…" forever.
    startWhatsAppPairing: async (id: string) => {
      setTimeout(() => { for (const cb of whatsappStatusSubs) cb({ sourceId: id, paired: false, lastError: 'WhatsApp pairing is unavailable in browser demo mode.' }) }, 300)
      return { started: false }
    },
    disconnectWhatsApp: async () => ({ ok: true }),
    onWhatsAppQr: () => () => {},
    onWhatsAppStatus: (cb) => { whatsappStatusSubs.add(cb); return () => whatsappStatusSubs.delete(cb) },
    pollTelegram: async () => {
      mockState.settings.botControl.telegram = { ...mockState.settings.botControl.telegram, lastPollAt: Date.now(), lastError: null }
      return { ok: true, processed: 0, skipped: 0, error: null, lastUpdateId: mockState.settings.botControl.telegram.lastUpdateId }
    },
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
    fetchModels: async (provider) => {
      await new Promise((r) => setTimeout(r, 700))
      if (provider.transport === 'codex-cli') return provider.models?.length ? provider.models : CODEX_CLI_MODELS
      if (provider.transport === 'claude-cli') return provider.models?.length ? provider.models : CLAUDE_CLI_MODELS
      const baseUrl = provider.baseUrl
      if (provider.source === 'openrouter' || baseUrl.includes('openrouter.ai')) return ['openrouter/free', 'openrouter/auto', 'openrouter/fusion', 'z-ai/glm-5.2', 'nvidia/nemotron-3-ultra-550b-a55b:free', 'nvidia/nemotron-3-super-120b-a12b:free', 'qwen/qwen3-coder:free', 'openai/gpt-oss-120b:free']
      if (provider.source === 'ai-gateway' || baseUrl.includes('ai-gateway.vercel.sh')) return ['openai/gpt-5.4', 'openai/gpt-5.5', 'openai/gpt-4.1', 'openai/o4-mini', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-opus-4.1', 'xai/grok-4.1-fast-reasoning', 'google/gemini-2.5-pro']
      if (baseUrl.includes('9000')) return ['Qwen3-Coder-Next-MLX-8bit', 'Qwen2.5-Coder-14B-Instruct-MLX-8bit']
      if (baseUrl.includes('11434')) return ['qwen2.5-coder:14b', 'llama3.1:8b', 'deepseek-r1:7b', 'mistral-nemo']
      if (baseUrl.includes('1234')) return ['local-model', 'qwen2.5-coder-7b-instruct']
      if (baseUrl.includes('openai')) return ['gpt-4o', 'gpt-4o-mini', 'o3', 'o4-mini']
      if (baseUrl.includes('z.ai')) return ['glm-5.2', 'glm-5.1', 'glm-4.6', 'glm-4-flash', 'glm-4-air']
      return ['model-a', 'model-b']
    },
    getModelReadiness: async (input) => {
      await new Promise((r) => setTimeout(r, 120))
      const providerId = input?.providerId ?? mockState.settings.activeProviderId
      const provider = mockState.settings.providers.find((p) => p.id === providerId)
      if (!provider) return { ready: false, reason: 'no-provider', message: 'Please go to Settings and connect a cloud or local model.' }
      if (provider.source === 'tensorfold') return mockEngineReadiness(providerId === mockState.settings.activeProviderId ? mockState.settings.activeModel ?? null : null)
      const name = provider.name || 'This provider'
      // Browser demo can't run local CLIs, but the default experience should still look like a
      // configured harness. Real desktop readiness uses the Electron CLI auth probe.
      if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') {
        return { ready: true, reason: 'ok', message: `${name} demo-ready.` }
      }
      const isLocal = provider.kind === 'local' || ['ollama', 'lmstudio', 'omlx'].includes(provider.source || '')
      if (!provider.baseUrl?.trim()) return { ready: false, reason: 'needs-endpoint', message: 'Please go to Settings and connect a cloud or local model.' }
      if (!isLocal && !provider.apiKey?.trim()) return { ready: true, reason: 'ok', message: `${name} demo-ready.` }
      return { ready: true, reason: 'ok', message: '' }
    },
    loginCliProvider: async () => {
      await new Promise((r) => setTimeout(r, 300))
      return { ok: false, loggedIn: false, message: 'CLI sign-in runs in the TensorFold desktop app.' }
    },
    cancelCliLogin: async () => ({ cancelled: true }),
    checkCliTransport: async () => ({ loggedIn: false, binaryPresent: false, message: 'CLI auth check unavailable in browser demo mode.' }),
    ...createMockWorktreeApi(),
    checkoutBranch: async () => ({ ok: true }),
    startLoop: async (input) => ({ id: 'mock-loop', source: input.source, surfaceId: input.surfaceId, goal: input.goal, mode: input.mode, status: 'running', cwd: '/Users/example/project', branch: 'vontra/loop-mock', phase: 'dev', cycle: 1, maxCycles: input.mode === 'ship' ? 7 : 5, checkpoints: 0, lastVerdict: null, blockers: [], log: ['Loop started (mock)'], roleThreadIds: [], createdAt: Date.now(), updatedAt: Date.now() }),
    listLoops: async () => [],
    stopLoop: async () => ({ stopped: true }),
    onLoopUpdate: () => () => {},
    // Browser demo mode has no live orchestration broker; the Command Centre still
    // snapshot-polls as the fallback, so a no-op subscription is enough not to crash.
    onOrchestrationEvent: () => () => {},
    discoverLocalProvider: async (kind) => {
      await new Promise((r) => setTimeout(r, 500))
      if (kind === 'ollama') return {
        kind,
        found: true,
        name: 'Ollama (local)',
        source: 'ollama',
        baseUrl: 'http://127.0.0.1:11434/v1',
        apiKey: 'ollama',
        models: ['qwen2.5-coder:14b', 'llama3.1:8b', 'deepseek-r1:7b'],
        defaultModel: 'qwen2.5-coder:14b',
        configPath: null,
        message: 'Demo Ollama server discovered.',
      }
      if (kind === 'lmstudio') return {
        kind,
        found: true,
        name: 'LM Studio (local)',
        source: 'lmstudio',
        baseUrl: 'http://127.0.0.1:1234/v1',
        apiKey: 'lm-studio',
        models: ['local-model', 'qwen2.5-coder-7b-instruct'],
        defaultModel: 'local-model',
        configPath: null,
        message: 'Demo LM Studio server discovered.',
      }
      return {
        kind,
        found: true,
        name: 'oMLX (local)',
        source: 'omlx',
        baseUrl: 'http://127.0.0.1:9000/v1',
        apiKey: 'demo-key',
        models: ['Qwen3-Coder-Next-MLX-8bit', 'Qwen2.5-Coder-14B-Instruct-MLX-8bit'],
        defaultModel: 'Qwen3-Coder-Next-MLX-8bit',
        configPath: '~/.omlx/settings.json',
        message: 'Demo oMLX config discovered.',
      }
    },
    discoverOmlxProvider: async () => ({
      found: true,
      baseUrl: 'http://127.0.0.1:9000/v1',
      apiKey: 'demo-key',
      models: ['Qwen3-Coder-Next-MLX-8bit', 'Qwen2.5-Coder-14B-Instruct-MLX-8bit'],
      defaultModel: 'Qwen3-Coder-Next-MLX-8bit',
      configPath: '~/.omlx/settings.json',
      message: 'Demo oMLX config discovered.',
    }),
    // Browser demo mode can't reach ElevenLabs or run a local TTS engine — there's no main
    // process to perform the HTTP call. Signal a graceful outcome so speakText() falls back to
    // the Web Speech voice instead of erroring.
    synthesizeVoice: async () => ({ ok: false, error: 'Provider speech synthesis runs in the desktop app. Falling back to the system voice.', fallback: true }),
    synthesizeLocalVoice: async () => ({ ok: false, error: 'Local TTS runs in the desktop app. Falling back to the system voice.', fallback: true }),
    transcribeVoice: async () => ({ ok: false, error: 'API speech transcription runs in the desktop app.' }),
    discoverLocalTts: async (baseUrl) => ({ found: false, baseUrl: baseUrl || 'http://127.0.0.1:8080', voices: [], message: 'Local TTS discovery runs in the desktop app.' }),
    getBranch: async () => 'feat/dark-mode',
    inspectGitRepository: async (cwd) => ({
      cwd,
      exists: !cwd.includes('new-folder-without-git'),
      insideWorkTree: !cwd.includes('new-folder-without-git'),
      topLevel: cwd.includes('new-folder-without-git') ? null : cwd,
      branch: cwd.includes('new-folder-without-git') ? null : 'main',
      hasCommits: !cwd.includes('new-folder-without-git'),
      message: cwd.includes('new-folder-without-git') ? 'This folder is not initialized as a Git repository.' : undefined,
    }),
    initGitRepository: async (cwd) => ({ cwd, exists: true, insideWorkTree: true, topLevel: cwd, branch: 'main', hasCommits: true }),
    scanGitHygiene: async (cwd, message = ''): Promise<GitHygieneReport> => mockGitHygieneReport(cwd, message),
    sanitizeGitHygiene: async (_cwd, findings): Promise<GitHygieneSanitizeResult> => ({
      ok: true,
      changedFiles: (() => { mockState.mockGitHygieneSanitized = true; return [...new Set(findings.filter((f) => f.autoFixable && f.file).map((f) => f.file!))] })(),
      skippedFindings: findings.filter((f) => !f.autoFixable).map((f) => f.id),
    }),
    listFiles: async () => [
      'package.json', 'tsconfig.json', 'vite.config.ts', 'README.md', 'index.html',
      'src/App.tsx', 'src/main.tsx', 'src/index.css',
      'src/components/Sidebar.tsx', 'src/components/Header.tsx', 'src/components/ThemeToggle.tsx', 'src/components/Composer.tsx',
      'src/lib/store.ts', 'src/lib/theme.ts', 'src/lib/api.ts',
      'src/hooks/useTheme.ts', 'src/pages/Dashboard.tsx', 'src/pages/Settings.tsx',
      'src/styles/tokens.css', 'tests/App.test.tsx', 'tests/theme.test.ts',
    ],
    resolveInstructions: async (cwd, providerId = null, workspace = null) => {
      if (workspace?.kind === 'ssh') return { rootPath: workspace.path || cwd, providerId, nodes: [], cycles: [], resolvedContent: '', tokenEstimate: 0 }
      const shared = '# AGENTS.md (demo)\nProject conventions: use the design tokens, keep PRs small.'
      const overlay = 'Provider overlay: prefer concise diffs and cite file:line.'
      const resolvedContent = `<!-- shared: AGENTS.md -->\n${shared}\n\n<!-- imported: docs/architecture.md -->\nThe app is Electron + React; main process owns tools.${providerId ? `\n\n<!-- provider: .vontra/${providerId}.md -->\n${overlay}` : ''}`
      return {
        rootPath: cwd || '~/demo', providerId,
        nodes: [
          { id: '1', kind: 'shared', path: 'AGENTS.md', content: shared, imports: ['docs/architecture.md'], ignored: false, tokenEstimate: 28 },
          { id: '2', kind: 'imported', path: 'docs/architecture.md', content: 'The app is Electron + React; main process owns tools.', imports: [], ignored: false, tokenEstimate: 18 },
          ...(providerId ? [{ id: '3', kind: 'provider' as const, path: `.vontra/${providerId}.md`, content: overlay, imports: [], ignored: false, tokenEstimate: 14 }] : []),
        ],
        cycles: [],
        resolvedContent,
        tokenEstimate: Math.round(resolvedContent.length / 4),
      }
    },
    gitStatus: async (cwd): Promise<GitStatus> => mockGitStatus(cwd),
    gitDiff: async (cwd, paths): Promise<GitDiffFile[]> => mockGitDiff(cwd, paths),
    revertFiles: async () => {},
    checkpointList: async () => [],
    checkpointUndoLast: async () => ({ ok: false, error: 'Checkpoints require the desktop app.' }),
    checkpointDiffLast: async () => ({ ok: false, error: 'Checkpoints require the desktop app.' }),
    // Turn checkpoints are SIMULATED in the browser demo (like session import): one demo
    // checkpoint per user message, revert truncates the mock timeline, fork clones it.
    checkpointTurns: async ({ threadId }) => {
      const t = mockState.threads.find((x) => x.id === threadId)
      if (!t) return []
      const users = t.timeline.filter((i) => i.kind === 'user')
      return users.map((item, idx) => ({
        threadId,
        turn: idx + 1,
        createdAt: Date.now() - (users.length - idx) * 120_000,
        fileRef: t.kind === 'chat' ? null : `refs/vontra/checkpoints/${threadId}/turn/${idx + 1}`,
        fileSha: t.kind === 'chat' ? null : 'demo1234',
        conversationId: `demo-cvk-${threadId}-${idx + 1}`,
        fingerprint: `demo-fp-${idx + 1}`,
        messageCount: Math.max(0, t.timeline.indexOf(item)),
        branch: 'main',
        planId: null,
        label: `turn ${idx + 1}`,
      }))
    },
    checkpointRevertTurn: async ({ threadId, turn }) => {
      const t = mockState.threads.find((x) => x.id === threadId)
      if (!t) return { ok: false, turn, error: 'Thread not found.' }
      let seen = 0
      const cut = t.timeline.findIndex((i) => i.kind === 'user' && ++seen === turn)
      if (cut < 0) return { ok: false, turn, error: `No demo checkpoint for turn ${turn}.` }
      t.timeline = t.timeline.slice(0, cut)
      t.updatedAt = Date.now()
      return { ok: true, turn }
    },
    checkpointForkTurn: async ({ threadId, turn, worktree }) => {
      const src = mockState.threads.find((x) => x.id === threadId)
      if (!src) return { ok: false, error: 'Thread not found.' }
      let seen = 0
      const cut = src.timeline.findIndex((i) => i.kind === 'user' && ++seen === turn)
      if (cut < 0) return { ok: false, error: `No demo checkpoint for turn ${turn}.` }
      const demoWorktree = worktree ? `~/.vontra/worktrees/${threadId.slice(0, 8)}-turn${turn}-fork-1` : null
      const t: Thread = {
        ...src,
        id: uid(),
        title: `Fork of ${src.title}`,
        ...(demoWorktree ? { cwd: demoWorktree } : {}),
        apiMessages: [],
        timeline: [
          ...src.timeline.slice(0, cut),
          { id: uid(), kind: 'notice', text: `Forked from “${src.title}” at turn ${turn} (branch fork-1). This thread continues independently — the original is untouched.${demoWorktree ? `\nThis fork lives in its own worktree: ${demoWorktree} (simulated in the browser demo).` : ''}`, createdAt: Date.now() },
        ],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        status: 'idle',
      }
      mockState.threads = [t, ...mockState.threads]
      return { ok: true, branch: 'fork-1', fileRef: null, thread: t, worktreePath: demoWorktree, worktreeError: null }
    },
    // Cross-engine handoff (VST-036) — simulated: a new thread on the target engine seeded
    // by a handoff notice (the desktop app also auto-runs the flattened transcript).
    // Demo-mode stand-in for the host's planning-guard preview: terms and emails only.
    previewWithheld: async ({ text, terms }) => {
      let total = 0
      let out = String(text ?? '').replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, () => { total++; return `[EMAIL_${total}]` })
      for (const [i, term] of (terms ?? []).filter((t) => t.trim()).entries()) {
        out = out.replace(new RegExp(`\\b${term.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi'), () => { total++; return `[TERM_${i + 1}]` })
      }
      return { text: out, withheld: total ? 'terms and emails (demo preview)' : '', total }
    },
    handoffThread: async ({ threadId, toProviderId, toModel }) => {
      const src = mockState.threads.find((x) => x.id === threadId)
      if (!src) throw new Error('The source thread no longer exists.')
      if (src.providerId === toProviderId) throw new Error('Handoff requires a different target engine.')
      const toProvider = mockState.settings.providers.find((p) => p.id === toProviderId)
      const t: Thread = {
        ...src,
        id: uid(),
        title: `${src.title} → ${toProvider?.name ?? toProviderId}`,
        providerId: toProviderId,
        model: toModel?.trim() || src.model,
        apiMessages: [],
        timeline: [{ id: uid(), kind: 'notice', text: `Handed off from “${src.title}” to ${toProvider?.name ?? toProviderId}. In the desktop app the flattened transcript runs as the first turn (simulated here).`, createdAt: Date.now() }],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        status: 'idle',
      }
      mockState.threads = [t, ...mockState.threads]
      return t
    },
    commit: async (cwd, message): Promise<GitCommitResult> => ({ ok: true, hygiene: mockGitHygieneReport(cwd, `${message} clean`), message: message.replace(/^co-authored-by:.*$/gim, '').trim() || 'TensorFold changes' }),
    exec: async (_cwd, command) => {
      await new Promise((r) => setTimeout(r, 250))
      const blocked = mockBlockedCommandReason(command)
      if (blocked) return { output: `Blocked by TensorFold safety policy: ${blocked}.`, code: 126 }
      if (/^ls/.test(command)) return { output: 'src/        electron/    package.json\nindex.html  README.md    tsconfig.json', code: 0 }
      if (/git status/.test(command)) return { output: 'On branch main\nnothing to commit, working tree clean', code: 0 }
      return { output: `(demo) ran: ${command}\nThe desktop app runs this in your workspace and streams real output.`, code: 0 }
    },
    onAgentEvent: (cb) => {
      mockState.listener = cb
      // Replay the demo's live run states once, so the sidebar shows running and waiting rows.
      if (!demoRunStatesSent) { demoRunStatesSent = true; setTimeout(() => demoRunStateEvents().forEach(emit), 300) }
      return () => { mockState.listener = null }
    },

    // SSH remote workspaces — demo stubs (real SSH only runs in the desktop app)
    listSshHosts: async () => mockState.settings.sshHosts ?? [],
    sshStatus: async (hostId) => ({ hostId, phase: 'idle' as const }),
    saveSshHost: async (host) => {
      const list = mockState.settings.sshHosts ?? []
      const idx = list.findIndex((h) => h.id === host.id)
      mockState.settings.sshHosts = idx >= 0 ? list.map((h) => (h.id === host.id ? host : h)) : [...list, host]
      return mockState.settings.sshHosts
    },
    removeSshHost: async (hostId) => {
      mockState.settings.sshHosts = (mockState.settings.sshHosts ?? []).filter((h) => h.id !== hostId)
      return mockState.settings.sshHosts
    },
    testSshHost: async () => ({ ok: false, message: 'SSH connections only run in the desktop app.' }),
    connectSshHost: async (hostId) => ({ hostId, phase: 'error' as const, lastError: 'SSH only runs in the desktop app.' }),
    disconnectSshHost: async () => ({ ok: true }),
    resolveSshSecret: async () => ({ ok: true }),
    resolveSshTrust: async () => ({ ok: true }),
    listRemoteDir: async () => [],
    onSshStatus: () => () => {},
    onPersistenceFailure: () => () => {},
    onSshSecretPrompt: () => () => {},
    onSshTrustPrompt: () => () => {},
    windowAction: async () => {},
    openExternal: async () => {},

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
    onProjectUpdate: () => () => {},
    onThreadUpdate: () => () => {},
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
    onRoomUpdate: () => () => {},
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
    cancelRoom: async () => {},
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
    onMission: () => () => {},
    onIncident: () => () => {},
    ...createMockKnowledgeApi(),
    ...engineApi,
    // Browser demo: pretend the plain shell + Claude Code CLIs are installed (no real PATH probe here).
    listClis: async () => [
      { id: 'shell', name: 'Shell', short: 'zsh', command: null, resumeCommand: null, color: '#8B8B97', icon: 'terminal', builtin: true },
      { id: 'claude', name: 'Claude Code', short: 'claude', command: 'claude', resumeCommand: 'claude --continue', color: '#D97757', icon: 'spark', builtin: true },
    ],
    listCliTools: async (): Promise<CliToolStatus[]> => [
      { id: 'shell', name: 'Shell', short: 'zsh', command: null, resumeCommand: null, color: '#8B8B97', icon: 'terminal', builtin: true, installed: true, version: null, effectiveCommand: null, configuredPath: null, providerIds: [] },
      { id: 'codex', name: 'Codex', short: 'codex', command: 'codex', resumeCommand: 'codex resume --last', color: '#2FB9B0', icon: 'code', builtin: true, installed: true, version: 'codex-cli 0.142.5', effectiveCommand: 'codex', configuredPath: null, providerIds: ['codex-cli'], latestVersion: '0.150.0', updateCommand: 'npm i -g @openai/codex', advisory: { status: 'update_available', currentVersion: '0.142.5', latestVersion: '0.150.0', updateCommand: 'npm i -g @openai/codex', canUpdate: true, message: 'Codex 0.150.0 is available (you have 0.142.5).' } },
      { id: 'claude', name: 'Claude Code', short: 'claude', command: 'claude', resumeCommand: 'claude --resume', color: '#D97757', icon: 'spark', builtin: true, installed: true, version: '2.1.280 (Claude Code)', effectiveCommand: 'claude', configuredPath: null, providerIds: ['claude-cli'] },
      { id: 'cursor', name: 'Cursor Agent', short: 'cursor', command: 'cursor-agent', resumeCommand: null, color: '#8B8B97', icon: 'spark', builtin: true, installed: false, version: null, effectiveCommand: 'cursor-agent', configuredPath: null, providerIds: [] },
      { id: 'gemini', name: 'Gemini', short: 'gemini', command: 'gemini', resumeCommand: null, color: '#6E7BF2', icon: 'globe', builtin: true, installed: false, version: null, effectiveCommand: 'gemini', configuredPath: null, providerIds: [] },
      { id: 'grok', name: 'Grok', short: 'grok', command: 'grok', resumeCommand: null, color: '#4DA3FF', icon: 'spark', builtin: true, installed: false, version: null, effectiveCommand: 'grok', configuredPath: null, providerIds: [] },
      // A wrapper script whose Python venv is gone: the binary is there but --version fails.
      { id: 'hermes', name: 'Hermes', short: 'hermes', command: 'hermes', resumeCommand: 'hermes --resume', color: '#C9A54E', icon: 'spark', builtin: true, installed: true, state: 'broken', problem: '~/.local/bin/hermes: line 4: ~/.hermes/venv/bin/python: No such file or directory', version: null, effectiveCommand: 'hermes', configuredPath: null, providerIds: [] },
      { id: 'opencode', name: 'OpenCode', short: 'opencode', command: 'opencode', resumeCommand: 'opencode --continue', color: '#E06C9F', icon: 'code', builtin: true, installed: true, version: '1.14.20', effectiveCommand: 'opencode', configuredPath: null, providerIds: [] },
      { id: 'kilo', name: 'Kilo', short: 'kilo', command: 'kilo', resumeCommand: null, color: '#B48AF7', icon: 'code', builtin: true, installed: false, version: null, effectiveCommand: 'kilo', configuredPath: null, providerIds: [] },
      { id: 'pi', name: 'Pi', short: 'pi', command: 'pi', resumeCommand: 'pi --resume', color: '#5BC873', icon: 'cpu', builtin: true, installed: true, version: '0.80.3', effectiveCommand: 'pi', configuredPath: null, providerIds: [] },
    ],
    // Browser demo spine: an empty, well-shaped replay surface (the demo's agent events are
    // simulated renderer-side, so there is no host log to replay).
    spineReadSince: async () => ({ events: [], hasMore: false, snapshotRequired: false }),
    spineReceipt: async () => null,
    spineAdvanceCursor: async (input) => input.sequence,
    // Browser demo: two importable external-harness sessions so the import flow is visible.
    listImportableSessions: async () => [
      { sessionId: 'demo-codex-session-1', source: 'codex' as const, title: 'Wire up the payments webhook', cwd: '/Users/demo/work/api', model: 'gpt-5.5', createdAt: Date.now() - 3 * 60 * 60_000, updatedAt: Date.now() - 20 * 60_000, messageCount: 34, resumeCursor: { threadId: 'demo-codex-session-1' }, truncated: false },
      { sessionId: 'demo-claude-session-1', source: 'claude' as const, title: 'Refactor auth middleware', cwd: '/Users/demo/work/api', model: 'claude-fable-5', createdAt: Date.now() - 26 * 60 * 60_000, updatedAt: Date.now() - 2 * 60 * 60_000, messageCount: 58, resumeCursor: { sessionId: 'demo-claude-session-1' }, truncated: false },
    ],
    importExternalSession: async (input) => {
      const sourceName = input.source === 'codex' ? 'Codex' : 'Claude Code'
      const thread: Thread = {
        id: `mock-import-${uid()}`,
        title: input.title || `Imported ${sourceName} session`,
        kind: 'task',
        projectId: null,
        cwd: input.cwd ?? '',
        workspace: null,
        providerId: null,
        providerAccountId: null,
        model: input.model ?? null,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        status: 'idle',
        apiMessages: [],
        timeline: [{ id: `mock-import-notice-${uid()}`, kind: 'notice', text: `Imported from a ${sourceName} session (${input.sessionId.slice(0, 8)}…). The next message resumes that session.`, createdAt: Date.now() }],
        ...(input.source === 'codex' ? { codexResumeCursor: { threadId: input.sessionId } } : { claudeResumeCursor: { sessionId: input.sessionId } }),
      }
      mockState.threads.unshift(thread)
      return thread
    },
    startTerminal: async (projectId, cliId, opts) => ({ id: `mock-term-${uid()}`, projectId, cliId, title: opts?.label?.trim() || (cliId === 'shell' ? 'Shell' : cliId), status: 'idle', createdAt: Date.now() }),
    writeTerminal: async () => {},
    resizeTerminal: async () => {},
    killTerminal: async () => {},
    renameTerminal: async () => {},
    listTerminals: async () => [],
    // Browser demo has no persisted PTYs, so nothing is ever restorable; restore is a no-op echo.
    restorableTerminals: async () => [],
    restoreTerminal: async (sessionId) => ({ id: sessionId, projectId: '', cliId: 'shell', title: 'Shell', status: 'idle', createdAt: Date.now() }),
    terminalBacklog: async () => '$ npm test\n\x1b[32mPASS\x1b[0m src/app.test.ts (2.1s)\n\x1b[31mFAIL\x1b[0m src/auth.test.ts — 1 failing\n  ● login › rejects a bad password\n',
    terminalSince: async () => ({ chunks: [], gap: false, head: 0 }),
    onTerminalData: () => () => {},
    onTerminalStatus: () => () => {},
    onTerminalSessions: () => () => {},
  }
}
