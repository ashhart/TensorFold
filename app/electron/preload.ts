import { contextBridge, ipcRenderer } from 'electron'
import type { AgentEvent, ArchiveKind, ArchiveSnapshot, Attachment, CueRunPayload, CueRunResult, Settings, StartRunPayload, Thread, RendererThreadPatch, Project, Room, RoomStreamEvent, Ticket, TicketRemoteLifecycleCommand, TicketWorkOptions, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitRepositoryState, GitStatus, TokenUsageSnapshot, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, ChannelApprovalRequest, ChannelApprovalResult, AppUpdateStatus, Provider, OmlxDiscovery, ProviderDiscovery, ProviderDiscoveryKind, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, EmailConnectorSettings, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSummaryResult, EmailSyncResult, EmailThreadResult, Skill, ProjectSkillView, SkillSuggestionDiff, Artifact, Command, SkillSuggestion, McpServerConfig, McpSnapshot, McpToolCallResult, BotControlPollResult, ControllerSnapshot, ControllerStatus, ResolvedInstructionGraph, ProjectMemoryCandidate, ProjectMemoryCandidateDiff, ProjectMemoryEntry, WebhookEndpoint, WebhookSnapshot, RemoteControlPairingInfo, RemoteControlStatus, RemoteConnectionTestResult, RemoteDirectoryListResult, RemoteOrchestratorSnapshotResult, PersistenceFailure, SecurityControlSnapshot, SecurityPurgeResult, SshHost, SshConnectionState, SshSecretPrompt, SshTrustPrompt, WorkspaceLocation, CliDefinition, CliSession, CliToolStatus, ImportableSession, ImportSessionSource, SpineAggregate, ReadSinceResult, CommandReceipt, TurnCheckpoint, PersistedProviderStatus } from './agent/types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarConnectorSettings, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DocumentConnectorSettings, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, IMessageItem, IMessageReadResult, IMessageSyncResult, WhatsAppBridgeConnectorSettings, DiscordConnectorSettings, DiscordChannel, DiscordSource, MattermostConnectorSettings, MattermostChannel, MattermostSource, MatrixConnectorSettings, MatrixRoom, MatrixSource, GoogleChatConnectorSettings, GoogleChatSpace, GoogleChatSource, SlackConnectorSettings, SlackListOptions, SlackMessage, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from './agent/types'
// Phase 6/7 (incident mode + run export) — separate import line, additive vs the Codex lane.
import type { Incident, CreateIncidentInput, IncidentEventKind, IncidentMitigation, ExportOptions, ExportResult, ExportSaveResult, ExportImportResult } from './agent/types'
import type { Readiness, CliLoginResult } from './agent/types'
import type { LoopRecord, LoopStartInput } from './agent/types'
import type { LocalTtsDiscovery, VoiceSynthesisResult, VoiceTranscriptionInput, VoiceTranscriptionResult } from './agent/types'
import type { KokoroStatus, KokoroSynthesisInput } from './agent/types'
import type { OrchestrationEvent } from './agent/types'
import type { ArtifactPreview } from './agent/types'
import type { RoomManualMemoryCommand, RoomMergeResult } from './agent/types'
import type { EngineConnection, EngineEvent, EngineLearningSnapshot, EngineInstalledModel, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress } from './agent/types'
import type { MobileBridgePairing, MobileBridgeStatus } from './agent/types/mobile-bridge'

import type { EngineModelLibrarySnapshot, EngineLibraryScanOptions } from './agent/types'
import type { PhoneSnapshot, PhoneConnectInput, PhoneDeveloperConnectInput, PhonePairInput, PhoneActionInput } from './agent/types'
import type { BotVmStatus, BotVmSetupStatus } from './agent/types'
import type { UserQuestion, UserQuestionAnswer } from './agent/types'

const api = {
  userQuestionAnswer: (input: UserQuestionAnswer): Promise<UserQuestion> => ipcRenderer.invoke('userQuestion:answer', input),
  userQuestionState: (input: { threadId: string; questionId: string }): Promise<UserQuestion | null> => ipcRenderer.invoke('userQuestion:state', input),
  botVmStatus: (botId: string): Promise<BotVmStatus> => ipcRenderer.invoke('botVm:status', botId),
  botVmManagedSetup: (botId: string): Promise<BotVmStatus> => ipcRenderer.invoke('botVm:managedSetup', botId),
  botVmSetupStatus: (): Promise<BotVmSetupStatus> => ipcRenderer.invoke('botVm:setupStatus'),
  botVmCancelSetup: (): Promise<BotVmSetupStatus> => ipcRenderer.invoke('botVm:cancelSetup'),
  botVmProvision: (botId: string): Promise<BotVmStatus> => ipcRenderer.invoke('botVm:provision', botId),
  botVmStart: (botId: string): Promise<BotVmStatus> => ipcRenderer.invoke('botVm:start', botId),
  botVmStop: (botId: string): Promise<BotVmStatus> => ipcRenderer.invoke('botVm:stop', botId),
  botVmRebuild: (botId: string): Promise<BotVmStatus> => ipcRenderer.invoke('botVm:rebuild', botId),
  botVmPickRuntime: (): Promise<boolean | null> => ipcRenderer.invoke('botVm:pickRuntime'),
  botVmStartRuntime: (): Promise<void> => ipcRenderer.invoke('botVm:startRuntime'),
  createBotChat: (presetId: string): Promise<Thread> => ipcRenderer.invoke('bots:createChat', { presetId }),
  phoneSnapshot: (): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:snapshot'),
  phoneRefresh: (): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:refresh'),
  phonePickAdb: (): Promise<PhoneSnapshot | null> => ipcRenderer.invoke('phone:pickAdb'),
  phonePairWifi: (input: PhonePairInput): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:pairWifi', input),
  phoneConnectWifi: (endpoint: string): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:connectWifi', endpoint),
  phoneConnect: (input: PhoneConnectInput): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:connect', input),
  phoneConnectDeveloper: (input: PhoneDeveloperConnectInput): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:connectDeveloper', input),
  phoneGrantChat: (threadId: string): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:grantChat', threadId),
  phoneStop: (): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:stop'),
  phoneDisconnect: (): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:disconnect'),
  phoneObserve: (): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:observe'),
  phoneAction: (input: PhoneActionInput): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:action', input),
  phoneMirroringPermissions: (): Promise<PhoneSnapshot> => ipcRenderer.invoke('phone:mirroringPermissions'),
  onPhoneEvent: (cb: (snapshot: PhoneSnapshot) => void) => { const h = (_e: unknown, snapshot: PhoneSnapshot) => cb(snapshot); ipcRenderer.on('phone:event', h); return () => ipcRenderer.removeListener('phone:event', h) },
  platform: process.platform,
  getSecurityControls: (): Promise<SecurityControlSnapshot> => ipcRenderer.invoke('security:getControls'),
  purgeLocalMemory: (): Promise<SecurityPurgeResult> => ipcRenderer.invoke('security:purgeLocalMemory'),
  writeClipboard: (text: string): Promise<{ ok: boolean }> => ipcRenderer.invoke('clipboard:writeText', text),

  getSettings: (): Promise<Settings> => ipcRenderer.invoke('settings:get'),
  saveSettings: (s: Settings): Promise<Settings> => ipcRenderer.invoke('settings:save', s),
  desktopModeSet: (mode: 'minimal' | 'full', onboardingComplete?: boolean): Promise<Settings> => ipcRenderer.invoke('desktop:modeSet', { mode, onboardingComplete }),
  desktopModeHide: (): Promise<{ hidden: boolean }> => ipcRenderer.invoke('desktop:hide'),
  onDesktopModeChanged: (cb: (settings: Settings) => void) => { const h = (_e: unknown, settings: Settings) => cb(settings); ipcRenderer.on('desktop:modeChanged', h); return () => ipcRenderer.removeListener('desktop:modeChanged', h) },
  onDesktopNavigate: (cb: (input: { target: 'overview' | 'models' | 'api' }) => void) => { const h = (_e: unknown, input: { target: 'overview' | 'models' | 'api' }) => cb(input); ipcRenderer.on('desktop:navigate', h); return () => ipcRenderer.removeListener('desktop:navigate', h) },
  revokeApprovalRule: (id: string): Promise<Settings> => ipcRenderer.invoke('approval:revokeRule', id),
  getRemoteStatus: (): Promise<RemoteControlStatus> => ipcRenderer.invoke('remote:status'),
  getMobileBridgeStatus: (): Promise<MobileBridgeStatus> => ipcRenderer.invoke('mobile:bridgeStatus'),
  configureMobileBridge: (input: { enabled: boolean; host: string; port: number }): Promise<MobileBridgeStatus> => ipcRenderer.invoke('mobile:configureBridge', input),
  issueMobilePairing: (label?: string): Promise<MobileBridgePairing> => ipcRenderer.invoke('mobile:issuePairing', label),
  getRemotePairingInfo: (): Promise<RemoteControlPairingInfo> => ipcRenderer.invoke('remote:pairingInfo'),
  regenerateRemoteToken: (): Promise<RemoteControlPairingInfo> => ipcRenderer.invoke('remote:regenerateToken'),
  // Per-device pairing (VST-035): one-time credential → device session; list/revoke devices.
  issueDevicePairing: (label?: string): Promise<{ credential: string; expiresAt: number; url: string | null }> => ipcRenderer.invoke('remote:issuePairing', label),
  listDeviceSessions: (): Promise<Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>> => ipcRenderer.invoke('remote:listDeviceSessions'),
  revokeDeviceSession: (sessionId: string): Promise<{ revoked: boolean }> => ipcRenderer.invoke('remote:revokeDeviceSession', sessionId),
  testRemoteClient: (input?: Partial<Settings['remoteClient']>): Promise<RemoteConnectionTestResult> => ipcRenderer.invoke('remote:testClient', input),
  getRemoteSnapshot: (hostId?: string | null): Promise<RemoteOrchestratorSnapshotResult> => ipcRenderer.invoke('remote:snapshot', hostId),
  getRemoteOrchestratorArchives: (hostId: string | null | undefined): Promise<ArchiveSnapshot> => ipcRenderer.invoke('remote:archives', { hostId }),
  listImportableSessions: (): Promise<ImportableSession[]> => ipcRenderer.invoke('sessions:importable'),
  spineReadSince: (input: { since?: number; aggregate?: SpineAggregate; streamId?: string; limit?: number }): Promise<ReadSinceResult> => ipcRenderer.invoke('spine:readSince', input),
  spineReceipt: (commandId: string): Promise<CommandReceipt | null> => ipcRenderer.invoke('spine:receipt', commandId),
  spineAdvanceCursor: (input: { projector: string; sequence: number }): Promise<number> => ipcRenderer.invoke('spine:advanceCursor', input),
  importExternalSession: (input: { sessionId: string; source: ImportSessionSource; title?: string; cwd?: string | null; model?: string | null }): Promise<Thread> => ipcRenderer.invoke('sessions:import', input),
  listRemoteOrchestratorDir: (hostId?: string | null, dir?: string | null): Promise<RemoteDirectoryListResult> => ipcRenderer.invoke('remote:listDir', { hostId, dir }),
  createRemoteOrchestratorProject: (hostId: string | null | undefined, project: Partial<Project> & { name?: string; cwd?: string }): Promise<Project> => ipcRenderer.invoke('remote:createProject', { hostId, project }),
  updateRemoteOrchestratorProject: (hostId: string | null | undefined, id: string, patch: Partial<Project>): Promise<Project> => ipcRenderer.invoke('remote:updateProject', { hostId, id, patch }),
  deleteRemoteOrchestratorProject: (hostId: string | null | undefined, id: string): Promise<Project | null> => ipcRenderer.invoke('remote:deleteProject', { hostId, id }),
  restoreRemoteOrchestratorProject: (hostId: string | null | undefined, id: string): Promise<Project | null> => ipcRenderer.invoke('remote:restoreProject', { hostId, id }),
  createRemoteOrchestratorThread: (hostId: string | null | undefined, input: { cwd?: string | null; kind?: 'task' | 'chat'; projectId?: string | null }): Promise<Thread> => ipcRenderer.invoke('remote:createThread', { hostId, ...input }),
  archiveRemoteOrchestratorThread: (hostId: string | null | undefined, id: string): Promise<Thread | null> => ipcRenderer.invoke('remote:archiveThread', { hostId, id }),
  restoreRemoteOrchestratorThread: (hostId: string | null | undefined, id: string): Promise<Thread | null> => ipcRenderer.invoke('remote:restoreThread', { hostId, id }),
  remoteListClis: (hostId: string | null | undefined): Promise<CliDefinition[]> => ipcRenderer.invoke('remote:listClis', { hostId }),
  remoteTerminalStart: (hostId: string | null | undefined, input: { projectId: string; cliId: string; command?: string; label?: string }): Promise<CliSession> => ipcRenderer.invoke('remote:terminalStart', { hostId, ...input }),
  remoteTerminalWrite: (hostId: string | null | undefined, sessionId: string, data: string): Promise<void> => ipcRenderer.invoke('remote:terminalWrite', { hostId, sessionId, data }),
  remoteTerminalResize: (hostId: string | null | undefined, sessionId: string, cols: number, rows: number): Promise<void> => ipcRenderer.invoke('remote:terminalResize', { hostId, sessionId, cols, rows }),
  remoteTerminalRename: (hostId: string | null | undefined, sessionId: string, title: string): Promise<void> => ipcRenderer.invoke('remote:terminalRename', { hostId, sessionId, title }),
  remoteTerminalKill: (hostId: string | null | undefined, sessionId: string): Promise<void> => ipcRenderer.invoke('remote:terminalKill', { hostId, sessionId }),
  remoteTerminalList: (hostId: string | null | undefined, projectId: string): Promise<CliSession[]> => ipcRenderer.invoke('remote:terminalList', { hostId, projectId }),
  approveRemoteOrchestrator: (hostId: string | null | undefined, callId: string, decision: 'allow' | 'allow_always' | 'deny'): Promise<void> => ipcRenderer.invoke('remote:approve', { hostId, callId, decision }),
  startRemoteOrchestratorThread: (hostId: string | null | undefined, payload: StartRunPayload): Promise<{ runId: string }> => ipcRenderer.invoke('remote:startThread', { hostId, payload }),
  guideRemoteOrchestratorThread: (hostId: string | null | undefined, payload: CueRunPayload): Promise<CueRunResult> => ipcRenderer.invoke('remote:guideThread', { hostId, payload }),
  cancelRemoteOrchestratorThread: (hostId: string | null | undefined, threadId: string): Promise<void> => ipcRenderer.invoke('remote:cancelThread', { hostId, threadId }),
  createRemoteOrchestratorRoom: (hostId: string | null | undefined, projectId: string, name: string): Promise<Room> => ipcRenderer.invoke('remote:createRoom', { hostId, projectId, name }),
  updateRemoteOrchestratorRoom: (hostId: string | null | undefined, id: string, patch: Partial<Room>): Promise<Room> => ipcRenderer.invoke('remote:updateRoom', { hostId, id, patch }),
  renameRemoteOrchestratorRoom: (hostId: string | null | undefined, id: string, name: string): Promise<Room> => ipcRenderer.invoke('remote:renameRoom', { hostId, id, name }),
  steerRemoteOrchestratorRoom: (hostId: string | null | undefined, roomId: string, text: string): Promise<Room['events'][number]> => ipcRenderer.invoke('remote:steerRoom', { hostId, roomId, text }),
  mutateRemoteOrchestratorRoomMemory: (hostId: string | null | undefined, roomId: string, command: RoomManualMemoryCommand): Promise<Room> => ipcRenderer.invoke('remote:mutateRoomMemory', { hostId, roomId, command }),
  deleteRemoteOrchestratorRoom: (hostId: string | null | undefined, id: string): Promise<Room | null> => ipcRenderer.invoke('remote:deleteRoom', { hostId, id }),
  restoreRemoteOrchestratorRoom: (hostId: string | null | undefined, id: string): Promise<Room | null> => ipcRenderer.invoke('remote:restoreRoom', { hostId, id }),
  startRemoteOrchestratorRoom: (hostId: string | null | undefined, roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }> => ipcRenderer.invoke('remote:startRoom', { hostId, roomId, problem, attachments }),
  cancelRemoteOrchestratorRoom: (hostId: string | null | undefined, roomId: string): Promise<void> => ipcRenderer.invoke('remote:cancelRoom', { hostId, roomId }),
  createRemoteOrchestratorTicket: (hostId: string | null | undefined, ticket: Partial<Ticket> & { projectId: string }): Promise<Ticket> => ipcRenderer.invoke('remote:createTicket', { hostId, ticket }),
  startRemoteOrchestratorTicketExecution: (hostId: string | null | undefined, id: string, userText: string): Promise<{ ticket: Ticket; thread: Thread; runId: string }> => ipcRenderer.invoke('remote:startTicketExecution', { hostId, id, userText }),
  updateRemoteOrchestratorTicket: (hostId: string | null | undefined, id: string, patch: Partial<Ticket>): Promise<Ticket> => ipcRenderer.invoke('remote:updateTicket', { hostId, id, patch }),
  transitionRemoteOrchestratorTicket: (hostId: string | null | undefined, id: string, command: TicketRemoteLifecycleCommand): Promise<Ticket> => ipcRenderer.invoke('remote:transitionTicket', { hostId, id, command }),
  deleteRemoteOrchestratorTicket: (hostId: string | null | undefined, id: string): Promise<Ticket | null> => ipcRenderer.invoke('remote:deleteTicket', { hostId, id }),
  restoreRemoteOrchestratorTicket: (hostId: string | null | undefined, id: string): Promise<Ticket | null> => ipcRenderer.invoke('remote:restoreTicket', { hostId, id }),

  // SSH remote workspaces
  listSshHosts: (): Promise<SshHost[]> => ipcRenderer.invoke('ssh:listHosts'),
  sshStatus: (hostId: string): Promise<SshConnectionState> => ipcRenderer.invoke('ssh:status', hostId),
  saveSshHost: (host: SshHost): Promise<SshHost[]> => ipcRenderer.invoke('ssh:saveHost', host),
  removeSshHost: (hostId: string): Promise<SshHost[]> => ipcRenderer.invoke('ssh:removeHost', hostId),
  testSshHost: (hostId: string): Promise<{ ok: boolean; message: string }> => ipcRenderer.invoke('ssh:test', hostId),
  connectSshHost: (hostId: string): Promise<SshConnectionState> => ipcRenderer.invoke('ssh:connect', hostId),
  disconnectSshHost: (hostId: string): Promise<{ ok: boolean }> => ipcRenderer.invoke('ssh:disconnect', hostId),
  resolveSshSecret: (promptId: string, value: string | null): Promise<{ ok: boolean }> => ipcRenderer.invoke('ssh:resolveSecret', promptId, value),
  resolveSshTrust: (promptId: string, ok: boolean): Promise<{ ok: boolean }> => ipcRenderer.invoke('ssh:resolveTrust', promptId, ok),
  listRemoteDir: (hostId: string, dir: string): Promise<Array<{ name: string; isDirectory: boolean }>> => ipcRenderer.invoke('ssh:listRemoteDir', hostId, dir),

  getThreads: (): Promise<Thread[]> => ipcRenderer.invoke('threads:get'),
  getArchives: (): Promise<ArchiveSnapshot> => ipcRenderer.invoke('archives:get'),
  archiveItem: (kind: ArchiveKind, id: string): Promise<ArchiveSnapshot> => ipcRenderer.invoke('archives:archive', { kind, id }),
  restoreItem: (kind: ArchiveKind, id: string): Promise<ArchiveSnapshot> => ipcRenderer.invoke('archives:restore', { kind, id }),
  createThread: (cwd: string, kind: 'task' | 'chat' = 'task', projectId: string | null = null, commandId?: string, incognito = false): Promise<Thread> => ipcRenderer.invoke('threads:create', { cwd, kind, projectId, commandId, incognito }),
  setThreadProject: (id: string, projectId: string | null): Promise<void> => ipcRenderer.invoke('threads:setProject', { id, projectId }),
  deleteThread: (id: string): Promise<void> => ipcRenderer.invoke('threads:delete', id),
  renameThread: (id: string, title: string): Promise<void> => ipcRenderer.invoke('threads:rename', { id, title }),
  setThreadEngine: (id: string, engine: 'native' | 'omp'): Promise<void> => ipcRenderer.invoke('threads:setEngine', { id, engine }),
  ompStatus: (): Promise<{ available: boolean; binary: string; version?: string }> => ipcRenderer.invoke('omp:status'),
  configureThread: (id: string, providerId: string, model: string, providerAccountId: string | null = null): Promise<void> => ipcRenderer.invoke('threads:configure', { id, providerId, model, providerAccountId }),
  patchThread: (id: string, patch: RendererThreadPatch): Promise<Thread> => ipcRenderer.invoke('threads:patch', { id, patch }),
  setThreadWorkspace: (id: string, workspace: WorkspaceLocation | null, cwd?: string | null): Promise<Thread> => ipcRenderer.invoke('threads:setWorkspace', { id, workspace, cwd }),
  appendThreadNotice: (id: string, text: string): Promise<void> => ipcRenderer.invoke('threads:appendNotice', { id, text }),
  compactThread: (id: string): Promise<{ ok: boolean; thread: Thread }> => ipcRenderer.invoke('thread:compact', id),

  startRun: (payload: StartRunPayload): Promise<{ runId: string }> => ipcRenderer.invoke('agent:start', payload),
  cueRun: (payload: CueRunPayload): Promise<CueRunResult> => ipcRenderer.invoke('agent:cue', payload),
  cancelRun: (threadId: string): Promise<void> => ipcRenderer.invoke('agent:cancel', threadId),
  approve: (callId: string, decision: 'allow' | 'allow_always' | 'deny'): Promise<void> =>
    ipcRenderer.invoke('agent:approve', { callId, decision }),

  pickDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:pickDir'),

  fetchModels: (provider: Provider): Promise<string[]> => ipcRenderer.invoke('provider:models', provider),
  testProviderResponse: (input: { provider: Provider; model: string }): Promise<ProviderGenerationProbe> => ipcRenderer.invoke('provider:testResponse', input),
  getModelReadiness: (input: { providerId?: string | null; accountId?: string | null; model?: string | null }): Promise<Readiness> => ipcRenderer.invoke('provider:readiness', input),
  loginCliProvider: (input: { providerId: string; accountId?: string | null }): Promise<CliLoginResult> => ipcRenderer.invoke('cliAuth:login', input),
  cancelCliLogin: (providerId: string): Promise<{ cancelled: boolean }> => ipcRenderer.invoke('cliAuth:cancelLogin', providerId),
  checkCliTransport: (input: { transport: 'codex-cli' | 'claude-cli'; cliPath?: string }): Promise<{ loggedIn: boolean; binaryPresent: boolean; detail?: string; message?: string }> => ipcRenderer.invoke('cliAuth:checkTransport', input),
  getWorktrees: (arg: { cwd: string; workspace?: WorkspaceLocation | null }) => ipcRenderer.invoke('workspace:worktrees', arg),
  checkoutBranch: (arg: { cwd: string; workspace?: WorkspaceLocation | null; branch: string }) => ipcRenderer.invoke('workspace:checkoutBranch', arg),
  createWorktree: (arg: { cwd: string; workspace?: WorkspaceLocation | null; name: string; mode: 'worktree' | 'branch' }) => ipcRenderer.invoke('workspace:createWorktree', arg),
  discoverLocalProvider: (kind: ProviderDiscoveryKind, baseUrl?: string, apiKey?: string): Promise<ProviderDiscovery> => ipcRenderer.invoke('provider:discoverLocal', { kind, baseUrl, apiKey }),
  discoverOmlxProvider: (): Promise<OmlxDiscovery> => ipcRenderer.invoke('provider:discoverOmlx'),
  synthesizeVoice: (text: string): Promise<VoiceSynthesisResult> => ipcRenderer.invoke('voice:synthesize', { text }),
  synthesizeLocalVoice: (text: string): Promise<VoiceSynthesisResult> => ipcRenderer.invoke('voice:localSynthesize', { text }),
  getKokoroStatus: (): Promise<KokoroStatus> => ipcRenderer.invoke('voice:kokoroStatus'),
  installKokoro: (): Promise<KokoroStatus> => ipcRenderer.invoke('voice:kokoroInstall'),
  cancelKokoroInstall: (): Promise<KokoroStatus> => ipcRenderer.invoke('voice:kokoroCancelInstall'),
  removeKokoro: (): Promise<KokoroStatus> => ipcRenderer.invoke('voice:kokoroRemove'),
  synthesizeKokoroVoice: (input: KokoroSynthesisInput): Promise<VoiceSynthesisResult> => ipcRenderer.invoke('voice:kokoroSynthesize', input),
  cancelKokoroVoice: (requestId: string): Promise<{ cancelled: boolean }> => ipcRenderer.invoke('voice:kokoroCancelSpeech', { requestId }),
  transcribeVoice: (input: VoiceTranscriptionInput): Promise<VoiceTranscriptionResult> => ipcRenderer.invoke('voice:transcribe', input),
  discoverLocalTts: (baseUrl?: string): Promise<LocalTtsDiscovery> => ipcRenderer.invoke('voice:discoverLocalTts', { baseUrl }),
  getTokenUsage: (): Promise<TokenUsageSnapshot> => ipcRenderer.invoke('usage:tokens'),
  getContextPressure: (): Promise<ContextPressureSnapshot> => ipcRenderer.invoke('usage:context-pressure'),
  // Provider rate-limit status (VST-037): persisted per-provider windows for the picker + doctor.
  getProviderStatus: (): Promise<PersistedProviderStatus> => ipcRenderer.invoke('providerStatus:get'),
  onProviderStatus: (cb: (s: PersistedProviderStatus) => void) => {
    const handler = (_: unknown, s: PersistedProviderStatus) => cb(s)
    ipcRenderer.on('providerStatus:update', handler)
    return () => ipcRenderer.removeListener('providerStatus:update', handler)
  },
  onUsageCleared: (cb: () => void) => {
    const handler = () => cb()
    ipcRenderer.on('usage:cleared', handler)
    return () => ipcRenderer.removeListener('usage:cleared', handler)
  },
  getModelRouterSnapshot: (): Promise<ModelRouterSnapshot> => ipcRenderer.invoke('modelRouter:snapshot'),
  getAssistantSnapshot: (): Promise<AssistantControlSnapshot> => ipcRenderer.invoke('assistant:snapshot'),
  recordConnectorInteraction: (input: ConnectorInteractionInput): Promise<ConnectorEvent> => ipcRenderer.invoke('assistant:interaction', input),
  resolveConnectorApproval: (eventId: string, decision: 'approve' | 'deny'): Promise<ConnectorEvent> => ipcRenderer.invoke('assistant:resolveApproval', { eventId, decision }),
  getControllerStatus: (): Promise<ControllerStatus> => ipcRenderer.invoke('controller:status'),
  enableController: (): Promise<ControllerStatus> => ipcRenderer.invoke('controller:enable'),
  captureController: (): Promise<ControllerSnapshot> => ipcRenderer.invoke('controller:capture'),
  getMcpSnapshot: (): Promise<McpSnapshot> => ipcRenderer.invoke('mcp:snapshot'),
  saveMcpServer: (server: Partial<McpServerConfig>): Promise<McpSnapshot> => ipcRenderer.invoke('mcp:saveServer', server),
  deleteMcpServer: (id: string): Promise<McpSnapshot> => ipcRenderer.invoke('mcp:deleteServer', id),
  connectMcpServer: (id: string): Promise<McpSnapshot> => ipcRenderer.invoke('mcp:connect', id),
  disconnectMcpServer: (id: string): Promise<McpSnapshot> => ipcRenderer.invoke('mcp:disconnect', id),
  refreshMcpServer: (id?: string): Promise<McpSnapshot> => ipcRenderer.invoke('mcp:refresh', id),
  testMcpTool: (callName: string, args?: Record<string, any>): Promise<McpToolCallResult> => ipcRenderer.invoke('mcp:testTool', { callName, args }),
  addEmailSource: (dir: string): Promise<EmailConnectorSettings> => ipcRenderer.invoke('email:addSource', dir),
  updateEmailSource: (id: string, patch: Partial<EmailSourceConfig>): Promise<EmailConnectorSettings> => ipcRenderer.invoke('email:updateSource', { id, patch }),
  removeEmailSource: (id: string): Promise<EmailConnectorSettings> => ipcRenderer.invoke('email:removeSource', id),
  syncEmail: (): Promise<EmailSyncResult> => ipcRenderer.invoke('email:sync'),
  listEmail: (opts?: EmailListOptions): Promise<EmailMessage[]> => ipcRenderer.invoke('email:list', opts),
  readEmail: (id: string): Promise<EmailReadResult> => ipcRenderer.invoke('email:read', id),
  readEmailThread: (id: string): Promise<EmailThreadResult> => ipcRenderer.invoke('email:thread', id),
  summarizeEmail: (opts?: EmailListOptions): Promise<EmailSummaryResult> => ipcRenderer.invoke('email:summary', opts),
  composeEmail: (input: EmailDraftInput): Promise<ConnectorEvent> => ipcRenderer.invoke('email:compose', input),
  resolveEmailThread: (input: EmailResolveInput): Promise<ConnectorEvent> => ipcRenderer.invoke('email:resolve', input),
  addSlackSource: (dir: string): Promise<SlackConnectorSettings> => ipcRenderer.invoke('slack:addSource', dir),
  updateSlackSource: (id: string, patch: Partial<SlackSourceConfig>): Promise<SlackConnectorSettings> => ipcRenderer.invoke('slack:updateSource', { id, patch }),
  removeSlackSource: (id: string): Promise<SlackConnectorSettings> => ipcRenderer.invoke('slack:removeSource', id),
  syncSlack: (): Promise<SlackSyncResult> => ipcRenderer.invoke('slack:sync'),
  listSlack: (opts?: SlackListOptions): Promise<SlackMessage[]> => ipcRenderer.invoke('slack:list', opts),
  readSlack: (id: string): Promise<SlackReadResult> => ipcRenderer.invoke('slack:read', id),
  summarizeSlack: (opts?: SlackListOptions): Promise<SlackSummaryResult> => ipcRenderer.invoke('slack:summary', opts),
  verifySlack: (id: string): Promise<{ ok: boolean; team?: string; bot?: string; error?: string }> => ipcRenderer.invoke('slack:verify', { id }),
  listSlackChannels: (id: string): Promise<{ id: string; name: string; isPrivate: boolean }[]> => ipcRenderer.invoke('slack:listChannels', { id }),
  sendSlack: (sourceId: string, channel: string, text: string, threadTs?: string): Promise<{ ok: boolean; ts?: string; threadTs?: string }> => ipcRenderer.invoke('slack:send', { sourceId, channel, text, threadTs }),
  addDiscordSource: (label: string): Promise<DiscordConnectorSettings> => ipcRenderer.invoke('discord:addSource', label),
  removeDiscordSource: (id: string): Promise<DiscordConnectorSettings> => ipcRenderer.invoke('discord:removeSource', id),
  verifyDiscord: (id: string): Promise<{ ok: boolean; bot?: string; guilds?: { id: string; name: string }[]; error?: string }> => ipcRenderer.invoke('discord:verify', { id }),
  listDiscordChannels: (id: string): Promise<DiscordChannel[]> => ipcRenderer.invoke('discord:listChannels', { id }),
  syncDiscord: (id: string): Promise<{ messages: number; events: number }> => ipcRenderer.invoke('discord:sync', { id }),
  sendDiscord: (sourceId: string, channelId: string, text: string, replyTo?: string): Promise<{ ok: boolean; id?: string }> => ipcRenderer.invoke('discord:send', { sourceId, channelId, text, replyTo }),
  addMattermostSource: (label: string): Promise<MattermostConnectorSettings> => ipcRenderer.invoke('mattermost:addSource', label),
  updateMattermostSource: (id: string, patch: Partial<MattermostSource>): Promise<MattermostConnectorSettings> => ipcRenderer.invoke('mattermost:updateSource', { id, patch }),
  removeMattermostSource: (id: string): Promise<MattermostConnectorSettings> => ipcRenderer.invoke('mattermost:removeSource', id),
  verifyMattermost: (id: string): Promise<{ ok: boolean; user?: string; userId?: string; error?: string }> => ipcRenderer.invoke('mattermost:verify', { id }),
  listMattermostChannels: (id: string): Promise<MattermostChannel[]> => ipcRenderer.invoke('mattermost:listChannels', { id }),
  syncMattermost: (id: string): Promise<{ messages: number; events: number }> => ipcRenderer.invoke('mattermost:sync', { id }),
  sendMattermost: (sourceId: string, channelId: string, text: string, replyTo?: string): Promise<{ ok: boolean; id?: string }> => ipcRenderer.invoke('mattermost:send', { sourceId, channelId, text, replyTo }),
  addMatrixSource: (label: string): Promise<MatrixConnectorSettings> => ipcRenderer.invoke('matrix:addSource', label),
  updateMatrixSource: (id: string, patch: Partial<MatrixSource>): Promise<MatrixConnectorSettings> => ipcRenderer.invoke('matrix:updateSource', { id, patch }),
  removeMatrixSource: (id: string): Promise<MatrixConnectorSettings> => ipcRenderer.invoke('matrix:removeSource', id),
  verifyMatrix: (id: string): Promise<{ ok: boolean; userId?: string; error?: string }> => ipcRenderer.invoke('matrix:verify', { id }),
  listMatrixChannels: (id: string): Promise<MatrixRoom[]> => ipcRenderer.invoke('matrix:listChannels', { id }),
  syncMatrix: (id: string): Promise<{ messages: number; events: number }> => ipcRenderer.invoke('matrix:sync', { id }),
  sendMatrix: (sourceId: string, roomId: string, text: string, replyTo?: string, threadRootId?: string): Promise<{ ok: boolean; id?: string }> => ipcRenderer.invoke('matrix:send', { sourceId, roomId, text, replyTo, threadRootId }),
  addGoogleChatSource: (label: string): Promise<GoogleChatConnectorSettings> => ipcRenderer.invoke('googlechat:addSource', label),
  updateGoogleChatSource: (id: string, patch: Partial<GoogleChatSource>): Promise<GoogleChatConnectorSettings> => ipcRenderer.invoke('googlechat:updateSource', { id, patch }),
  removeGoogleChatSource: (id: string): Promise<GoogleChatConnectorSettings> => ipcRenderer.invoke('googlechat:removeSource', id),
  verifyGoogleChat: (id: string): Promise<{ ok: boolean; mode: 'webhook' | 'token'; space?: string | null; error?: string }> => ipcRenderer.invoke('googlechat:verify', { id }),
  listGoogleChatChannels: (id: string): Promise<GoogleChatSpace[]> => ipcRenderer.invoke('googlechat:listChannels', { id }),
  syncGoogleChat: (id: string): Promise<{ messages: number; events: number }> => ipcRenderer.invoke('googlechat:sync', { id }),
  sendGoogleChat: (sourceId: string, space: string, text: string, threadKey?: string, threadName?: string): Promise<{ ok: boolean; id?: string; threadName?: string | null }> => ipcRenderer.invoke('googlechat:send', { sourceId, space, text, threadKey, threadName }),
  sendIMessage: (to: string, text: string): Promise<{ ok: boolean }> => ipcRenderer.invoke('imessage:send', { to, text }),
  addCalendarSource: (target: string): Promise<CalendarConnectorSettings> => ipcRenderer.invoke('calendar:addSource', target),
  updateCalendarSource: (id: string, patch: Partial<CalendarSourceConfig>): Promise<CalendarConnectorSettings> => ipcRenderer.invoke('calendar:updateSource', { id, patch }),
  removeCalendarSource: (id: string): Promise<CalendarConnectorSettings> => ipcRenderer.invoke('calendar:removeSource', id),
  syncCalendar: (): Promise<CalendarSyncResult> => ipcRenderer.invoke('calendar:sync'),
  listCalendar: (opts?: CalendarListOptions): Promise<CalendarEvent[]> => ipcRenderer.invoke('calendar:list', opts),
  readCalendar: (id: string): Promise<CalendarReadResult> => ipcRenderer.invoke('calendar:read', id),
  summarizeCalendar: (opts?: CalendarListOptions): Promise<CalendarSummaryResult> => ipcRenderer.invoke('calendar:summary', opts),
  createCalendarEvent: (input: CalendarMutationInput): Promise<ConnectorEvent> => ipcRenderer.invoke('calendar:create', input),
  updateCalendarEvent: (input: CalendarMutationInput & { id: string }): Promise<ConnectorEvent> => ipcRenderer.invoke('calendar:update', input),
  addDocumentSource: (target: string): Promise<DocumentConnectorSettings> => ipcRenderer.invoke('documents:addSource', target),
  updateDocumentSource: (id: string, patch: Partial<DocumentSourceConfig>): Promise<DocumentConnectorSettings> => ipcRenderer.invoke('documents:updateSource', { id, patch }),
  removeDocumentSource: (id: string): Promise<DocumentConnectorSettings> => ipcRenderer.invoke('documents:removeSource', id),
  syncDocuments: (): Promise<DocumentSyncResult> => ipcRenderer.invoke('documents:sync'),
  listDocuments: (opts?: DocumentListOptions): Promise<DocumentItem[]> => ipcRenderer.invoke('documents:list', opts),
  readDocument: (id: string): Promise<DocumentReadResult> => ipcRenderer.invoke('documents:read', id),
  summarizeDocuments: (opts?: DocumentListOptions): Promise<DocumentSummaryResult> => ipcRenderer.invoke('documents:summary', opts),
  addBrowserCurrentTabSource: (input: BrowserSnapshotInput): Promise<BrowserConnectorSettings> => ipcRenderer.invoke('browser:addCurrentTabSource', input),
  addBrowserHistorySource: (target: string): Promise<BrowserConnectorSettings> => ipcRenderer.invoke('browser:addHistorySource', target),
  updateBrowserSource: (id: string, patch: Partial<BrowserSourceConfig>): Promise<BrowserConnectorSettings> => ipcRenderer.invoke('browser:updateSource', { id, patch }),
  removeBrowserSource: (id: string): Promise<BrowserConnectorSettings> => ipcRenderer.invoke('browser:removeSource', id),
  syncBrowser: (): Promise<BrowserSyncResult> => ipcRenderer.invoke('browser:sync'),
  listBrowser: (opts?: BrowserListOptions): Promise<BrowserItem[]> => ipcRenderer.invoke('browser:list', opts),
  readBrowser: (id: string): Promise<BrowserReadResult> => ipcRenderer.invoke('browser:read', id),
  summarizeBrowser: (opts?: BrowserListOptions): Promise<BrowserSummaryResult> => ipcRenderer.invoke('browser:summary', opts),
  imessageAccess: (): Promise<{ state: string; detail: string }> => ipcRenderer.invoke('imessage:access'),
  syncImessage: (): Promise<IMessageSyncResult> => ipcRenderer.invoke('imessage:sync'),
  listImessage: (): Promise<IMessageItem[]> => ipcRenderer.invoke('imessage:list'),
  readImessage: (id: string): Promise<IMessageReadResult | null> => ipcRenderer.invoke('imessage:read', id),
  // Open the macOS Full Disk Access pane so the user can grant TensorFold read access
  // to ~/Library/Messages/chat.db (iMessage). The iMessage bridge can't read the
  // database until this is toggled on.
  openFullDiskAccess: (): Promise<void> => ipcRenderer.invoke('system:openFullDiskAccess'),
  openExternal: (url: string): Promise<void> => ipcRenderer.invoke('system:openExternal', url),
  addWhatsAppSource: (label: string): Promise<WhatsAppBridgeConnectorSettings> => ipcRenderer.invoke('whatsapp:addSource', label),
  removeWhatsAppSource: (id: string): Promise<WhatsAppBridgeConnectorSettings> => ipcRenderer.invoke('whatsapp:removeSource', id),
  startWhatsAppPairing: (id: string): Promise<{ started: boolean }> => ipcRenderer.invoke('whatsapp:startPairing', id),
  disconnectWhatsApp: (id: string): Promise<{ ok: boolean }> => ipcRenderer.invoke('whatsapp:disconnect', id),
  onWhatsAppQr: (cb: (payload: { sourceId: string; qr: string }) => void) => { const h = (_e: unknown, p: { sourceId: string; qr: string }) => cb(p); ipcRenderer.on('whatsapp:qr', h); return () => ipcRenderer.removeListener('whatsapp:qr', h) },
  onWhatsAppStatus: (cb: (payload: { sourceId: string; paired?: boolean; lastError?: string | null }) => void) => { const h = (_e: unknown, p: { sourceId: string; paired?: boolean; lastError?: string | null }) => cb(p); ipcRenderer.on('whatsapp:status', h); return () => ipcRenderer.removeListener('whatsapp:status', h) },
  pollTelegram: (): Promise<BotControlPollResult> => ipcRenderer.invoke('bot:pollTelegram'),
  getMissionSnapshot: (): Promise<MissionRuntimeSnapshot> => ipcRenderer.invoke('missions:snapshot'),
  inspectMission: (id: string): Promise<MissionInspectionSnapshot> => ipcRenderer.invoke('missions:inspect', id),
  createMission: (input: CreateMissionInput): Promise<Mission> => ipcRenderer.invoke('missions:create', input),
  createLocalPrLoop: (input: CreatePrLoopInput): Promise<CreatePrLoopResult> => ipcRenderer.invoke('prLoop:createLocal', input),
  setMissionStatus: (id: string, status: MissionStatus, message?: string): Promise<Mission> => ipcRenderer.invoke('missions:status', { id, status, message }),
  recordMissionHeartbeat: (id: string, message?: string): Promise<Mission> => ipcRenderer.invoke('missions:heartbeat', { id, message }),
  addMissionCheckpoint: (id: string, checkpoint: Partial<MissionCheckpoint> & { label: string; summary: string }): Promise<Mission> => ipcRenderer.invoke('missions:checkpoint', { id, checkpoint }),
  appendMissionJournal: (id: string, event: Partial<MissionJournalEvent> & { message: string }): Promise<Mission> => ipcRenderer.invoke('missions:journal', { id, event }),
  requestMissionApproval: (id: string, approval: Partial<MissionPendingApproval> & { callId: string; title: string; summary: string }): Promise<Mission> => ipcRenderer.invoke('missions:approval:request', { id, approval }),
  resolveMissionApproval: (id: string, approvalId: string, decision: 'allow' | 'deny'): Promise<Mission> => ipcRenderer.invoke('missions:approval:resolve', { id, approvalId, decision }),
  getUpdateStatus: (): Promise<AppUpdateStatus> => ipcRenderer.invoke('updates:get'),
  checkForUpdates: (): Promise<AppUpdateStatus> => ipcRenderer.invoke('updates:check'),
  openUpdateDownload: (): Promise<{ ok: boolean }> => ipcRenderer.invoke('updates:open'),
  downloadAppUpdate: (): Promise<{ ok: boolean }> => ipcRenderer.invoke('updates:download'),
  cancelAppUpdateDownload: (): Promise<{ cancelled: boolean }> => ipcRenderer.invoke('updates:cancel'),
  installAppUpdate: (): Promise<{ ok: boolean }> => ipcRenderer.invoke('updates:install'),
  onUpdateStatus: (cb: (s: AppUpdateStatus) => void) => {
    const handler = (_: unknown, s: AppUpdateStatus) => cb(s)
    ipcRenderer.on('app:update', handler)
    return () => ipcRenderer.removeListener('app:update', handler)
  },

  getBranch: (cwd: string, workspace?: WorkspaceLocation | null): Promise<string | null> => ipcRenderer.invoke('workspace:branch', { cwd, workspace }),
  inspectGitRepository: (cwd: string, workspace?: WorkspaceLocation | null): Promise<GitRepositoryState> => ipcRenderer.invoke('workspace:gitRepository', { cwd, workspace }),
  initGitRepository: (cwd: string, workspace?: WorkspaceLocation | null): Promise<GitRepositoryState> => ipcRenderer.invoke('workspace:initGitRepository', { cwd, workspace }),
  scanGitHygiene: (cwd: string, message = '', workspace?: WorkspaceLocation | null): Promise<GitHygieneReport> => ipcRenderer.invoke('workspace:gitHygiene', { cwd, message, workspace }),
  sanitizeGitHygiene: (cwd: string, findings: GitHygieneFinding[], workspace?: WorkspaceLocation | null): Promise<GitHygieneSanitizeResult> => ipcRenderer.invoke('workspace:sanitizeGitHygiene', { cwd, findings, workspace }),
  listFiles: (cwd: string, workspace?: WorkspaceLocation | null): Promise<string[]> => ipcRenderer.invoke('workspace:files', { cwd, workspace }),
  resolveInstructions: (cwd: string, providerId?: string | null, workspace?: WorkspaceLocation | null): Promise<ResolvedInstructionGraph> => ipcRenderer.invoke('instructions:resolve', cwd, providerId, workspace),
  gitStatus: (cwd: string, workspace?: WorkspaceLocation | null): Promise<GitStatus> => ipcRenderer.invoke('workspace:gitStatus', { cwd, workspace }),
  gitDiff: (cwd: string, paths?: string[], workspace?: WorkspaceLocation | null): Promise<GitDiffFile[]> => ipcRenderer.invoke('workspace:gitDiff', { cwd, paths, workspace }),
  revertFiles: (cwd: string, paths: string[], workspace?: WorkspaceLocation | null): Promise<void> => ipcRenderer.invoke('workspace:revertFiles', { cwd, paths, workspace }),
  commit: (cwd: string, message: string, workspace?: WorkspaceLocation | null): Promise<GitCommitResult> => ipcRenderer.invoke('workspace:commit', { cwd, message, workspace }),
  exec: (cwd: string, command: string, workspace?: WorkspaceLocation | null): Promise<{ output: string; code: number }> => ipcRenderer.invoke('workspace:exec', { cwd, command, workspace }),

  // Per-turn workspace checkpoints (undo-a-turn / turn diff).
  checkpointList: (arg: { threadId: string }): Promise<number[]> => ipcRenderer.invoke('checkpoint:list', arg),
  checkpointUndoLast: (arg: { threadId: string }): Promise<{ ok: boolean; turn?: number; error?: string; warning?: string }> => ipcRenderer.invoke('checkpoint:undoLast', arg),
  checkpointDiffLast: (arg: { threadId: string }): Promise<{ ok: boolean; turn?: number; diff?: string; error?: string }> => ipcRenderer.invoke('checkpoint:diffLast', arg),
  // Unified turn checkpoints (VST-036): files + conversation.
  checkpointTurns: (arg: { threadId: string }): Promise<TurnCheckpoint[]> => ipcRenderer.invoke('checkpoint:turns', arg),
  checkpointRevertTurn: (arg: { threadId: string; turn: number }): Promise<{ ok: boolean; turn: number; error?: string | null; warning?: string | null }> => ipcRenderer.invoke('checkpoint:revertTurn', arg),
  checkpointForkTurn: (arg: { threadId: string; turn: number; branch?: string; label?: string; worktree?: boolean }): Promise<{ ok: boolean; branch?: string; fileRef?: string | null; thread?: Thread | null; worktreePath?: string | null; worktreeError?: string | null; error?: string }> => ipcRenderer.invoke('checkpoint:forkTurn', arg),
  // Cross-engine handoff (VST-036): continue this conversation on a different engine.
  handoffThread: (arg: { threadId: string; toProviderId: string; toModel?: string | null; autoStart?: boolean; intent?: 'run-plan' }): Promise<Thread> => ipcRenderer.invoke('thread:handoff', arg),
  previewWithheld: (arg: { text: string; terms?: string[] }): Promise<{ text: string; withheld: string; total: number }> => ipcRenderer.invoke('guard:previewWithheld', arg),

  onAgentEvent: (cb: (e: AgentEvent) => void) => {
    const handler = (_: unknown, e: AgentEvent) => cb(e)
    ipcRenderer.on('agent:event', handler)
    return () => ipcRenderer.removeListener('agent:event', handler)
  },

  startLoop: (input: LoopStartInput): Promise<LoopRecord> => ipcRenderer.invoke('loop:start', input),
  listLoops: (): Promise<LoopRecord[]> => ipcRenderer.invoke('loop:list'),
  stopLoop: (id: string): Promise<{ stopped: boolean }> => ipcRenderer.invoke('loop:stop', id),
  onLoopUpdate: (cb: (rec: LoopRecord) => void) => {
    const handler = (_: unknown, rec: LoopRecord) => cb(rec)
    ipcRenderer.on('loop:update', handler)
    return () => ipcRenderer.removeListener('loop:update', handler)
  },

  // Live orchestration lifecycle push (additive — missions/rooms/jobs still snapshot-poll
  // as the reconciling fallback). Mirrors onLoopUpdate: subscribe to main's pushes, return
  // an unsubscribe.
  onOrchestrationEvent: (cb: (event: OrchestrationEvent) => void) => {
    const handler = (_: unknown, event: OrchestrationEvent) => cb(event)
    ipcRenderer.on('orchestration:event', handler)
    return () => ipcRenderer.removeListener('orchestration:event', handler)
  },

  onSshStatus: (cb: (s: SshConnectionState) => void) => {
    const handler = (_: unknown, s: SshConnectionState) => cb(s)
    ipcRenderer.on('ssh:status', handler)
    return () => ipcRenderer.removeListener('ssh:status', handler)
  },
  // A run notification was clicked: show that thread.
  onOpenThread: (cb: (threadId: string) => void) => {
    const handler = (_: unknown, threadId: string) => cb(threadId)
    ipcRenderer.on('app:open-thread', handler)
    return () => ipcRenderer.removeListener('app:open-thread', handler)
  },
  onPersistenceFailure: (cb: (failure: PersistenceFailure) => void) => {
    const handler = (_: unknown, failure: PersistenceFailure) => cb(failure)
    ipcRenderer.on('host:persistence-failure', handler)
    return () => ipcRenderer.removeListener('host:persistence-failure', handler)
  },
  onSshSecretPrompt: (cb: (p: SshSecretPrompt) => void) => {
    const handler = (_: unknown, p: SshSecretPrompt) => cb(p)
    ipcRenderer.on('ssh:secret-prompt', handler)
    return () => ipcRenderer.removeListener('ssh:secret-prompt', handler)
  },
  onSshTrustPrompt: (cb: (p: SshTrustPrompt) => void) => {
    const handler = (_: unknown, p: SshTrustPrompt) => cb(p)
    ipcRenderer.on('ssh:trust-prompt', handler)
    return () => ipcRenderer.removeListener('ssh:trust-prompt', handler)
  },

  // projects & multi-agent rooms
  getProjects: (): Promise<Project[]> => ipcRenderer.invoke('projects:get'),
  createProject: (opts: Partial<Project> & { name: string }): Promise<Project> => ipcRenderer.invoke('projects:create', opts),
  updateProject: (id: string, patch: Partial<Project>): Promise<Project> => ipcRenderer.invoke('projects:update', { id, patch }),
  setProjectExecutionTarget: (id: string, executionTarget: Project['executionTarget'], orchestratorId?: string | null, configPatch?: Partial<Project>): Promise<Project> => ipcRenderer.invoke('projects:setExecutionTarget', { id, executionTarget, orchestratorId, configPatch }),
  renameProject: (id: string, name: string): Promise<void> => ipcRenderer.invoke('projects:rename', { id, name }),
  deleteProject: (id: string): Promise<void> => ipcRenderer.invoke('projects:delete', id),
  onProjectUpdate: (cb: (p: Project) => void) => {
    const handler = (_: unknown, p: Project) => cb(p)
    ipcRenderer.on('project:update', handler)
    return () => ipcRenderer.removeListener('project:update', handler)
  },
  onThreadUpdate: (cb: (t: Thread) => void) => {
    const handler = (_: unknown, t: Thread) => cb(t)
    ipcRenderer.on('thread:update', handler)
    return () => ipcRenderer.removeListener('thread:update', handler)
  },
  getRooms: (): Promise<Room[]> => ipcRenderer.invoke('rooms:get'),
  createRoom: (projectId: string, name: string): Promise<Room> => ipcRenderer.invoke('rooms:create', { projectId, name }),
  updateRoom: (id: string, patch: Partial<Room>): Promise<Room> => ipcRenderer.invoke('rooms:update', { id, patch }),
  renameRoom: (id: string, name: string): Promise<void> => ipcRenderer.invoke('rooms:rename', { id, name }),
  deleteRoom: (id: string): Promise<void> => ipcRenderer.invoke('rooms:delete', id),
  onRoomUpdate: (cb: (r: Room) => void) => {
    const handler = (_: unknown, r: Room) => cb(r)
    ipcRenderer.on('room:update', handler)
    return () => ipcRenderer.removeListener('room:update', handler)
  },
  startRoom: (roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }> => ipcRenderer.invoke('room:start', { roomId, problem, attachments }),
  steerRoom: (roomId: string, text: string): Promise<Room['events'][number]> => ipcRenderer.invoke('room:steer', { roomId, text }),
  mutateRoomMemory: (roomId: string, command: RoomManualMemoryCommand): Promise<Room> => ipcRenderer.invoke('room:manualMemory', { roomId, command }),
  cancelRoom: (roomId: string): Promise<void> => ipcRenderer.invoke('room:cancel', roomId),
  mergeRoomWorktree: (roomId: string): Promise<RoomMergeResult> => ipcRenderer.invoke('room:mergeWorktree', { roomId }),

  getTickets: (): Promise<Ticket[]> => ipcRenderer.invoke('tickets:get'),
  createTicket: (t: Partial<Ticket> & { projectId: string }): Promise<Ticket> => ipcRenderer.invoke('tickets:create', t),
  updateTicket: (id: string, patch: Partial<Ticket>): Promise<Ticket> => ipcRenderer.invoke('tickets:update', { id, patch }),
  deleteTicket: (id: string): Promise<void> => ipcRenderer.invoke('tickets:delete', id),
  generateTickets: (projectId: string, idea: string, opts?: TicketWorkOptions): Promise<Ticket[]> => ipcRenderer.invoke('tickets:generate', { projectId, idea, opts }),
  workTicket: (ticketId: string, opts?: TicketWorkOptions): Promise<Thread> => ipcRenderer.invoke('ticket:work', { ticketId, opts }),
  onTicketUpdate: (cb: (t: Ticket) => void) => {
    const handler = (_: unknown, t: Ticket) => cb(t)
    ipcRenderer.on('ticket:update', handler)
    return () => ipcRenderer.removeListener('ticket:update', handler)
  },
  getJobs: (): Promise<Job[]> => ipcRenderer.invoke('jobs:get'),
  retryJob: (id: string, reason?: string, nextWakeAt?: number | null, resumePlan?: string | null): Promise<Job> => ipcRenderer.invoke('jobs:retry', { id, reason, nextWakeAt, resumePlan }),
  resumeJob: (id: string, resumePlan?: string | null): Promise<Job> => ipcRenderer.invoke('jobs:resume', { id, resumePlan }),
  deadLetterJob: (id: string, reason?: string): Promise<Job> => ipcRenderer.invoke('jobs:deadLetter', { id, reason }),
  onJobUpdate: (cb: (j: Job) => void) => {
    const handler = (_: unknown, j: Job) => cb(j)
    ipcRenderer.on('job:update', handler)
    return () => ipcRenderer.removeListener('job:update', handler)
  },
  onRoomEvent: (cb: (e: RoomStreamEvent) => void) => {
    const handler = (_: unknown, e: RoomStreamEvent) => cb(e)
    ipcRenderer.on('room:event', handler)
    return () => ipcRenderer.removeListener('room:event', handler)
  },

  // automations
  getAutomations: (): Promise<Automation[]> => ipcRenderer.invoke('automations:get'),
  createAutomation: (a: Partial<Automation>): Promise<Automation> => ipcRenderer.invoke('automations:create', a),
  updateAutomation: (id: string, patch: Partial<Automation>): Promise<Automation> => ipcRenderer.invoke('automations:update', { id, patch }),
  deleteAutomation: (id: string): Promise<void> => ipcRenderer.invoke('automations:delete', id),
  runAutomation: (id: string): Promise<{ ok: boolean }> => ipcRenderer.invoke('automation:run', id),
  onAutomationUpdate: (cb: (a: Automation) => void) => {
    const handler = (_: unknown, a: Automation) => cb(a)
    ipcRenderer.on('automation:update', handler)
    return () => ipcRenderer.removeListener('automation:update', handler)
  },
  onMission: (cb: (m: Mission) => void) => {
    const handler = (_: unknown, m: Mission) => cb(m)
    ipcRenderer.on('mission:update', handler)
    return () => ipcRenderer.removeListener('mission:update', handler)
  },
  onIncident: (cb: (i: Incident) => void) => {
    const handler = (_: unknown, i: Incident) => cb(i)
    ipcRenderer.on('incident:update', handler)
    return () => ipcRenderer.removeListener('incident:update', handler)
  },

  // skills & custom commands
  getSkills: (): Promise<Skill[]> => ipcRenderer.invoke('skills:get'),
  // VST-010: repo-authored project skills (hash-pinned trust) + suggestion diffs.
  getProjectSkills: (projectId: string): Promise<ProjectSkillView[]> => ipcRenderer.invoke('skills:project', projectId),
  trustProjectSkill: (input: { projectId: string; skillId: string; trusted: boolean }): Promise<ProjectSkillView[]> => ipcRenderer.invoke('skills:trustProjectSkill', input),
  suggestionDiff: (id: string): Promise<SkillSuggestionDiff> => ipcRenderer.invoke('suggestions:diff', id),
  // First-class artifacts (VST-011).
  listArtifacts: (threadId?: string): Promise<Artifact[]> => ipcRenderer.invoke('artifacts:list', threadId),
  previewArtifact: (id: string): Promise<ArtifactPreview> => ipcRenderer.invoke('artifacts:preview', id),
  redactArtifact: (id: string): Promise<Artifact> => ipcRenderer.invoke('artifacts:redact', id),
  exportArtifact: (id: string): Promise<{ artifact: Artifact; content: string | null }> => ipcRenderer.invoke('artifacts:export', id),
  deleteArtifact: (id: string): Promise<void> => ipcRenderer.invoke('artifacts:delete', id),
  // Observer hooks health (VST-012): per-hook delivery/failure/disabled + recent ring.
  getObserverStatus: (): Promise<{ hooks: Array<{ name: string; delivered: number; failed: number; consecutiveFailures: number; disabled: boolean }>; recent: Array<{ kind: string; at: number; threadId?: string | null; runId?: string | null; payload: Record<string, unknown> }> }> => ipcRenderer.invoke('observers:status'),
  saveSkill: (s: Partial<Skill>): Promise<Skill> => ipcRenderer.invoke('skills:save', s),
  deleteSkill: (id: string): Promise<void> => ipcRenderer.invoke('skills:delete', id),
  getCommands: (): Promise<Command[]> => ipcRenderer.invoke('commands:get'),
  saveCommand: (c: Partial<Command>): Promise<Command> => ipcRenderer.invoke('commands:save', c),
  deleteCommand: (id: string): Promise<void> => ipcRenderer.invoke('commands:delete', id),
  replyToThread: (threadKey: string, text: string, inReplyTo?: string): Promise<{ ok: boolean; error?: string; sentId?: string; continuationIds?: string[]; chunkCount?: number }> => ipcRenderer.invoke('connector:reply', { threadKey, text, inReplyTo }),
  requestChannelApproval: (threadKey: string, request: ChannelApprovalRequest): Promise<ChannelApprovalResult> => ipcRenderer.invoke('connector:requestApproval', { threadKey, request }),
  getSuggestions: (): Promise<SkillSuggestion[]> => ipcRenderer.invoke('suggestions:get'),
  acceptSuggestion: (id: string): Promise<{ ok: boolean; skill: Skill | null; command: Command | null }> => ipcRenderer.invoke('suggestions:accept', id),
  dismissSuggestion: (id: string): Promise<void> => ipcRenderer.invoke('suggestions:dismiss', id),
  onSuggestionsUpdate: (cb: (list: SkillSuggestion[]) => void) => { const h = (_e: unknown, list: SkillSuggestion[]) => cb(list); ipcRenderer.on('suggestions:update', h); return () => ipcRenderer.removeListener('suggestions:update', h) },
  getProjectMemory: (projectId: string): Promise<ProjectMemoryEntry[]> => ipcRenderer.invoke('memory:get', projectId),
  getProjectMemoryCandidates: (projectId: string): Promise<ProjectMemoryCandidate[]> => ipcRenderer.invoke('memory:candidates', projectId),
  proposeProjectMemory: (m: Partial<ProjectMemoryCandidate>): Promise<ProjectMemoryCandidate> => ipcRenderer.invoke('memory:propose', m),
  approveProjectMemoryCandidate: (id: string): Promise<ProjectMemoryEntry> => ipcRenderer.invoke('memory:approveCandidate', id),
  rejectProjectMemoryCandidate: (id: string): Promise<ProjectMemoryCandidate> => ipcRenderer.invoke('memory:rejectCandidate', id),
  // VST-009 reviewed-memory loop: what an approval would DO, and the way back out.
  memoryCandidateDiff: (id: string): Promise<ProjectMemoryCandidateDiff> => ipcRenderer.invoke('memory:candidateDiff', id),
  rollbackMemoryEntry: (id: string): Promise<{ action: 'restored_previous' | 'removed'; entry: ProjectMemoryEntry | null }> => ipcRenderer.invoke('memory:rollbackEntry', id),
  saveProjectMemory: (m: Partial<ProjectMemoryEntry>): Promise<ProjectMemoryEntry> => ipcRenderer.invoke('memory:save', m),
  deleteProjectMemory: (id: string): Promise<void> => ipcRenderer.invoke('memory:delete', id),
  getWebhooks: (): Promise<WebhookSnapshot> => ipcRenderer.invoke('webhooks:get'),
  saveWebhook: (w: Partial<WebhookEndpoint>): Promise<WebhookSnapshot> => ipcRenderer.invoke('webhooks:save', w),
  approveWebhook: (id: string): Promise<WebhookSnapshot> => ipcRenderer.invoke('webhooks:approve', id),
  deleteWebhook: (id: string): Promise<WebhookSnapshot> => ipcRenderer.invoke('webhooks:delete', id),
  testWebhook: (id: string): Promise<WebhookSnapshot> => ipcRenderer.invoke('webhooks:test', id),
  // Phase 7 — run export & sharing
  buildExport: (threadId: string, opts?: ExportOptions): Promise<ExportResult> => ipcRenderer.invoke('export:build', threadId, opts),
  saveExport: (threadId: string, opts?: ExportOptions): Promise<ExportSaveResult> => ipcRenderer.invoke('export:save', threadId, opts),
  importBundle: (): Promise<ExportImportResult> => ipcRenderer.invoke('export:importBundle'),

  // Phase 6 — production incident mode
  getIncidents: (): Promise<Incident[]> => ipcRenderer.invoke('incidents:get'),
  createIncident: (input: CreateIncidentInput): Promise<Incident> => ipcRenderer.invoke('incidents:create', input),
  updateIncident: (id: string, patch: Partial<Incident>): Promise<Incident> => ipcRenderer.invoke('incidents:update', id, patch),
  deleteIncident: (id: string): Promise<void> => ipcRenderer.invoke('incidents:delete', id),
  addIncidentEvent: (id: string, kind: IncidentEventKind, text: string): Promise<Incident> => ipcRenderer.invoke('incidents:addEvent', id, kind, text),
  proposeMitigation: (id: string, text: string, rationale?: string): Promise<Incident> => ipcRenderer.invoke('incidents:proposeMitigation', id, text, rationale),
  decideMitigation: (id: string, mitigationId: string, decision: IncidentMitigation['status']): Promise<Incident> => ipcRenderer.invoke('incidents:decideMitigation', id, mitigationId, decision),
  executeMitigation: (id: string, mitigationId: string): Promise<Incident> => ipcRenderer.invoke('incidents:executeMitigation', id, mitigationId),
  scanIncident: (id: string): Promise<Incident> => ipcRenderer.invoke('incidents:scan', id),
  generatePostmortem: (id: string): Promise<Incident> => ipcRenderer.invoke('incidents:postmortem', id),

  listClis: (): Promise<CliDefinition[]> => ipcRenderer.invoke('terminal:listClis'),
  listCliTools: (): Promise<CliToolStatus[]> => ipcRenderer.invoke('terminal:listCliTools'),
  startTerminal: (projectId: string, cliId: string, opts?: { command?: string; label?: string }): Promise<CliSession> => ipcRenderer.invoke('terminal:start', { projectId, cliId, command: opts?.command, label: opts?.label }),
  writeTerminal: (sessionId: string, data: string): Promise<void> => ipcRenderer.invoke('terminal:write', { sessionId, data }),
  resizeTerminal: (sessionId: string, cols: number, rows: number): Promise<void> => ipcRenderer.invoke('terminal:resize', { sessionId, cols, rows }),
  killTerminal: (sessionId: string): Promise<void> => ipcRenderer.invoke('terminal:kill', { sessionId }),
  renameTerminal: (sessionId: string, title: string): Promise<void> => ipcRenderer.invoke('terminal:rename', { sessionId, title }),
  listTerminals: (projectId: string): Promise<CliSession[]> => ipcRenderer.invoke('terminal:list', projectId),
  restorableTerminals: (projectId: string): Promise<CliSession[]> => ipcRenderer.invoke('terminal:restorable', projectId),
  restoreTerminal: (sessionId: string): Promise<CliSession> => ipcRenderer.invoke('terminal:restore', { sessionId }),
  terminalBacklog: (sessionId: string): Promise<string> => ipcRenderer.invoke('terminal:backlog', sessionId),
  // Sequenced gap-fill (VST-039): reconnect/paired clients resume output from a cursor.
  terminalSince: (sessionId: string, since?: number): Promise<{ chunks: { seq: number; data: string }[]; gap: boolean; head: number }> => ipcRenderer.invoke('terminal:since', { sessionId, since }),
  onTerminalData: (cb: (p: { sessionId: string; chunk: string; orchestratorHostId?: string | null }) => void) => { const h = (_e: unknown, p: { sessionId: string; chunk: string; orchestratorHostId?: string | null }) => cb(p); ipcRenderer.on('terminal:data', h); return () => ipcRenderer.removeListener('terminal:data', h) },
  onTerminalStatus: (cb: (p: { sessionId: string; status: CliSession['status']; orchestratorHostId?: string | null }) => void) => { const h = (_e: unknown, p: { sessionId: string; status: CliSession['status']; orchestratorHostId?: string | null }) => cb(p); ipcRenderer.on('terminal:status', h); return () => ipcRenderer.removeListener('terminal:status', h) },
  onTerminalSessions: (cb: (p: { projectId: string; sessions: CliSession[]; orchestratorHostId?: string | null }) => void) => { const h = (_e: unknown, p: { projectId: string; sessions: CliSession[]; orchestratorHostId?: string | null }) => cb(p); ipcRenderer.on('terminal:sessions', h); return () => ipcRenderer.removeListener('terminal:sessions', h) },

  // window controls (used on non-mac or custom buttons)
  windowAction: (action: 'minimize' | 'maximize' | 'close'): Promise<void> => ipcRenderer.invoke('window:action', action),

  // Built-in TensorFold engine: status, installed models, Hugging Face search and downloads.
  engineSnapshot: (): Promise<EngineSnapshot> => ipcRenderer.invoke('engine:snapshot'),
  engineConnection: (): Promise<EngineConnection | null> => ipcRenderer.invoke('engine:connection'),
  engineRefresh: (): Promise<EngineSnapshot> => ipcRenderer.invoke('engine:refresh'),
  engineLibrary: (): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:library'),
  engineScanLibrary: (options?: EngineLibraryScanOptions): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:scanLibrary', options),
  engineAddLibraryFolder: (): Promise<EngineModelLibrarySnapshot | null> => ipcRenderer.invoke('engine:addLibraryFolder'),
  engineRemoveLibraryFolder: (id: string): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:removeLibraryFolder', id),
  engineCancelLibraryScan: (): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:cancelLibraryScan'),
  engineSearch: (query: string): Promise<HubModelSummary[]> => ipcRenderer.invoke('engine:search', query),
  enginePreflight: (repo: string, revision?: string | null): Promise<ModelPreflight> => ipcRenderer.invoke('engine:preflight', { repo, revision }),
  enginePull: (repo: string, revision?: string | null): Promise<PullProgress> => ipcRenderer.invoke('engine:pull', { repo, revision }),
  engineCancelPull: (repo: string): Promise<{ cancelled: boolean }> => ipcRenderer.invoke('engine:cancelPull', repo),
  engineLoad: (repo: string): Promise<EngineStatus> => ipcRenderer.invoke('engine:load', repo),
  engineUnload: (): Promise<EngineStatus> => ipcRenderer.invoke('engine:unload'),
  engineSetCortheon: (enabled: boolean): Promise<EngineStatus> => ipcRenderer.invoke('engine:setCortheon', enabled),
  engineLearning: (): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:learning'),
  engineSetLearning: (enabled: boolean, model?: string): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:setLearning', { enabled, model }),
  engineTeach: (text: string): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:teach', text),
  engineCancelTeach: (): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:cancelTeach'),
  engineDelete: (repo: string): Promise<EngineInstalledModel[]> => ipcRenderer.invoke('engine:delete', repo),
  onEngineEvent: (cb: (event: EngineEvent) => void) => { const h = (_e: unknown, event: EngineEvent) => cb(event); ipcRenderer.on('engine:event', h); return () => ipcRenderer.removeListener('engine:event', h) },
}

contextBridge.exposeInMainWorld('tensorfoldStudio', api)
// Compatibility aliases for existing integrations.
contextBridge.exposeInMainWorld('vontra', api)

export type TensorFoldStudioApi = typeof api
import type { ProviderGenerationProbe } from './agent/provider-generation-probe'
