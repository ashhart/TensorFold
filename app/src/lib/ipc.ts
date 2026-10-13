import type { AgentEvent, ArchiveKind, ArchiveSnapshot, Attachment, CueRunPayload, CueRunResult, Settings, StartRunPayload, Thread, RendererThreadPatch, Project, Room, RoomStreamEvent, Ticket, TicketRemoteLifecycleCommand, TicketWorkOptions, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitRepositoryState, GitStatus, TokenUsageSnapshot, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, ChannelApprovalRequest, ChannelApprovalResult, AppUpdateStatus, Provider, OmlxDiscovery, ProviderDiscovery, ProviderDiscoveryKind, ResolvedInstructionGraph, ProjectMemoryCandidate, ProjectMemoryCandidateDiff, ProjectMemoryEntry, WebhookEndpoint, WebhookSnapshot, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, EmailConnectorSettings, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSummaryResult, EmailSyncResult, EmailThreadResult, Skill, ProjectSkillView, SkillSuggestionDiff, Artifact, ArtifactPreview, Command, SkillSuggestion, McpServerConfig, McpSnapshot, McpToolCallResult, BotControlPollResult, ControllerSnapshot, ControllerStatus, RemoteControlPairingInfo, RemoteControlStatus, RemoteConnectionTestResult, RemoteDirectoryListResult, RemoteOrchestratorSnapshotResult, PersistenceFailure, SecurityControlSnapshot, SecurityPurgeResult, SshHost, SshConnectionState, SshSecretPrompt, SshTrustPrompt, WorkspaceLocation, CliDefinition, CliSession, CliToolStatus, ImportableSession, ImportSessionSource, SpineAggregate, ReadSinceResult, CommandReceipt, TurnCheckpoint, PersistedProviderStatus } from './types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarConnectorSettings, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DocumentConnectorSettings, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, IMessageItem, IMessageReadResult, IMessageSyncResult, WhatsAppBridgeConnectorSettings, WhatsAppBridgeSource, DiscordConnectorSettings, DiscordChannel, DiscordSource, MattermostConnectorSettings, MattermostChannel, MattermostSource, MatrixConnectorSettings, MatrixRoom, MatrixSource, GoogleChatConnectorSettings, GoogleChatSpace, GoogleChatSource, SlackConnectorSettings, SlackListOptions, SlackMessage, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from './types'
// Phase 6/7 (incident mode + run export) — additive import line alongside the Codex lane.
import type { Incident, CreateIncidentInput, IncidentEventKind, IncidentMitigation, ExportOptions, ExportResult, ExportSaveResult, ExportImportResult } from './types'
import type { Readiness, CliLoginResult } from './types'
import type { LoopRecord, LoopStartInput } from './types'
import type { LocalTtsDiscovery, VoiceSynthesisResult, VoiceTranscriptionInput, VoiceTranscriptionResult } from './types'
import type { OrchestrationEvent } from './types'
import type { RoomManualMemoryCommand, RoomMergeResult } from './types'
import type { EngineConnection, EngineEvent, EngineInstalledModel, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress } from './types'
import type { MobileBridgePairing, MobileBridgeStatus } from '../../electron/agent/types/mobile-bridge'

export interface TensorFoldStudioApi {
  platform: string
  getSecurityControls(): Promise<SecurityControlSnapshot>
  purgeLocalMemory(): Promise<SecurityPurgeResult>
  writeClipboard(text: string): Promise<{ ok: boolean }>
  getSettings(): Promise<Settings>
  saveSettings(s: Settings): Promise<Settings>
  revokeApprovalRule(id: string): Promise<Settings>
  getRemoteStatus(): Promise<RemoteControlStatus>
  getMobileBridgeStatus(): Promise<MobileBridgeStatus>
  configureMobileBridge(input: { enabled: boolean; host: string; port: number }): Promise<MobileBridgeStatus>
  issueMobilePairing(label?: string): Promise<MobileBridgePairing>
  getRemotePairingInfo(): Promise<RemoteControlPairingInfo>
  regenerateRemoteToken(): Promise<RemoteControlPairingInfo>
  // Provider rate-limit status (VST-037): persisted per-provider windows for the picker + doctor.
  getProviderStatus(): Promise<PersistedProviderStatus>
  onProviderStatus(cb: (s: PersistedProviderStatus) => void): () => void
  onUsageCleared(cb: () => void): () => void
  // Per-device pairing (VST-035): one-time credential → device session; list/revoke devices.
  issueDevicePairing(label?: string): Promise<{ credential: string; expiresAt: number; url: string | null }>
  listDeviceSessions(): Promise<Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>>
  revokeDeviceSession(sessionId: string): Promise<{ revoked: boolean }>
  testRemoteClient(input?: Partial<Settings['remoteClient']>): Promise<RemoteConnectionTestResult>
  getRemoteSnapshot(hostId?: string | null): Promise<RemoteOrchestratorSnapshotResult>
  getRemoteOrchestratorArchives(hostId: string | null | undefined): Promise<ArchiveSnapshot>
  listImportableSessions(): Promise<ImportableSession[]>
  importExternalSession(input: { sessionId: string; source: ImportSessionSource; title?: string; cwd?: string | null; model?: string | null }): Promise<Thread>
  spineReadSince(input: { since?: number; aggregate?: SpineAggregate; streamId?: string; limit?: number }): Promise<ReadSinceResult>
  spineReceipt(commandId: string): Promise<CommandReceipt | null>
  spineAdvanceCursor(input: { projector: string; sequence: number }): Promise<number>
  listRemoteOrchestratorDir(hostId: string | null | undefined, dir?: string | null): Promise<RemoteDirectoryListResult>
  createRemoteOrchestratorProject(hostId: string | null | undefined, project: Partial<Project> & { name?: string; cwd?: string }): Promise<Project>
  updateRemoteOrchestratorProject(hostId: string | null | undefined, id: string, patch: Partial<Project>): Promise<Project>
  deleteRemoteOrchestratorProject(hostId: string | null | undefined, id: string): Promise<Project | null>
  restoreRemoteOrchestratorProject(hostId: string | null | undefined, id: string): Promise<Project | null>
  createRemoteOrchestratorThread(hostId: string | null | undefined, input: { cwd?: string | null; kind?: 'task' | 'chat'; projectId?: string | null }): Promise<Thread>
  archiveRemoteOrchestratorThread(hostId: string | null | undefined, id: string): Promise<Thread | null>
  restoreRemoteOrchestratorThread(hostId: string | null | undefined, id: string): Promise<Thread | null>
  remoteListClis(hostId: string | null | undefined): Promise<CliDefinition[]>
  remoteTerminalStart(hostId: string | null | undefined, input: { projectId: string; cliId: string; command?: string; label?: string }): Promise<CliSession>
  remoteTerminalWrite(hostId: string | null | undefined, sessionId: string, data: string): Promise<void>
  remoteTerminalResize(hostId: string | null | undefined, sessionId: string, cols: number, rows: number): Promise<void>
  remoteTerminalRename(hostId: string | null | undefined, sessionId: string, title: string): Promise<void>
  remoteTerminalKill(hostId: string | null | undefined, sessionId: string): Promise<void>
  remoteTerminalList(hostId: string | null | undefined, projectId: string): Promise<CliSession[]>
  approveRemoteOrchestrator(hostId: string | null | undefined, callId: string, decision: 'allow' | 'allow_always' | 'deny'): Promise<void>
  startRemoteOrchestratorThread(hostId: string | null | undefined, payload: StartRunPayload): Promise<{ runId: string }>
  guideRemoteOrchestratorThread(hostId: string | null | undefined, payload: CueRunPayload): Promise<CueRunResult>
  cancelRemoteOrchestratorThread(hostId: string | null | undefined, threadId: string): Promise<void>
  createRemoteOrchestratorRoom(hostId: string | null | undefined, projectId: string, name: string): Promise<Room>
  updateRemoteOrchestratorRoom(hostId: string | null | undefined, id: string, patch: Partial<Room>): Promise<Room>
  renameRemoteOrchestratorRoom(hostId: string | null | undefined, id: string, name: string): Promise<Room>
  steerRemoteOrchestratorRoom(hostId: string | null | undefined, roomId: string, text: string): Promise<Room['events'][number]>
  mutateRemoteOrchestratorRoomMemory(hostId: string | null | undefined, roomId: string, command: RoomManualMemoryCommand): Promise<Room>
  deleteRemoteOrchestratorRoom(hostId: string | null | undefined, id: string): Promise<Room | null>
  restoreRemoteOrchestratorRoom(hostId: string | null | undefined, id: string): Promise<Room | null>
  startRemoteOrchestratorRoom(hostId: string | null | undefined, roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }>
  cancelRemoteOrchestratorRoom(hostId: string | null | undefined, roomId: string): Promise<void>
  createRemoteOrchestratorTicket(hostId: string | null | undefined, ticket: Partial<Ticket> & { projectId: string }): Promise<Ticket>
  startRemoteOrchestratorTicketExecution(hostId: string | null | undefined, id: string, userText: string): Promise<{ ticket: Ticket; thread: Thread; runId: string }>
  updateRemoteOrchestratorTicket(hostId: string | null | undefined, id: string, patch: Partial<Ticket>): Promise<Ticket>
  transitionRemoteOrchestratorTicket(hostId: string | null | undefined, id: string, command: TicketRemoteLifecycleCommand): Promise<Ticket>
  deleteRemoteOrchestratorTicket(hostId: string | null | undefined, id: string): Promise<Ticket | null>
  restoreRemoteOrchestratorTicket(hostId: string | null | undefined, id: string): Promise<Ticket | null>
  // SSH remote workspaces
  listSshHosts(): Promise<SshHost[]>
  sshStatus(hostId: string): Promise<SshConnectionState>
  saveSshHost(host: SshHost): Promise<SshHost[]>
  removeSshHost(hostId: string): Promise<SshHost[]>
  testSshHost(hostId: string): Promise<{ ok: boolean; message: string }>
  connectSshHost(hostId: string): Promise<SshConnectionState>
  disconnectSshHost(hostId: string): Promise<{ ok: boolean }>
  resolveSshSecret(promptId: string, value: string | null): Promise<{ ok: boolean }>
  resolveSshTrust(promptId: string, ok: boolean): Promise<{ ok: boolean }>
  listRemoteDir(hostId: string, dir: string): Promise<Array<{ name: string; isDirectory: boolean }>>
  getThreads(): Promise<Thread[]>
  getArchives(): Promise<ArchiveSnapshot>
  archiveItem(kind: ArchiveKind, id: string): Promise<ArchiveSnapshot>
  restoreItem(kind: ArchiveKind, id: string): Promise<ArchiveSnapshot>
  // commandId (optional) makes the create exactly-once through the event spine: a retried
  // call with the same id returns the SAME thread instead of creating a duplicate.
  createThread(cwd: string, kind?: 'task' | 'chat', projectId?: string | null, commandId?: string): Promise<Thread>
  setThreadProject(id: string, projectId: string | null): Promise<void>
  deleteThread(id: string): Promise<void>
  renameThread(id: string, title: string): Promise<void>
  setThreadEngine(id: string, engine: 'native' | 'omp'): Promise<void>
  ompStatus(): Promise<{ available: boolean; binary: string; version?: string }>
  configureThread(id: string, providerId: string, model: string, providerAccountId?: string | null): Promise<void>
  patchThread(id: string, patch: RendererThreadPatch): Promise<Thread>
  setThreadWorkspace(id: string, workspace: WorkspaceLocation | null, cwd?: string | null): Promise<Thread>
  appendThreadNotice(id: string, text: string): Promise<void>
  compactThread(id: string): Promise<{ ok: boolean; thread: Thread }>
  startRun(p: StartRunPayload): Promise<{ runId: string }>
  cueRun(p: CueRunPayload): Promise<CueRunResult>
  cancelRun(threadId: string): Promise<void>
  approve(callId: string, decision: 'allow' | 'allow_always' | 'deny'): Promise<void>
  pickDirectory(): Promise<string | null>
  fetchModels(provider: Provider): Promise<string[]>
  getModelReadiness(input: { providerId?: string | null; accountId?: string | null }): Promise<Readiness>
  loginCliProvider(input: { providerId: string; accountId?: string | null }): Promise<CliLoginResult>
  cancelCliLogin(providerId: string): Promise<{ cancelled: boolean }>
  checkCliTransport(input: { transport: 'codex-cli' | 'claude-cli'; cliPath?: string }): Promise<{ loggedIn: boolean; binaryPresent: boolean; detail?: string; message?: string }>
  getWorktrees(arg: { cwd: string; workspace?: WorkspaceLocation | null }): Promise<{ branches: string[]; worktrees: Array<{ path: string; head: string | null; branch: string | null; detached: boolean; bare: boolean }>; current: string | null }>
  checkoutBranch(arg: { cwd: string; workspace?: WorkspaceLocation | null; branch: string }): Promise<{ ok: boolean; error?: string }>
  createWorktree(arg: { cwd: string; workspace?: WorkspaceLocation | null; name: string; mode: 'worktree' | 'branch' }): Promise<{ ok: boolean; branch: string; mode: 'worktree' | 'branch'; path: string; error?: string }>
  startLoop(input: LoopStartInput): Promise<LoopRecord>
  listLoops(): Promise<LoopRecord[]>
  stopLoop(id: string): Promise<{ stopped: boolean }>
  onLoopUpdate(cb: (rec: LoopRecord) => void): () => void
  onOrchestrationEvent(cb: (event: OrchestrationEvent) => void): () => void
  discoverLocalProvider(kind: ProviderDiscoveryKind, baseUrl?: string, apiKey?: string): Promise<ProviderDiscovery>
  discoverOmlxProvider(): Promise<OmlxDiscovery>
  synthesizeVoice(text: string): Promise<VoiceSynthesisResult>
  synthesizeLocalVoice(text: string): Promise<VoiceSynthesisResult>
  transcribeVoice(input: VoiceTranscriptionInput): Promise<VoiceTranscriptionResult>
  discoverLocalTts(baseUrl?: string): Promise<LocalTtsDiscovery>
  getTokenUsage(): Promise<TokenUsageSnapshot>
  getContextPressure(): Promise<ContextPressureSnapshot>
  getModelRouterSnapshot(): Promise<ModelRouterSnapshot>
  getAssistantSnapshot(): Promise<AssistantControlSnapshot>
  recordConnectorInteraction(input: ConnectorInteractionInput): Promise<ConnectorEvent>
  resolveConnectorApproval(eventId: string, decision: 'approve' | 'deny'): Promise<ConnectorEvent>
  getControllerStatus(): Promise<ControllerStatus>
  enableController(): Promise<ControllerStatus>
  captureController(): Promise<ControllerSnapshot>
  getMcpSnapshot(): Promise<McpSnapshot>
  saveMcpServer(server: Partial<McpServerConfig>): Promise<McpSnapshot>
  deleteMcpServer(id: string): Promise<McpSnapshot>
  connectMcpServer(id: string): Promise<McpSnapshot>
  disconnectMcpServer(id: string): Promise<McpSnapshot>
  refreshMcpServer(id?: string): Promise<McpSnapshot>
  testMcpTool(callName: string, args?: Record<string, any>): Promise<McpToolCallResult>
  addEmailSource(dir: string): Promise<EmailConnectorSettings>
  updateEmailSource(id: string, patch: Partial<EmailSourceConfig>): Promise<EmailConnectorSettings>
  removeEmailSource(id: string): Promise<EmailConnectorSettings>
  syncEmail(): Promise<EmailSyncResult>
  listEmail(opts?: EmailListOptions): Promise<EmailMessage[]>
  readEmail(id: string): Promise<EmailReadResult>
  readEmailThread(id: string): Promise<EmailThreadResult>
  summarizeEmail(opts?: EmailListOptions): Promise<EmailSummaryResult>
  composeEmail(input: EmailDraftInput): Promise<ConnectorEvent>
  resolveEmailThread(input: EmailResolveInput): Promise<ConnectorEvent>
  addSlackSource(dir: string): Promise<SlackConnectorSettings>
  updateSlackSource(id: string, patch: Partial<SlackSourceConfig>): Promise<SlackConnectorSettings>
  removeSlackSource(id: string): Promise<SlackConnectorSettings>
  syncSlack(): Promise<SlackSyncResult>
  listSlack(opts?: SlackListOptions): Promise<SlackMessage[]>
  readSlack(id: string): Promise<SlackReadResult>
  summarizeSlack(opts?: SlackListOptions): Promise<SlackSummaryResult>
  verifySlack(id: string): Promise<{ ok: boolean; team?: string; bot?: string; error?: string }>
  listSlackChannels(id: string): Promise<{ id: string; name: string; isPrivate: boolean }[]>
  sendSlack(sourceId: string, channel: string, text: string, threadTs?: string): Promise<{ ok: boolean; ts?: string; threadTs?: string }>
  addDiscordSource(label: string): Promise<DiscordConnectorSettings>
  removeDiscordSource(id: string): Promise<DiscordConnectorSettings>
  verifyDiscord(id: string): Promise<{ ok: boolean; bot?: string; guilds?: { id: string; name: string }[]; error?: string }>
  listDiscordChannels(id: string): Promise<DiscordChannel[]>
  syncDiscord(id: string): Promise<{ messages: number; events: number }>
  sendDiscord(sourceId: string, channelId: string, text: string, replyTo?: string): Promise<{ ok: boolean; id?: string }>
  addMattermostSource(label: string): Promise<MattermostConnectorSettings>
  updateMattermostSource(id: string, patch: Partial<MattermostSource>): Promise<MattermostConnectorSettings>
  removeMattermostSource(id: string): Promise<MattermostConnectorSettings>
  verifyMattermost(id: string): Promise<{ ok: boolean; user?: string; userId?: string; error?: string }>
  listMattermostChannels(id: string): Promise<MattermostChannel[]>
  syncMattermost(id: string): Promise<{ messages: number; events: number }>
  sendMattermost(sourceId: string, channelId: string, text: string, replyTo?: string): Promise<{ ok: boolean; id?: string }>
  addMatrixSource(label: string): Promise<MatrixConnectorSettings>
  updateMatrixSource(id: string, patch: Partial<MatrixSource>): Promise<MatrixConnectorSettings>
  removeMatrixSource(id: string): Promise<MatrixConnectorSettings>
  verifyMatrix(id: string): Promise<{ ok: boolean; userId?: string; error?: string }>
  listMatrixChannels(id: string): Promise<MatrixRoom[]>
  syncMatrix(id: string): Promise<{ messages: number; events: number }>
  sendMatrix(sourceId: string, roomId: string, text: string, replyTo?: string, threadRootId?: string): Promise<{ ok: boolean; id?: string }>
  addGoogleChatSource(label: string): Promise<GoogleChatConnectorSettings>
  updateGoogleChatSource(id: string, patch: Partial<GoogleChatSource>): Promise<GoogleChatConnectorSettings>
  removeGoogleChatSource(id: string): Promise<GoogleChatConnectorSettings>
  verifyGoogleChat(id: string): Promise<{ ok: boolean; mode: 'webhook' | 'token'; space?: string | null; error?: string }>
  listGoogleChatChannels(id: string): Promise<GoogleChatSpace[]>
  syncGoogleChat(id: string): Promise<{ messages: number; events: number }>
  sendGoogleChat(sourceId: string, space: string, text: string, threadKey?: string, threadName?: string): Promise<{ ok: boolean; id?: string; threadName?: string | null }>
  sendIMessage(to: string, text: string): Promise<{ ok: boolean }>
  addCalendarSource(target: string): Promise<CalendarConnectorSettings>
  updateCalendarSource(id: string, patch: Partial<CalendarSourceConfig>): Promise<CalendarConnectorSettings>
  removeCalendarSource(id: string): Promise<CalendarConnectorSettings>
  syncCalendar(): Promise<CalendarSyncResult>
  listCalendar(opts?: CalendarListOptions): Promise<CalendarEvent[]>
  readCalendar(id: string): Promise<CalendarReadResult>
  summarizeCalendar(opts?: CalendarListOptions): Promise<CalendarSummaryResult>
  createCalendarEvent(input: CalendarMutationInput): Promise<ConnectorEvent>
  updateCalendarEvent(input: CalendarMutationInput & { id: string }): Promise<ConnectorEvent>
  addDocumentSource(target: string): Promise<DocumentConnectorSettings>
  updateDocumentSource(id: string, patch: Partial<DocumentSourceConfig>): Promise<DocumentConnectorSettings>
  removeDocumentSource(id: string): Promise<DocumentConnectorSettings>
  syncDocuments(): Promise<DocumentSyncResult>
  listDocuments(opts?: DocumentListOptions): Promise<DocumentItem[]>
  readDocument(id: string): Promise<DocumentReadResult>
  summarizeDocuments(opts?: DocumentListOptions): Promise<DocumentSummaryResult>
  addBrowserCurrentTabSource(input: BrowserSnapshotInput): Promise<BrowserConnectorSettings>
  addBrowserHistorySource(target: string): Promise<BrowserConnectorSettings>
  updateBrowserSource(id: string, patch: Partial<BrowserSourceConfig>): Promise<BrowserConnectorSettings>
  removeBrowserSource(id: string): Promise<BrowserConnectorSettings>
  syncBrowser(): Promise<BrowserSyncResult>
  listBrowser(opts?: BrowserListOptions): Promise<BrowserItem[]>
  readBrowser(id: string): Promise<BrowserReadResult>
  summarizeBrowser(opts?: BrowserListOptions): Promise<BrowserSummaryResult>
  imessageAccess(): Promise<{ state: string; detail: string }>
  syncImessage(): Promise<IMessageSyncResult>
  listImessage(): Promise<IMessageItem[]>
  readImessage(id: string): Promise<IMessageReadResult | null>
  openFullDiskAccess(): Promise<void>
  addWhatsAppSource(label: string): Promise<WhatsAppBridgeConnectorSettings>
  removeWhatsAppSource(id: string): Promise<WhatsAppBridgeConnectorSettings>
  startWhatsAppPairing(id: string): Promise<{ started: boolean }>
  disconnectWhatsApp(id: string): Promise<{ ok: boolean }>
  onWhatsAppQr(cb: (payload: { sourceId: string; qr: string }) => void): () => void
  onWhatsAppStatus(cb: (payload: { sourceId: string; paired?: boolean; lastError?: string | null }) => void): () => void
  pollTelegram(): Promise<BotControlPollResult>
  getMissionSnapshot(): Promise<MissionRuntimeSnapshot>
  inspectMission(id: string): Promise<MissionInspectionSnapshot>
  createMission(input: CreateMissionInput): Promise<Mission>
  createLocalPrLoop(input: CreatePrLoopInput): Promise<CreatePrLoopResult>
  setMissionStatus(id: string, status: MissionStatus, message?: string): Promise<Mission>
  recordMissionHeartbeat(id: string, message?: string): Promise<Mission>
  addMissionCheckpoint(id: string, checkpoint: Partial<MissionCheckpoint> & { label: string; summary: string }): Promise<Mission>
  appendMissionJournal(id: string, event: Partial<MissionJournalEvent> & { message: string }): Promise<Mission>
  requestMissionApproval(id: string, approval: Partial<MissionPendingApproval> & { callId: string; title: string; summary: string }): Promise<Mission>
  resolveMissionApproval(id: string, approvalId: string, decision: 'allow' | 'deny'): Promise<Mission>
  getUpdateStatus(): Promise<AppUpdateStatus>
  checkForUpdates(): Promise<AppUpdateStatus>
  openUpdateDownload(): Promise<{ ok: boolean }>
  downloadAppUpdate(): Promise<{ ok: boolean }>
  cancelAppUpdateDownload(): Promise<{ cancelled: boolean }>
  installAppUpdate(): Promise<{ ok: boolean }>
  onUpdateStatus(cb: (s: AppUpdateStatus) => void): () => void
  getBranch(cwd: string, workspace?: WorkspaceLocation | null): Promise<string | null>
  inspectGitRepository(cwd: string, workspace?: WorkspaceLocation | null): Promise<GitRepositoryState>
  initGitRepository(cwd: string, workspace?: WorkspaceLocation | null): Promise<GitRepositoryState>
  scanGitHygiene(cwd: string, message?: string, workspace?: WorkspaceLocation | null): Promise<GitHygieneReport>
  sanitizeGitHygiene(cwd: string, findings: GitHygieneFinding[], workspace?: WorkspaceLocation | null): Promise<GitHygieneSanitizeResult>
  listFiles(cwd: string, workspace?: WorkspaceLocation | null): Promise<string[]>
  resolveInstructions(cwd: string, providerId?: string | null, workspace?: WorkspaceLocation | null): Promise<ResolvedInstructionGraph>
  gitStatus(cwd: string, workspace?: WorkspaceLocation | null): Promise<GitStatus>
  gitDiff(cwd: string, paths?: string[], workspace?: WorkspaceLocation | null): Promise<GitDiffFile[]>
  revertFiles(cwd: string, paths: string[], workspace?: WorkspaceLocation | null): Promise<void>
  commit(cwd: string, message: string, workspace?: WorkspaceLocation | null): Promise<GitCommitResult>
  exec(cwd: string, command: string, workspace?: WorkspaceLocation | null): Promise<{ output: string; code: number }>
  checkpointList(arg: { threadId: string }): Promise<number[]>
  checkpointUndoLast(arg: { threadId: string }): Promise<{ ok: boolean; turn?: number; error?: string; warning?: string }>
  // Unified turn checkpoints (VST-036): files + conversation.
  checkpointTurns(arg: { threadId: string }): Promise<TurnCheckpoint[]>
  checkpointRevertTurn(arg: { threadId: string; turn: number }): Promise<{ ok: boolean; turn: number; error?: string | null; warning?: string | null }>
  checkpointForkTurn(arg: { threadId: string; turn: number; branch?: string; label?: string; worktree?: boolean }): Promise<{ ok: boolean; branch?: string; fileRef?: string | null; thread?: Thread | null; worktreePath?: string | null; worktreeError?: string | null; error?: string }>
  // Cross-engine handoff (VST-036): continue this conversation on a different engine.
  handoffThread(arg: { threadId: string; toProviderId: string; toModel?: string | null; autoStart?: boolean; intent?: 'run-plan' }): Promise<Thread>
  // Planning-only guard: what an external model would receive for `text` (Settings preview).
  previewWithheld(arg: { text: string; terms?: string[] }): Promise<{ text: string; withheld: string; total: number }>
  checkpointDiffLast(arg: { threadId: string }): Promise<{ ok: boolean; turn?: number; diff?: string; error?: string }>
  onAgentEvent(cb: (e: AgentEvent) => void): () => void
  onPersistenceFailure(cb: (failure: PersistenceFailure) => void): () => void
  onSshStatus(cb: (s: SshConnectionState) => void): () => void
  onSshSecretPrompt(cb: (p: SshSecretPrompt) => void): () => void
  onSshTrustPrompt(cb: (p: SshTrustPrompt) => void): () => void
  windowAction(a: 'minimize' | 'maximize' | 'close'): Promise<void>
  openExternal(url: string): Promise<void>

  getProjects(): Promise<Project[]>
  createProject(opts: Partial<Project> & { name: string }): Promise<Project>
  updateProject(id: string, patch: Partial<Project>): Promise<Project>
  setProjectExecutionTarget(id: string, executionTarget: Project['executionTarget'], orchestratorId?: string | null, configPatch?: Partial<Project>): Promise<Project>
  renameProject(id: string, name: string): Promise<void>
  deleteProject(id: string): Promise<void>
  onProjectUpdate(cb: (p: Project) => void): () => void
  onThreadUpdate(cb: (t: Thread) => void): () => void
  getRooms(): Promise<Room[]>
  createRoom(projectId: string, name: string): Promise<Room>
  updateRoom(id: string, patch: Partial<Room>): Promise<Room>
  renameRoom(id: string, name: string): Promise<void>
  deleteRoom(id: string): Promise<void>
  onRoomUpdate(cb: (r: Room) => void): () => void
  startRoom(roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }>
  steerRoom(roomId: string, text: string): Promise<Room['events'][number]>
  mutateRoomMemory(roomId: string, command: RoomManualMemoryCommand): Promise<Room>
  cancelRoom(roomId: string): Promise<void>
  mergeRoomWorktree(roomId: string): Promise<RoomMergeResult>
  onRoomEvent(cb: (e: RoomStreamEvent) => void): () => void

  getTickets(): Promise<Ticket[]>
  createTicket(t: Partial<Ticket> & { projectId: string }): Promise<Ticket>
  updateTicket(id: string, patch: Partial<Ticket>): Promise<Ticket>
  deleteTicket(id: string): Promise<void>
  generateTickets(projectId: string, idea: string, opts?: TicketWorkOptions): Promise<Ticket[]>
  workTicket(ticketId: string, opts?: TicketWorkOptions): Promise<Thread>
  onTicketUpdate(cb: (t: Ticket) => void): () => void
  getJobs(): Promise<Job[]>
  retryJob(id: string, reason?: string, nextWakeAt?: number | null, resumePlan?: string | null): Promise<Job>
  resumeJob(id: string, resumePlan?: string | null): Promise<Job>
  deadLetterJob(id: string, reason?: string): Promise<Job>
  onJobUpdate(cb: (j: Job) => void): () => void

  getAutomations(): Promise<Automation[]>
  createAutomation(a: Partial<Automation>): Promise<Automation>
  updateAutomation(id: string, patch: Partial<Automation>): Promise<Automation>
  deleteAutomation(id: string): Promise<void>
  runAutomation(id: string): Promise<{ ok: boolean }>
  onAutomationUpdate(cb: (a: Automation) => void): () => void
  onMission(cb: (m: Mission) => void): () => void
  onIncident(cb: (i: Incident) => void): () => void

  getSkills(): Promise<Skill[]>
  // VST-010: repo-authored project skills (hash-pinned trust) + suggestion diffs.
  getProjectSkills(projectId: string): Promise<ProjectSkillView[]>
  trustProjectSkill(input: { projectId: string; skillId: string; trusted: boolean }): Promise<ProjectSkillView[]>
  suggestionDiff(id: string): Promise<SkillSuggestionDiff>
  // First-class artifacts (VST-011).
  listArtifacts(threadId?: string): Promise<Artifact[]>
  previewArtifact(id: string): Promise<ArtifactPreview>
  redactArtifact(id: string): Promise<Artifact>
  exportArtifact(id: string): Promise<{ artifact: Artifact; content: string | null }>
  deleteArtifact(id: string): Promise<void>
  // Observer hooks health (VST-012).
  getObserverStatus(): Promise<{ hooks: Array<{ name: string; delivered: number; failed: number; consecutiveFailures: number; disabled: boolean }>; recent: Array<{ kind: string; at: number; threadId?: string | null; runId?: string | null; payload: Record<string, unknown> }> }>
  saveSkill(s: Partial<Skill>): Promise<Skill>
  deleteSkill(id: string): Promise<void>
  getCommands(): Promise<Command[]>
  saveCommand(c: Partial<Command>): Promise<Command>
  deleteCommand(id: string): Promise<void>
  replyToThread(threadKey: string, text: string, inReplyTo?: string): Promise<{ ok: boolean; error?: string; sentId?: string }>
  requestChannelApproval(threadKey: string, request: ChannelApprovalRequest): Promise<ChannelApprovalResult>
  getSuggestions(): Promise<SkillSuggestion[]>
  acceptSuggestion(id: string): Promise<{ ok: boolean; skill: Skill | null; command: Command | null }>
  dismissSuggestion(id: string): Promise<void>
  onSuggestionsUpdate(cb: (list: SkillSuggestion[]) => void): () => void
  getProjectMemory(projectId: string): Promise<ProjectMemoryEntry[]>
  getProjectMemoryCandidates(projectId: string): Promise<ProjectMemoryCandidate[]>
  proposeProjectMemory(m: Partial<ProjectMemoryCandidate>): Promise<ProjectMemoryCandidate>
  approveProjectMemoryCandidate(id: string): Promise<ProjectMemoryEntry>
  // VST-009 reviewed-memory loop: what an approval would DO, and the way back out.
  memoryCandidateDiff(id: string): Promise<ProjectMemoryCandidateDiff>
  rollbackMemoryEntry(id: string): Promise<{ action: 'restored_previous' | 'removed'; entry: ProjectMemoryEntry | null }>
  rejectProjectMemoryCandidate(id: string): Promise<ProjectMemoryCandidate>
  saveProjectMemory(m: Partial<ProjectMemoryEntry>): Promise<ProjectMemoryEntry>
  deleteProjectMemory(id: string): Promise<void>
  getWebhooks(): Promise<WebhookSnapshot>
  saveWebhook(w: Partial<WebhookEndpoint>): Promise<WebhookSnapshot>
  approveWebhook(id: string): Promise<WebhookSnapshot>
  deleteWebhook(id: string): Promise<WebhookSnapshot>
  testWebhook(id: string): Promise<WebhookSnapshot>
  // Phase 7 — run export & sharing
  buildExport(threadId: string, opts?: ExportOptions): Promise<ExportResult>
  saveExport(threadId: string, opts?: ExportOptions): Promise<ExportSaveResult>
  importBundle(): Promise<ExportImportResult>
  // Phase 6 — production incident mode
  getIncidents(): Promise<Incident[]>
  createIncident(input: CreateIncidentInput): Promise<Incident>
  updateIncident(id: string, patch: Partial<Incident>): Promise<Incident>
  deleteIncident(id: string): Promise<void>
  addIncidentEvent(id: string, kind: IncidentEventKind, text: string): Promise<Incident>
  proposeMitigation(id: string, text: string, rationale?: string): Promise<Incident>
  decideMitigation(id: string, mitigationId: string, decision: IncidentMitigation['status']): Promise<Incident>
  executeMitigation(id: string, mitigationId: string): Promise<Incident>
  scanIncident(id: string): Promise<Incident>
  generatePostmortem(id: string): Promise<Incident>
  // Plan 1a — project CLI terminals
  listClis(): Promise<CliDefinition[]>
  listCliTools(): Promise<CliToolStatus[]>
  startTerminal(projectId: string, cliId: string, opts?: { command?: string; label?: string }): Promise<CliSession>
  writeTerminal(sessionId: string, data: string): Promise<void>
  resizeTerminal(sessionId: string, cols: number, rows: number): Promise<void>
  killTerminal(sessionId: string): Promise<void>
  renameTerminal(sessionId: string, title: string): Promise<void>
  listTerminals(projectId: string): Promise<CliSession[]>
  restorableTerminals(projectId: string): Promise<CliSession[]>
  restoreTerminal(sessionId: string): Promise<CliSession>
  terminalBacklog(sessionId: string): Promise<string>
  // Sequenced gap-fill (VST-039): reconnect/paired clients resume output from a cursor.
  terminalSince(sessionId: string, since?: number): Promise<{ chunks: { seq: number; data: string }[]; gap: boolean; head: number }>
  onTerminalData(cb: (p: { sessionId: string; chunk: string; orchestratorHostId?: string | null }) => void): () => void
  onTerminalStatus(cb: (p: { sessionId: string; status: CliSession['status']; orchestratorHostId?: string | null }) => void): () => void
  onTerminalSessions(cb: (p: { projectId: string; sessions: CliSession[]; orchestratorHostId?: string | null }) => void): () => void

  // Built-in TensorFold engine: status, installed models, Hugging Face search and downloads.
  // A native run notification was clicked (desktop only).
  onOpenThread?(cb: (threadId: string) => void): () => void
  engineSnapshot(): Promise<EngineSnapshot>
  // Contains a launch credential: request only in the user's explicit copy-connection action.
  engineConnection(): Promise<EngineConnection | null>
  engineRefresh(): Promise<EngineSnapshot>
  engineSearch(query: string): Promise<HubModelSummary[]>
  enginePreflight(repo: string, revision?: string | null): Promise<ModelPreflight>
  enginePull(repo: string, revision?: string | null): Promise<PullProgress>
  engineCancelPull(repo: string): Promise<{ cancelled: boolean }>
  engineLoad(repo: string): Promise<EngineStatus>
  engineUnload(): Promise<EngineStatus>
  engineSetCortheon(enabled: boolean): Promise<EngineStatus>
  engineDelete(repo: string): Promise<EngineInstalledModel[]>
  onEngineEvent(cb: (event: EngineEvent) => void): () => void
}

declare global {
  interface Window {
    tensorfoldStudio?: TensorFoldStudioApi
    vontra?: TensorFoldStudioApi
  }
}

// In the desktop app the preload bridge provides `window.tensorfoldStudio`.
// In a plain browser (Vite dev URL) we fall back to a demo implementation.
import { createMockApi } from './mock'
import { createRemoteBrowserApi } from './remote-transport'
import { pairedToken } from './remote-pairing'

// Background reads/polls fail quietly (they drive UI state, not user actions).
// Every other method is a user-initiated action — surface its failure so nothing
// fails silently. App.tsx listens for `vontra:ipc-error` and shows a notice toast.
// Engine reads and Hub checks report their errors inline on the Models page.
const QUIET_IPC = /^(get|list|fetch|discover|check|gitStatus|gitDiff|engine(?:Snapshot|Refresh|Search|Preflight))/

function reportIpcError(channel: string, e: unknown): void {
  const raw = String((e as { message?: string })?.message ?? e)
  const message = raw.replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '').slice(0, 240)
  try { window.dispatchEvent(new CustomEvent('vontra:ipc-error', { detail: message || `${channel} failed.` })) } catch { /* no window (tests) */ }
}

function surfaceIpcErrors(raw: TensorFoldStudioApi): TensorFoldStudioApi {
  const out: Record<string, unknown> = {}
  for (const key of Object.keys(raw) as string[]) {
    const value = (raw as unknown as Record<string, unknown>)[key]
    if (typeof value !== 'function' || QUIET_IPC.test(key) || key === 'sshStatus') {
      out[key] = value
      continue
    }
    out[key] = (...args: unknown[]) => {
      let result: unknown
      try {
        result = (value as (...a: unknown[]) => unknown).apply(raw, args)
      } catch (e) {
        reportIpcError(key, e)
        throw e
      }
      if (result && typeof (result as Promise<unknown>).then === 'function') {
        return (result as Promise<unknown>).catch((e: unknown) => { reportIpcError(key, e); throw e })
      }
      return result
    }
  }
  return out as unknown as TensorFoldStudioApi
}

// Transport pick (VST-035), in order: the Electron preload bridge; a paired /v1 session
// against the host that served this page (token established pre-boot in main.tsx); the
// browser demo mock. The paired branch reads the token synchronously — main.tsx guarantees
// it exists before this module loads.
function pickBrowserTransport(): TensorFoldStudioApi {
  const token = pairedToken()
  if (token) return createRemoteBrowserApi({ token })
  return createMockApi()
}

export const api: TensorFoldStudioApi = surfaceIpcErrors(window.tensorfoldStudio ?? window.vontra ?? pickBrowserTransport())
