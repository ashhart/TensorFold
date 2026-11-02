import type { WorkPaneTab } from '../work-pane'
import type { FullWorkspaceLayout } from '../full-workspace-layout'
import type { AgentEvent, ArchiveKind, ArchiveSnapshot, Attachment, Settings, Thread, RendererThreadPatch, TimelineItem, Project, ProjectExecutionTarget, Room, RoomManualMemoryCommand, Seat, RoomStreamEvent, Ticket, TicketStatus, TicketWorkOptions, Automation, Job, AppUpdateStatus, Provider, PersistenceFailure, SecurityPurgeResult, SshHost, SshConnectionState, SshSecretPrompt, SshTrustPrompt, WorkspaceLocation, CliDefinition, CliSession } from '../types'

export type { AgentEvent, ArchiveKind, ArchiveSnapshot, Attachment, Settings, Thread, RendererThreadPatch, TimelineItem, Project, ProjectExecutionTarget, Room, RoomManualMemoryCommand, Seat, RoomStreamEvent, Ticket, TicketStatus, TicketWorkOptions, Automation, Job, AppUpdateStatus, Provider, PersistenceFailure, SecurityPurgeResult, SshHost, SshConnectionState, SshSecretPrompt, SshTrustPrompt, WorkspaceLocation, CliDefinition, CliSession }

export type ProjectTab = 'overview' | 'rooms' | 'board' | 'chats' | 'files' | 'terminals' | 'settings'
export type View = { name: 'home' } | { name: 'bots' } | { name: 'thread'; id: string } | { name: 'commandCentre' } | { name: 'assistant' } | { name: 'settings'; section?: string } | { name: 'plugins' } | { name: 'automations' } | { name: 'projects' } | { name: 'project'; id: string; tab?: ProjectTab } | { name: 'room'; id: string } | { name: 'models'; tab?: 'Models' | 'Workers' | 'Learning' | 'API' }
export type RunStatus = 'idle' | 'thinking' | 'streaming' | 'tool' | 'awaiting' | 'done' | 'error'
export interface Approval { callId: string; itemId: string }
export const BUSY_RUN = new Set<RunStatus>(['thinking', 'streaming', 'tool', 'awaiting'])

// Map a live run status (the `status` map, updated on every agent event) onto the persisted
// Thread.status field. The renderer never receives a thread_update carrying status, so without this
// the agent tree (buildThreadAgentTree reads thread.status) stays stale at 'idle' while a run streams.
export function threadStatusFromRun(run: RunStatus): Thread['status'] {
  if (run === 'error') return 'error'
  if (run === 'idle' || run === 'done') return 'idle'
  return 'running'
}

export interface AppState {
  ready: boolean
  settings: Settings | null
  threads: Thread[]
  view: View
  notice: { kind: 'error' | 'info'; message: string; authored?: true } | null
  status: Record<string, RunStatus>
  approvals: Record<string, Approval[]>
  resolvedTheme: 'light' | 'dark'
  projects: Project[]
  rooms: Room[]
  roomStatus: Record<string, Room['state']>
  roomLive: Record<string, { seat: string; text: string } | undefined>
  tickets: Ticket[]
  jobs: Job[]
  automations: Automation[]
  archives: ArchiveSnapshot | null
  updateStatus: AppUpdateStatus | null
  // Update/changelog modal. `updateModalOpen` is the explicit UI gate;
  // `updateModalSeen` records release notes opened explicitly by the user.
  updateModalOpen: boolean
  /** Explicit install request awaiting native preflight or handoff. */
  updateInstallPending: boolean
  updateModalSeen: string | null
  newProjectOpen: boolean
  newProjectIntent: number
  pendingMention: { path: string; n: number } | null
  // Text injected into the composer from elsewhere (VST-040 ask-why → chat reference). The
  // composer watches `n` and inserts `text` at the caret. Nonce-keyed like pendingMention.
  composerSeed: { text: string; n: number } | null
  homeComposerWorkspace: { workspace: WorkspaceLocation | null; projectId: string | null; cwd: string } | null
  threadDockRequest: { threadId: string; dock: 'terminal' | 'preview'; nonce: number } | null
  sidebarCollapsed: boolean
  fullWorkspaceLayout: FullWorkspaceLayout
  // Thread work pane: folder threads follow rightPanelHidden, a folderless chat opens only as chatPaneOpenId.
  rightPanelHidden: boolean
  chatPaneOpenId: string | null
  workPaneTabs: Record<string, WorkPaneTab>
  workPaneFocus: { threadId: string; tab: WorkPaneTab; path: string | null; nonce: number } | null
  // Sidebar pins, mirrored to localStorage on this device.
  pinnedThreadIds: string[]
  workspaceDockOpen: boolean
  // Assistant source filter ('all' | a ConnectorId). Lifted to the store so the app
  // sidebar drives it and the Assistant has a single sidebar (no duplicate panel).
  assistantSource: string
  // Code/Chat sidebar tag filter ('all' | 'untagged' | a tag name). Store UI state
  // (like assistantSource) so it survives view switches; also mirrored to
  // localStorage so it survives reloads.
  projectTagFilter: string
  // Thread activity display prefs (Tasks #34 / #40). When true, streaming
  // reasoning blocks render as a compact indicator and grouped tool runs start
  // collapsed. Plain UI state (not persisted Settings), like sidebarCollapsed.
  collapseReasoningDefault: boolean
  collapseToolRunsDefault: boolean
  sshHosts: SshHost[]
  sshStatus: Record<string, SshConnectionState>
  sshSecretPrompt: SshSecretPrompt | null
  sshTrustPrompt: SshTrustPrompt | null
  // Live terminal sessions per project (renderer cache; PTYs live in main). Kept in sync by the
  // terminal:sessions broadcast + the actions below. activeTerminalId tracks which one the
  // Terminal view shows for each project (set by clicking a terminal in the sidebar).
  terminals: Record<string, CliSession[]>
  activeTerminalId: Record<string, string | null>

  init(options?: { retry?: boolean }): Promise<void>
  toggleSidebar(): void
  setFullWorkspaceLayout(layout: FullWorkspaceLayout): void
  toggleRightPanel(): void
  openWorkPane(tab: WorkPaneTab, opts?: { path?: string }): void
  setWorkPaneTab(threadId: string, tab: WorkPaneTab): void
  closeWorkPane(): void
  togglePinnedThread(id: string): void
  toggleWorkspaceDock(): void
  setWorkspaceDockOpen(open: boolean): void
  setAssistantSource(source: string): void
  setProjectTagFilter(tag: string): void
  setCollapseReasoningDefault(value: boolean): void
  setCollapseToolRunsDefault(value: boolean): void
  cycleApprovalMode(): void
  setView(v: View): void
  dismissNotice(): void
  setNotice(notice: { kind: 'error' | 'info'; message: string; authored?: true } | null): void
  newThread(projectId?: string | null): Promise<void>
  newChat(projectId?: string | null): Promise<void>
  // Fork a thread from one of its turn checkpoints (VST-036). The host creates and seeds
  // the new thread; the store only inserts it and navigates there. `worktree` asks for a
  // dedicated fork worktree (falls back to the shared cwd with worktreeError explaining why).
  forkThreadFromTurn(threadId: string, turn: number, options?: { worktree?: boolean }): Promise<{ ok: boolean; error?: string; worktreePath?: string; worktreeError?: string }>
  // Hand this conversation off to a DIFFERENT engine (VST-036): the host flattens the
  // transcript and auto-runs it as the new thread's first turn.
  // intent 'run-plan' hands an external model's plan to a trusted local engine to carry out.
  handoffThread(threadId: string, toProviderId: string, toModel?: string | null, intent?: 'run-plan'): Promise<{ ok: boolean; error?: string }>
  launch(opts: { text: string; cwd: string; providerId: string; model: string; approvalMode: Settings['approvalMode']; providerAccountId?: string | null; projectId?: string | null; workspace?: WorkspaceLocation | null; attachments?: Attachment[]; sourceThreadKey?: string; persistActiveModel?: boolean; incognito?: boolean; memoryPolicy?: import('../types').ThreadMemoryPolicy }): Promise<{ ok: boolean; threadId: string; error?: string }>
  refreshSshHosts(): Promise<void>
  connectSshHost(hostId: string): Promise<void>
  disconnectSshHost(hostId: string): Promise<void>
  setThreadWorkspace(threadId: string, workspace: WorkspaceLocation | null, cwd?: string | null): Promise<void>
  resolveSshSecret(value: string | null): Promise<void>
  resolveSshTrust(ok: boolean): Promise<void>
  setThreadProject(threadId: string, projectId: string | null): Promise<void>
  addProject(name: string, cwd: string): Promise<Project>
  openThread(id: string): void
  newTask(): void
  deleteThread(id: string): Promise<void>
  archiveItem(kind: ArchiveKind, id: string): Promise<void>
  restoreItem(kind: ArchiveKind, id: string, originHostId?: string | null): Promise<void>
  refreshArchives(): Promise<void>
  send(text: string, attachments?: Attachment[], threadId?: string, quotes?: import('../../../shared/message-quotes').MessageQuote[]): Promise<boolean>
  cancel(threadId: string): Promise<void>
  approve(threadId: string, callId: string, decision: 'allow' | 'allow_always' | 'deny'): Promise<void>
  saveSettings(s: Settings, opts?: { silent?: boolean }): Promise<void>
  revokeApprovalRule(id: string): Promise<void>
  purgeLocalMemory(): Promise<SecurityPurgeResult>
  setThreadProvider(threadId: string, providerId: string, model: string, providerAccountId?: string | null): Promise<void>
  noteModelChange(threadId: string, providerId: string, model: string, providerAccountId?: string | null): Promise<void>
  patchThread(id: string, patch: RendererThreadPatch): Promise<void>
  renameThread(id: string, title: string): Promise<void>
  setThreadEngine(id: string, engine: 'native' | 'omp'): Promise<void>
  compactThread(id: string): Promise<void>
  applyEvent(e: AgentEvent): void
  applyThreadUpdate(t: Thread): void
  // #42 — confirm or dismiss a chat-requested setting change.
  resolveAppAction(threadId: string, itemId: string, decision: 'apply' | 'dismiss'): Promise<void>

  createProject(name: string): Promise<void>
  newProject(opts: { name: string; cwd?: string; mode: 'solo' | 'room'; localOnly?: boolean; executionTarget?: ProjectExecutionTarget; orchestratorId?: string | null; defaultProviderId?: string | null; defaultProviderAccountId?: string | null; defaultModel?: string | null; tags?: string[] }, modalIntent?: number): Promise<Project>
  syncRemoteProjects(hostId: string, projects: Project[]): Promise<void>
  applyProjectUpdate(p: Project): void
  openNewProject(): void
  closeNewProject(): void
  addFileMention(path: string): void
  seedComposer(text: string): void
  openThreadDock(threadId: string, dock: 'terminal' | 'preview'): void
  listTerminalClis(projectId: string): Promise<CliDefinition[]>
  loadTerminals(projectId: string): Promise<void>
  startTerminal(projectId: string, cliId: string, opts?: { command?: string; label?: string }): Promise<CliSession | null>
  killTerminal(projectId: string, sessionId: string): Promise<void>
  renameTerminal(projectId: string, sessionId: string, title: string): Promise<void>
  setActiveTerminal(projectId: string, sessionId: string | null): void
  openTerminal(projectId: string, sessionId: string): void
  openProject(id: string): void
  updateProject(id: string, patch: Partial<Project>): Promise<void>
  renameProject(id: string, name: string): Promise<void>
  deleteProject(id: string): Promise<void>
  createRoom(projectId: string, name: string): Promise<Room>
  openRoom(id: string): void
  deleteRoom(id: string): Promise<void>
  updateRoom(id: string, patch: Partial<Room>): Promise<void>
  applyRoomUpdate(r: Room): void
  renameRoom(id: string, name: string): Promise<void>
  setRoomSeats(roomId: string, seats: Seat[]): Promise<void>
  steerRoom(roomId: string, text: string): Promise<void>
  mutateRoomMemory(roomId: string, command: RoomManualMemoryCommand): Promise<void>
  startRoom(roomId: string, problem: string, attachments?: Attachment[]): Promise<void>
  startRoomFromTicket(ticketId: string, roomId?: string | null): Promise<void>
  cancelRoom(roomId: string): Promise<void>
  applyRoomEvent(e: RoomStreamEvent): void

  createTicket(t: Partial<Ticket> & { projectId: string }): Promise<void>
  updateTicket(id: string, patch: Partial<Ticket>): Promise<void>
  moveTicket(id: string, status: TicketStatus): void
  deleteTicket(id: string): Promise<void>
  generateTickets(projectId: string, idea: string, opts?: TicketWorkOptions): Promise<void>
  workTicket(id: string, opts?: TicketWorkOptions): Promise<void>
  workTicketRemote(id: string): Promise<void>
  applyTicketUpdate(t: Ticket): void
  applyJobUpdate(j: Job): void
  retryJob(id: string, reason?: string, nextWakeAt?: number | null, resumePlan?: string | null): Promise<void>
  resumeJob(id: string, resumePlan?: string | null): Promise<void>
  deadLetterJob(id: string, reason?: string): Promise<void>

  createAutomation(a: Partial<Automation>): Promise<Automation>
  updateAutomation(id: string, patch: Partial<Automation>): Promise<void>
  deleteAutomation(id: string): Promise<void>
  runAutomation(id: string): Promise<void>
  applyAutomationUpdate(a: Automation): void
  checkForUpdates(): Promise<void>
  openUpdateDownload(): Promise<void>
  downloadAppUpdate(): Promise<void>
  cancelAppUpdateDownload(): Promise<void>
  installAppUpdate(): Promise<void>
  applyUpdateStatus(s: AppUpdateStatus): void
  openUpdateModal(): void
  closeUpdateModal(): void
}

// SetState accepts either a partial state object or an updater function. Slices use this type so
// they can call `set((st) => ({...}))` exactly as the original module-scope action bodies did.
export type SetState = (partial: AppState | Partial<AppState> | ((state: AppState) => AppState | Partial<AppState>)) => void
export type GetState = () => AppState

// A slice is a function of (set, get) that returns a Partial<AppState>. Annotating a slice's
// return type as `Slice` lets TypeScript infer each action's parameter types from the matching
// AppState method signature — exactly the contextual typing the original `create<AppState>(...)`
// body got for free. Slices may omit fields they don't own.
export type Slice = (set: SetState, get: GetState) => Partial<AppState>
