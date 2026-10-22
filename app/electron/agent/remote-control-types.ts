import type { RemoteWorkspaceScope } from './local-workspace-policy'
import type { EnginePerformance, EngineStatus } from './types'
import type { CliDefinition, CliSession } from './types/cli'
import type {
  CueRunResult,
  CueRunPayload,
  Mission,
  MissionStatus,
  Project,
  RemoteDirectoryListResult,
  RemoteControlDoctorSnapshot,
  RemoteControlEvent,
  RemoteControlSettings,
  RemoteControlSnapshot,
  RemoteControlStatus,
  StartRunPayload,
  Attachment,
  ArchiveSnapshot,
  Thread,
  Room,
  RoomEventItem,
  RoomManualMemoryCommand,
  Ticket,
  TicketRemoteLifecycleCommand,
  SpineAggregate,
  ReadSinceResult,
  CommandReceipt,
  SecurityRole,
} from './types'


export interface RemoteControlServerDeps {
  getSettings: () => RemoteControlSettings
  getSnapshot: () => RemoteControlSnapshot | Promise<RemoteControlSnapshot>
  getArchives: () => ArchiveSnapshot | Promise<ArchiveSnapshot>
  getPairedSnapshot?: () => RemoteControlSnapshot | Promise<RemoteControlSnapshot>
  getPairedArchives: () => ArchiveSnapshot | Promise<ArchiveSnapshot>
  redactPairedEvent?: (event: Pick<RemoteControlEvent, 'kind' | 'data'>) => Pick<RemoteControlEvent, 'kind' | 'data'>
  isPublishVisible?: (event: Pick<RemoteControlEvent, 'kind' | 'data'>) => boolean
  getDoctor: () => RemoteControlDoctorSnapshot | Promise<RemoteControlDoctorSnapshot>
  getDeviceEngine?: () => Promise<{ status: EngineStatus | null; performance: EnginePerformance | null }>
  createThread: (input: { cwd?: string; kind?: 'task' | 'chat'; projectId?: string | null }, scope?: RemoteWorkspaceScope) => Promise<Thread>
  startThread: (payload: StartRunPayload, scope?: RemoteWorkspaceScope) => Promise<{ runId: string }>
  guideThread: (payload: CueRunPayload, scope?: RemoteWorkspaceScope) => Promise<CueRunResult>
  cancelThread: (threadId: string) => void | Promise<void>
  deleteThread: (threadId: string) => Promise<Thread | null>
  restoreThread: (threadId: string) => Promise<Thread | null>
  approve?: (callId: string, decision: 'allow' | 'allow_always' | 'deny') => void | Promise<void>
  updateProject: (id: string, patch: Partial<Project>) => Promise<Project>
  deleteProject: (id: string) => Promise<Project | null>
  restoreProject: (id: string) => Promise<Project | null>
  createRoom: (input: { projectId: string; name?: string }) => Promise<Room>
  updateRoom: (id: string, patch: Partial<Room>) => Promise<Room>
  renameRoom: (id: string, name: string) => Promise<Room>
  steerRoom: (id: string, text: string, scope?: RemoteWorkspaceScope) => Promise<RoomEventItem>
  mutateRoomMemory: (id: string, command: RoomManualMemoryCommand) => Promise<Room>
  deleteRoom: (id: string) => Promise<Room | null>
  restoreRoom: (id: string) => Promise<Room | null>
  startRoom: (roomId: string, problem: string, attachments?: Attachment[], scope?: RemoteWorkspaceScope) => Promise<{ ok: boolean }>
  cancelRoom: (roomId: string) => void | Promise<void>
  createTicket: (input: Partial<Ticket> & { projectId: string }) => Promise<Ticket>
  startTicketExecution: (id: string, input: { hostId: string; userText: string }, scope?: RemoteWorkspaceScope) => Promise<{ ticket: Ticket; thread: Thread; runId: string }>
  updateTicket: (id: string, patch: Partial<Ticket>) => Promise<Ticket>
  transitionTicket: (id: string, command: TicketRemoteLifecycleCommand) => Promise<Ticket>
  deleteTicket: (id: string) => Promise<Ticket | null>
  restoreTicket: (id: string) => Promise<Ticket | null>
  setMissionStatus: (id: string, status: MissionStatus, message?: string) => Promise<Mission>
  recordMissionHeartbeat: (id: string, message?: string) => Promise<Mission>
  listDirectories: (dir: string, options?: { confineToHome?: boolean }) => Promise<RemoteDirectoryListResult>
  createProject: (input: Partial<Project> & { name?: string; cwd?: string }, scope?: RemoteWorkspaceScope) => Promise<Project>
  listClis?: () => CliDefinition[] | Promise<CliDefinition[]>
  startTerminal?: (input: { projectId: string; cliId: string; command?: string; label?: string }) => CliSession
  writeTerminal?: (sessionId: string, data: string) => void
  resizeTerminal?: (sessionId: string, cols: number, rows: number) => void
  renameTerminal?: (sessionId: string, title: string) => void
  killTerminal?: (sessionId: string) => void
  listTerminals?: (projectId: string) => CliSession[]
  spineReadSince?: (since: number, options?: { aggregate?: SpineAggregate; streamId?: string; limit?: number }) => ReadSinceResult
  spineReceipt?: (commandId: string) => CommandReceipt | null
  pairing?: {
    issuePairingLink: (label: string) => { credential: string; expiresAt: number } | Promise<{ credential: string; expiresAt: number }>
    authorizeSessionToken: (token: string) => { ok: boolean; sessionId?: string }
    redeemPairingLink: (credential: string, clientInfo?: string | null) => { ok: boolean; session?: { id: string; token: string; label: string; clientInfo?: string | null }; error?: string } | Promise<{ ok: boolean; session?: { id: string; token: string; label: string; clientInfo?: string | null }; error?: string }>
    listSessions: () => Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>
    revokeSession: (id: string) => boolean | Promise<boolean>
  }
  staticRoot?: () => string | null
  observePublish?: (kind: string) => void
  onClientSeen: (at: number) => void
  onStarted: (at: number, error: string | null) => void
  onError: (message: string | null) => void
  audit: (kind: string, data: Record<string, unknown>) => void
  heartbeatIntervalMs?: number
  sseClientBufferBytes?: number
}

export interface RemoteControlServer {
  syncFromSettings(): Promise<void>
  stop(): Promise<void>
  status(): RemoteControlStatus
  publish(event: Omit<RemoteControlEvent, 'id' | 'createdAt'>): void
  issuePairingLink(label: string): Promise<{ credential: string; expiresAt: number }>
  listDeviceSessions(): Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>
  revokeDeviceSession(id: string): Promise<boolean>
}
