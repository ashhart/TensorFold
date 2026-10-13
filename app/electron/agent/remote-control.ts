import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { timingSafeEqual } from 'node:crypto'
import { currentSecurityRoleFromEnv } from './security'
import { canonicalHomeDirectory, resolvePairedWorkspace, type RemoteWorkspaceScope } from './local-workspace-policy'
import { remoteDeviceStatus } from './device-status'
import type { EnginePerformance, EngineStatus } from './types'
import { normalizeAttachments } from './attachment-policy'
import { loadStaticFile } from './remote-static'
import { parseTerminalStart, parseTerminalWrite, parseTerminalResize, parseTerminalRename } from './terminal-forward'
import { DEFAULT_REMOTE_STREAM_BUFFER_BYTES, KeepTailRemoteEventBuffer } from './remote-stream-buffer'
import type { CliDefinition, CliSession } from './types/cli'
import type {
  CueRunResult,
  CueRunPayload,
  Mission,
  MissionStatus,
  Project,
  RemoteDirectoryListResult,
  RemoteControlCommandDescriptor,
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

// Large enough to accept the max attachment payload (18 MiB aggregate per
// attachment-policy.ts) plus JSON overhead. The server is loopback-only and
// token-gated, so the DoS surface from a larger limit is bounded to an
// authenticated local peer.
const MAX_BODY_BYTES = 20 * 1024 * 1024
const MISSION_STATUSES = new Set<MissionStatus>(['draft', 'queued', 'running', 'waiting_approval', 'paused', 'stalled', 'completed', 'failed', 'cancelled'])
const REMOTE_WEB_CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
const SECURITY_ROLE_RANK: Record<SecurityRole, number> = { viewer: 0, operator: 1, owner: 2 }

interface PairingAttemptState {
  failures: number
  windowStartedAt: number
  lastSeenAt: number
}

export interface PairingAttemptLimiterOptions {
  now?: () => number
  maxFailures?: number
  windowMs?: number
  maxEntries?: number
}

// Per-source-IP brute-force limiter for the only unauthenticated mutation. Entries expire
// with the attempt window and the map has a hard cap so spoofed/source churn cannot grow it
// without bound. The socket address is used directly; forwarded headers are untrusted here.
export class PairingAttemptLimiter {
  private readonly attempts = new Map<string, PairingAttemptState>()
  private readonly now: () => number
  private readonly maxFailures: number
  private readonly windowMs: number
  private readonly maxEntries: number

  constructor(options: PairingAttemptLimiterOptions = {}) {
    this.now = options.now ?? (() => Date.now())
    this.maxFailures = Math.max(1, Math.floor(options.maxFailures ?? 5))
    this.windowMs = Math.max(1_000, Math.floor(options.windowMs ?? 60_000))
    this.maxEntries = Math.max(1, Math.floor(options.maxEntries ?? 1_024))
  }

  check(key: string): { blocked: boolean; retryAfterMs: number } {
    const at = this.now()
    this.prune(at)
    const state = this.attempts.get(key)
    if (!state || state.failures < this.maxFailures) return { blocked: false, retryAfterMs: 0 }
    return { blocked: true, retryAfterMs: Math.max(1, state.windowStartedAt + this.windowMs - at) }
  }

  recordFailure(key: string): void {
    const at = this.now()
    this.prune(at)
    const current = this.attempts.get(key)
    if (current) {
      current.failures++
      current.lastSeenAt = at
      return
    }
    while (this.attempts.size >= this.maxEntries) {
      const oldest = this.attempts.keys().next().value as string | undefined
      if (oldest == null) break
      this.attempts.delete(oldest)
    }
    this.attempts.set(key, { failures: 1, windowStartedAt: at, lastSeenAt: at })
  }

  clear(key: string): void {
    this.attempts.delete(key)
  }

  trackedSourceCount(): number {
    this.prune(this.now())
    return this.attempts.size
  }

  private prune(at: number): void {
    for (const [key, state] of this.attempts) {
      if (at - state.windowStartedAt >= this.windowMs) this.attempts.delete(key)
    }
  }
}

export interface RemoteControlServerDeps {
  getSettings: () => RemoteControlSettings
  getSnapshot: () => RemoteControlSnapshot | Promise<RemoteControlSnapshot>
  getArchives: () => ArchiveSnapshot | Promise<ArchiveSnapshot>
  /** Least-privilege projection served to revocable paired-device tokens. */
  getPairedSnapshot?: () => RemoteControlSnapshot | Promise<RemoteControlSnapshot>
  getPairedArchives: () => ArchiveSnapshot | Promise<ArchiveSnapshot>
  redactPairedEvent?: (event: Pick<RemoteControlEvent, 'kind' | 'data'>) => Pick<RemoteControlEvent, 'kind' | 'data'>
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
  // Terminals (orchestrator-forwarded): drive PTYs hosted on THIS machine for a remote client.
  // Output/status stream back over the /v1/events SSE as kind:'terminal' events (see main-app wiring).
  // Optional — a deps provider that doesn't host terminals simply omits them (routes return 501).
  listClis?: () => CliDefinition[] | Promise<CliDefinition[]>
  startTerminal?: (input: { projectId: string; cliId: string; command?: string; label?: string }) => CliSession
  writeTerminal?: (sessionId: string, data: string) => void
  resizeTerminal?: (sessionId: string, cols: number, rows: number) => void
  renameTerminal?: (sessionId: string, title: string) => void
  killTerminal?: (sessionId: string) => void
  listTerminals?: (projectId: string) => CliSession[]
  // Ordered event spine replay (VST-034). Optional — a deps provider without a spine
  // serves 501 for the spine routes, mirroring the terminals pattern above.
  spineReadSince?: (since: number, options?: { aggregate?: SpineAggregate; streamId?: string; limit?: number }) => ReadSinceResult
  spineReceipt?: (commandId: string) => CommandReceipt | null
  // Per-device pairing + sessions (VST-035). Optional — absent means legacy shared-token-
  // only auth (the `/v1/auth/*` routes serve 501 and the gate accepts only settings.authToken).
  pairing?: {
    issuePairingLink: (label: string) => { credential: string; expiresAt: number } | Promise<{ credential: string; expiresAt: number }>
    authorizeSessionToken: (token: string) => { ok: boolean; sessionId?: string }
    redeemPairingLink: (credential: string, clientInfo?: string | null) => { ok: boolean; session?: { id: string; token: string; label: string; clientInfo?: string | null }; error?: string } | Promise<{ ok: boolean; session?: { id: string; token: string; label: string; clientInfo?: string | null }; error?: string }>
    listSessions: () => Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>
    revokeSession: (id: string) => boolean | Promise<boolean>
  }
  // Where the built renderer bundle lives (dist/). Absent/null → the web client is not
  // served (data API only), and GETs outside /v1 explain that instead of 404ing opaquely.
  staticRoot?: () => string | null
  // Observer hooks (VST-012): each published remote event is observable — event KIND only.
  observePublish?: (kind: string) => void
  onClientSeen: (at: number) => void
  onStarted: (at: number, error: string | null) => void
  onError: (message: string | null) => void
  audit: (kind: string, data: Record<string, unknown>) => void
  /** Test seam; production emits a credential-free SSE comment every 15 seconds. */
  heartbeatIntervalMs?: number
  /** Test seam for the per-client blocked-write queue. */
  sseClientBufferBytes?: number
}

const REMOTE_CONTROL_COMMANDS: RemoteControlCommandDescriptor[] = [
  {
    id: 'control-plane',
    title: 'Control-plane catalog',
    method: 'GET',
    path: '/v1/control-plane',
    effect: 'read',
    description: 'List the authenticated remote-control commands exposed by this TensorFold host.',
  },
  {
    id: 'device.status', title: 'Device status', method: 'GET', path: '/v1/device-status', effect: 'read',
    description: 'Return redacted engine status and effective device capabilities without credentials or task content.',
  },
  {
    id: 'doctor',
    title: 'Remote doctor',
    method: 'GET',
    path: '/v1/doctor',
    effect: 'read',
    description: 'Return redacted runtime health counts, work queue state, approvals, and active-run counts.',
  },
  {
    id: 'archives',
    title: 'List archives',
    method: 'GET',
    path: '/v1/archives',
    effect: 'read',
    description: 'List soft-archived projects, threads, rooms, and tickets available for restoration.',
  },
  {
    id: 'snapshot',
    title: 'Full snapshot',
    method: 'GET',
    path: '/v1/snapshot',
    effect: 'read',
    description: 'Return the full remote orchestration snapshot for trusted clients.',
  },
  {
    id: 'files.list',
    title: 'List folders',
    method: 'GET',
    path: '/v1/files',
    effect: 'read',
    description: 'List directories on this Orchestrator host for remote project setup.',
  },
  {
    id: 'clis.list',
    title: 'List CLIs',
    method: 'GET',
    path: '/v1/clis',
    effect: 'read',
    description: 'List launchable CLIs installed on this Orchestrator host.',
  },
  {
    id: 'project.create',
    title: 'Create project',
    method: 'POST',
    path: '/v1/projects',
    effect: 'write',
    description: 'Create a project record on this Orchestrator host.',
  },
  {
    id: 'project.update',
    title: 'Update project',
    method: 'POST',
    path: '/v1/projects/:id/update',
    effect: 'write',
    description: 'Update a project record on this Orchestrator host.',
  },
  {
    id: 'project.delete',
    title: 'Archive project',
    method: 'POST',
    path: '/v1/projects/:id/delete',
    effect: 'write',
    description: 'Soft-archive a project and its child records on this Orchestrator host; no project data or thread journals are deleted.',
  },
  {
    id: 'project.restore',
    title: 'Restore project',
    method: 'POST',
    path: '/v1/projects/:id/restore',
    effect: 'write',
    description: 'Restore a soft-archived project and its child records on this Orchestrator host.',
  },
  {
    id: 'thread.create',
    title: 'Create thread',
    method: 'POST',
    path: '/v1/threads',
    effect: 'write',
    description: 'Create a task or chat thread on this Orchestrator host.',
  },
  {
    id: 'events',
    title: 'Event stream',
    method: 'GET',
    path: '/v1/events',
    effect: 'read',
    description: 'Open the authenticated SSE event stream with replay support.',
  },
  {
    id: 'spine.events',
    title: 'Spine replay',
    method: 'GET',
    path: '/v1/spine/events',
    effect: 'read',
    description: 'Cursor-based replay of the durable ordered event spine (lean payloads; survives host restarts).',
  },
  {
    id: 'spine.receipt',
    title: 'Command receipt',
    method: 'GET',
    path: '/v1/spine/receipt',
    effect: 'read',
    description: 'Look up the durable receipt for a command id (exactly-once mutation verification).',
  },
  {
    id: 'auth.pair',
    title: 'Redeem pairing link',
    method: 'POST',
    path: '/v1/auth/pair',
    effect: 'write',
    description: 'Exchange a one-time pairing credential for a durable per-device session token.',
  },
  {
    id: 'auth.sessions',
    title: 'List device sessions',
    method: 'GET',
    path: '/v1/auth/sessions',
    effect: 'read',
    description: 'List paired device sessions (owner token only). Never returns raw tokens.',
  },
  {
    id: 'auth.sessions.revoke',
    title: 'Revoke device session',
    method: 'POST',
    path: '/v1/auth/sessions/revoke',
    effect: 'write',
    description: 'Revoke a paired device session by id (owner token only).',
  },
  {
    id: 'approval.resolve',
    title: 'Resolve approval',
    method: 'POST',
    path: '/v1/approvals/:callId',
    effect: 'write',
    description: 'Resolve a pending tool approval on this Orchestrator.',
  },
  {
    id: 'terminal.list',
    title: 'List terminals',
    method: 'GET',
    path: '/v1/terminals',
    effect: 'read',
    description: 'List terminals hosted by this Orchestrator for a project.',
  },
  {
    id: 'terminal.start',
    title: 'Start terminal',
    method: 'POST',
    path: '/v1/terminals',
    effect: 'write',
    description: 'Start a terminal hosted by this Orchestrator for a project.',
  },
  {
    id: 'terminal.write',
    title: 'Write terminal input',
    method: 'POST',
    path: '/v1/terminals/:id/write',
    effect: 'write',
    description: 'Write input to a terminal hosted by this Orchestrator.',
  },
  {
    id: 'terminal.resize',
    title: 'Resize terminal',
    method: 'POST',
    path: '/v1/terminals/:id/resize',
    effect: 'write',
    description: 'Resize a terminal hosted by this Orchestrator.',
  },
  {
    id: 'terminal.rename',
    title: 'Rename terminal',
    method: 'POST',
    path: '/v1/terminals/:id/rename',
    effect: 'write',
    description: 'Rename a terminal hosted by this Orchestrator.',
  },
  {
    id: 'terminal.kill',
    title: 'Kill terminal',
    method: 'POST',
    path: '/v1/terminals/:id/kill',
    effect: 'write',
    description: 'Kill a terminal hosted by this Orchestrator.',
  },
  {
    id: 'thread.start',
    title: 'Start thread',
    method: 'POST',
    path: '/v1/threads/:id/start',
    effect: 'write',
    description: 'Start an agent run in a thread with a userText body.',
  },
  {
    id: 'thread.guide',
    title: 'Guide thread',
    method: 'POST',
    path: '/v1/threads/:id/guide',
    effect: 'write',
    description: 'Queue guidance into a running thread with a userText body.',
  },
  {
    id: 'thread.cancel',
    title: 'Cancel thread',
    method: 'POST',
    path: '/v1/threads/:id/cancel',
    effect: 'write',
    description: 'Cancel a running thread.',
  },
  {
    id: 'thread.archive',
    title: 'Archive thread',
    method: 'POST',
    path: '/v1/threads/:id/archive',
    effect: 'write',
    description: 'Soft-archive a thread while retaining its transcript journal for restoration.',
  },
  {
    id: 'thread.restore',
    title: 'Restore thread',
    method: 'POST',
    path: '/v1/threads/:id/restore',
    effect: 'write',
    description: 'Restore a soft-archived thread on this Orchestrator host.',
  },
  {
    id: 'room.start',
    title: 'Start room',
    method: 'POST',
    path: '/v1/rooms/:id/start',
    effect: 'write',
    description: 'Start a room run with a problem body.',
  },
  {
    id: 'room.create',
    title: 'Create room',
    method: 'POST',
    path: '/v1/rooms',
    effect: 'write',
    description: 'Create a room on this Orchestrator host.',
  },
  {
    id: 'room.update',
    title: 'Update room',
    method: 'POST',
    path: '/v1/rooms/:id/update',
    effect: 'write',
    description: 'Update a room on this Orchestrator host.',
  },
  {
    id: 'room.delete',
    title: 'Archive room',
    method: 'POST',
    path: '/v1/rooms/:id/delete',
    effect: 'write',
    description: 'Soft-archive a room and its room-scoped tickets on this Orchestrator host.',
  },
  {
    id: 'room.restore',
    title: 'Restore room',
    method: 'POST',
    path: '/v1/rooms/:id/restore',
    effect: 'write',
    description: 'Restore a soft-archived room and its room-scoped tickets on this Orchestrator host.',
  },
  {
    id: 'room.cancel',
    title: 'Cancel room',
    method: 'POST',
    path: '/v1/rooms/:id/cancel',
    effect: 'write',
    description: 'Cancel a running room.',
  },
  {
    id: 'room.steer',
    title: 'Steer room',
    method: 'POST',
    path: '/v1/rooms/:id/steer',
    effect: 'write',
    description: 'Queue a live steering message into a running room.',
  },
  {
    id: 'room.rename',
    title: 'Rename room',
    method: 'POST',
    path: '/v1/rooms/:id/rename',
    effect: 'write',
    description: 'Rename a room on this Orchestrator host.',
  },
  {
    id: 'room.memory',
    title: 'Mutate room memory',
    method: 'POST',
    path: '/v1/rooms/:id/memory',
    effect: 'write',
    description: 'Add or remove a manual room memory entry.',
  },
  {
    id: 'mission.status',
    title: 'Set mission status',
    method: 'POST',
    path: '/v1/missions/:id/status',
    effect: 'write',
    description: 'Set a mission status with an optional message.',
  },
  {
    id: 'mission.heartbeat',
    title: 'Record mission heartbeat',
    method: 'POST',
    path: '/v1/missions/:id/heartbeat',
    effect: 'write',
    description: 'Record a heartbeat for a running mission.',
  },
  {
    id: 'ticket.create',
    title: 'Create ticket',
    method: 'POST',
    path: '/v1/tickets',
    effect: 'write',
    description: 'Create a Kanban ticket on this Orchestrator host.',
  },
  {
    id: 'ticket.update',
    title: 'Update ticket',
    method: 'POST',
    path: '/v1/tickets/:id/update',
    effect: 'write',
    description: 'Update a Kanban ticket on this Orchestrator host.',
  },
  {
    id: 'ticket.delete',
    title: 'Archive ticket',
    method: 'POST',
    path: '/v1/tickets/:id/delete',
    effect: 'write',
    description: 'Soft-archive a Kanban ticket on this Orchestrator host.',
  },
  {
    id: 'ticket.restore',
    title: 'Restore ticket',
    method: 'POST',
    path: '/v1/tickets/:id/restore',
    effect: 'write',
    description: 'Restore a soft-archived Kanban ticket on this Orchestrator host.',
  },
  {
    id: 'ticket.start',
    title: 'Start ticket',
    method: 'POST',
    path: '/v1/tickets/:id/start',
    effect: 'write',
    description: 'Start a Kanban ticket execution run on this Orchestrator host.',
  },
  {
    id: 'ticket.lifecycle',
    title: 'Ticket lifecycle',
    method: 'POST',
    path: '/v1/tickets/:id/lifecycle',
    effect: 'write',
    description: 'Advance or resolve a Kanban ticket lifecycle state.',
  },
]

export interface RemoteControlServer {
  syncFromSettings(): Promise<void>
  stop(): Promise<void>
  status(): RemoteControlStatus
  publish(event: Omit<RemoteControlEvent, 'id' | 'createdAt'>): void
  // Per-device pairing (VST-035). No-ops / empty when no pairing registry was injected.
  issuePairingLink(label: string): Promise<{ credential: string; expiresAt: number }>
  listDeviceSessions(): Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>
  revokeDeviceSession(id: string): Promise<boolean>
}

function authToken(req: IncomingMessage): string {
  const header = req.headers.authorization
  if (typeof header === 'string' && /^bearer\s+/i.test(header)) return header.replace(/^bearer\s+/i, '').trim()
  const token = req.headers['x-vontra-token']
  return Array.isArray(token) ? token[0] : (token ?? '')
}

function tokenMatches(expected: string, actual: string): boolean {
  if (!expected || !actual) return false
  const a = Buffer.from(expected)
  const b = Buffer.from(actual)
  if (a.length !== b.length || a.length === 0) return false
  return timingSafeEqual(a, b)
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
    size += buf.length
    if (size > MAX_BODY_BYTES) throw Object.assign(new Error('Payload too large.'), { code: 413 })
    chunks.push(buf)
  }
  if (!chunks.length) return {}
  try {
    const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    throw Object.assign(new Error('Invalid JSON.'), { code: 400 })
  }
}

function writeJson(res: ServerResponse, code: number, body: unknown): void {
  res.writeHead(code, {
    'content-type': 'application/json',
    'cache-control': 'no-store',
  })
  res.end(JSON.stringify(body))
}

async function writeArchiveMutation<T>(res: ServerResponse, operation: Promise<T | null>): Promise<void> {
  const record = await operation
  if (!record) {
    writeJson(res, 404, { error: 'not_found' })
    return
  }
  writeJson(res, 202, record)
}

function writeRemoteStatic(res: ServerResponse, body: Buffer, contentType: string, cacheControl: string): void {
  res.writeHead(200, {
    'content-type': contentType,
    'cache-control': cacheControl,
    'content-security-policy': REMOTE_WEB_CSP,
    'x-content-type-options': 'nosniff',
    'x-frame-options': 'DENY',
    'referrer-policy': 'no-referrer',
    'cross-origin-resource-policy': 'same-origin',
  })
  res.end(body)
}

function securityRoleAllows(required: SecurityRole): boolean {
  return SECURITY_ROLE_RANK[currentSecurityRoleFromEnv(process.env)] >= SECURITY_ROLE_RANK[required]
}

/**
 * Paired-device confinement for remotely created workspaces: the cwd must be an
 * absolute path inside the user's home subtree (never a filesystem root). The owner
 * token browses and creates unrestricted.
 */
export async function isHomeConfinedPath(value: string): Promise<boolean> {
  try { await canonicalHomeDirectory(value.trim(), undefined, false); return true } catch { return false }
}

function requiredRoleForRemoteMutation(pathname: string): SecurityRole {
  if (pathname === '/v1/auth/sessions/revoke' || pathname === '/v1/terminals' || /^\/v1\/terminals\/[^/]+\/(?:write|resize|rename|kill)$/.test(pathname)) {
    return 'owner'
  }
  return 'operator'
}

function sseWire(event: RemoteControlEvent): string {
  return `id: ${event.id}\nevent: ${event.kind}\ndata: ${JSON.stringify(event)}\n\n`
}

function writeSse(res: ServerResponse, event: RemoteControlEvent): boolean {
  return res.write(sseWire(event))
}

export const REMOTE_CONTROL_SSE_HEARTBEAT_MS = 15_000
const REMOTE_CONTROL_SSE_HEARTBEAT = ': heartbeat\n\n'

function isMissionStatus(input: string): input is MissionStatus {
  return MISSION_STATUSES.has(input as MissionStatus)
}

function rejectsArchiveStateMutation(body: Record<string, unknown>, update = false): string | null {
  if (Object.prototype.hasOwnProperty.call(body, 'archivedAt')) return 'archive_state_requires_archive_or_restore_route'
  if (update && Object.prototype.hasOwnProperty.call(body, 'id')) return 'record_id_is_immutable'
  return null
}

function remoteControlCommandCatalog() {
  return {
    generatedAt: Date.now(),
    commands: REMOTE_CONTROL_COMMANDS,
  }
}

export function createRemoteControlServer(deps: RemoteControlServerDeps): RemoteControlServer {
  let server: Server | null = null
  let host: RemoteControlSettings['bindHost'] | null = null
  let port: number | null = null
  let lastStartedAt: number | null = null
  let lastClientAt: number | null = null
  let lastError: string | null = null
  interface SseClientState {
    scope: 'owner' | 'paired'
    buffer: KeepTailRemoteEventBuffer
    blocked: boolean
    closed: boolean
  }
  const clients = new Map<ServerResponse, SseClientState>()
  const pendingClients = new Map<ServerResponse, SseClientState>()
  const sseResponses = new Set<ServerResponse>()
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  // Monotonic event sequence + a bounded replay buffer so a dropped client can gap-fill via
  // Last-Event-ID instead of losing every event in the disconnect window.
  const REPLAY_LIMIT = 1000
  let seq = 0
  const recent: RemoteControlEvent[] = []
  const pairingAttempts = new PairingAttemptLimiter()

  function displayUrl(): string | null {
    if (!server || port == null || !host) return null
    return `http://${host}:${port}`
  }

  function settingsHost(settings: RemoteControlSettings): RemoteControlSettings['bindHost'] {
    // The built-in server is plain HTTP. Keep it loopback-only; cross-machine access must
    // arrive through SSH/Tailscale serve/a TLS reverse proxy terminating on loopback.
    return '127.0.0.1'
  }

  const snapshotFor = (owner: boolean) => owner || !deps.getPairedSnapshot ? deps.getSnapshot() : deps.getPairedSnapshot()
  const archivesFor = (owner: boolean) => owner ? deps.getArchives() : deps.getPairedArchives()
  const eventFor = (event: RemoteControlEvent, owner: boolean): RemoteControlEvent => {
    if (owner || !deps.redactPairedEvent) return event
    return { ...event, ...deps.redactPairedEvent(event) }
  }

  const newClientState = (scope: 'owner' | 'paired'): SseClientState => ({
    scope,
    buffer: new KeepTailRemoteEventBuffer(Math.max(1_024, deps.sseClientBufferBytes ?? DEFAULT_REMOTE_STREAM_BUFFER_BYTES)),
    blocked: false,
    closed: false,
  })

  function removeClient(client: ServerResponse, state?: SseClientState): void {
    const current = state ?? clients.get(client) ?? pendingClients.get(client)
    if (current) {
      current.closed = true
      current.buffer.clear()
    }
    sseResponses.delete(client)
    clients.delete(client)
    pendingClients.delete(client)
  }

  function closeForRebaseline(client: ServerResponse, state: SseClientState, event: RemoteControlEvent): void {
    if (state.closed) return
    const droppedEvents = state.buffer.length + 1
    const droppedBytes = state.buffer.byteLength
    state.closed = true
    state.buffer.clear()
    sseResponses.delete(client)
    clients.delete(client)
    pendingClients.delete(client)
    const gap: RemoteControlEvent = {
      id: event.id,
      kind: 'stream_gap',
      createdAt: Date.now(),
      data: {
        reason: 'backpressure',
        droppedEvents,
        droppedBytes,
        fromEventId: event.id,
        toEventId: event.id,
        recovery: 'snapshot',
      },
    }
    try { client.end(sseWire(gap)) } catch { try { client.destroy() } catch { /* already gone */ } }
  }

  function markClientBlocked(client: ServerResponse, state: SseClientState): void {
    if (state.blocked || state.closed) return
    state.blocked = true
    client.once('drain', () => {
      if (state.closed) return
      state.blocked = false
      flushClient(client, state)
    })
  }

  function flushClient(client: ServerResponse, state: SseClientState): void {
    if (state.closed || state.blocked) return
    if (client.destroyed || client.writableEnded) {
      removeClient(client, state)
      return
    }
    try {
      const gap = state.buffer.takeGapEvent()
      if (gap && !writeSse(client, gap)) {
        markClientBlocked(client, state)
        return
      }
      while (!state.blocked) {
        const event = state.buffer.shift()
        if (!event) return
        if (!writeSse(client, event)) markClientBlocked(client, state)
      }
    } catch {
      removeClient(client, state)
      try { client.destroy() } catch { /* already gone */ }
    }
  }

  function deliverClient(client: ServerResponse, state: SseClientState, event: RemoteControlEvent): void {
    if (state.closed) return
    if (client.destroyed || client.writableEnded) {
      removeClient(client, state)
      return
    }
    if (state.blocked) {
      const queued = state.buffer.enqueue(event)
      if (queued.overflow) closeForRebaseline(client, state, event)
      return
    }
    try {
      if (!writeSse(client, event)) markClientBlocked(client, state)
    } catch {
      removeClient(client, state)
      try { client.destroy() } catch { /* already gone */ }
    }
  }

  function markClient(): void {
    lastClientAt = Date.now()
    deps.onClientSeen(lastClientAt)
  }

  function stopHeartbeats(): void {
    if (heartbeatTimer) clearInterval(heartbeatTimer)
    heartbeatTimer = null
  }

  function startHeartbeats(): void {
    stopHeartbeats()
    const intervalMs = Math.max(1, deps.heartbeatIntervalMs ?? REMOTE_CONTROL_SSE_HEARTBEAT_MS)
    heartbeatTimer = setInterval(() => {
      for (const client of sseResponses) {
        if (client.destroyed || client.writableEnded) {
          sseResponses.delete(client)
          clients.delete(client)
          pendingClients.delete(client)
          continue
        }
        try {
          // SSE comments are intentionally static: they keep intermediaries and readers alive
          // without consuming an event id or disclosing timestamps, state, or credentials.
          const state = clients.get(client) ?? pendingClients.get(client)
          if (!state?.blocked && !client.write(REMOTE_CONTROL_SSE_HEARTBEAT) && state) markClientBlocked(client, state)
        } catch {
          sseResponses.delete(client)
          clients.delete(client)
          pendingClients.delete(client)
          try { client.destroy() } catch { /* already gone */ }
        }
      }
    }, intervalMs)
    ;(heartbeatTimer as NodeJS.Timeout).unref?.()
  }

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url || '/', `http://${host || '127.0.0.1'}`)
    if (url.pathname === '/health') {
      // /health is reachable before the auth check below, so it must not disclose identifying
      // details (the user-chosen serverName) to unauthenticated callers -- that aids fingerprinting
      // of an exposed host. Return only an opaque liveness response.
      writeJson(res, 200, { ok: true, listening: !!server })
      return
    }

    // Static renderer shell (VST-035): every GET outside /v1 serves the built web client so
    // a paired browser/phone loads TensorFold straight from this host. Pre-auth BY DESIGN — the
    // shell is public app code with no user data (like any hosted web app); every data
    // endpoint below stays behind the token gate, and a fresh browser sits on the pairing
    // screen until it redeems a credential.
    if (req.method === 'GET' && !url.pathname.startsWith('/v1/')) {
      const root = deps.staticRoot?.() ?? null
      const file = root ? await loadStaticFile(root, url.pathname) : null
      if (file) {
        writeRemoteStatic(res, file.content, file.contentType, file.cacheControl)
        return
      }
      writeJson(res, 404, { error: root ? 'not_found' : 'web_client_not_served' })
      return
    }

    // Redeem a one-time pairing credential for a device session token (VST-035). Handled
    // BEFORE the bearer-token gate: a fresh device has no session/owner token yet — the
    // credential IS its authorization, verified constant-time inside redeemPairingLink. This
    // is the only pre-gate mutation, and it mints nothing without a valid, unconsumed link.
    if (req.method === 'POST' && url.pathname === '/v1/auth/pair') {
      if (!deps.pairing) { writeJson(res, 501, { error: 'pairing_unavailable' }); return }
      if (!securityRoleAllows('operator')) {
        deps.audit('remote_control.reject', { path: url.pathname, reason: 'insufficient_role', requiredRole: 'operator' })
        writeJson(res, 403, { error: 'insufficient_role', requiredRole: 'operator' })
        return
      }
      const source = req.socket.remoteAddress || 'unknown'
      const rate = pairingAttempts.check(source)
      if (rate.blocked) {
        res.setHeader('retry-after', String(Math.max(1, Math.ceil(rate.retryAfterMs / 1_000))))
        deps.audit('remote_control.reject', { path: url.pathname, reason: 'pairing_rate_limited' })
        writeJson(res, 429, { error: 'pairing_rate_limited' })
        return
      }
      const body = await readJson(req)
      const credential = String(body.credential ?? authToken(req) ?? '')
      const result = await deps.pairing.redeemPairingLink(credential, typeof body.clientInfo === 'string' ? body.clientInfo : null)
      if (!result.ok || !result.session) {
        pairingAttempts.recordFailure(source)
        writeJson(res, 400, { error: result.error ?? 'pair_failed' })
        return
      }
      pairingAttempts.clear(source)
      deps.audit('remote_control.pair', { label: result.session.label, clientInfo: result.session.clientInfo ?? null })
      markClient()
      writeJson(res, 200, { sessionId: result.session.id, token: result.session.token, label: result.session.label })
      return
    }

    const settings = deps.getSettings()
    // Authentication is header-only, including for the fetch-backed browser SSE client.
    // Credentials in query strings leak through browser history, proxy access logs, crash
    // reports, copied links, and referrers, so no remote-control route accepts them.
    const presented = authToken(req)
    // Accept EITHER the shared owner token (the legacy single secret, still the owner's key)
    // OR a live per-device session token (VST-035). A device session authorizes only the
    // data API — issuing/redeeming pairing links itself stays owner-token-only (below), so a
    // paired phone can never mint new devices.
    const ownerAuthed = tokenMatches(settings.authToken, presented)
    const sessionAuth = !ownerAuthed && deps.pairing ? deps.pairing.authorizeSessionToken(presented) : { ok: false }
    if (!ownerAuthed && !sessionAuth.ok) {
      deps.audit('remote_control.reject', { path: url.pathname, reason: 'unauthorized' })
      writeJson(res, 401, { error: 'unauthorized' })
      return
    }
    markClient()
    const workspaceScope: RemoteWorkspaceScope = { confineToHome: !ownerAuthed }

    // Bearer authentication identifies the caller; the process security role caps what
    // this host is willing to do. Viewer hosts remain fully usable for authenticated reads
    // but cannot be mutated through REST. Terminal control and session revocation retain
    // their owner tier, matching the local IPC boundary.
    if (req.method === 'POST') {
      const requiredRole = requiredRoleForRemoteMutation(url.pathname)
      if (!securityRoleAllows(requiredRole)) {
        deps.audit('remote_control.reject', { path: url.pathname, reason: 'insufficient_role', requiredRole })
        writeJson(res, 403, { error: 'insufficient_role', requiredRole })
        return
      }
      if (requiredRole === 'owner' && !ownerAuthed) {
        writeJson(res, 403, { error: 'owner_token_required' })
        return
      }
    }

    // Listing/revoking device sessions is an OWNER action — a paired device can't enumerate
    // or evict its siblings.
    if (req.method === 'GET' && url.pathname === '/v1/auth/sessions') {
      if (!ownerAuthed || !securityRoleAllows('owner')) { writeJson(res, 403, { error: 'owner_role_and_token_required' }); return }
      writeJson(res, 200, { sessions: deps.pairing?.listSessions() ?? [] })
      return
    }
    if (req.method === 'POST' && url.pathname === '/v1/auth/sessions/revoke') {
      if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
      if (!deps.pairing) { writeJson(res, 501, { error: 'pairing_unavailable' }); return }
      const body = await readJson(req)
      const id = String(body.sessionId ?? '')
      const revoked = await deps.pairing.revokeSession(id)
      deps.audit('remote_control.revoke_session', { sessionId: id, revoked })
      writeJson(res, 200, { revoked })
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/snapshot') {
      writeJson(res, 200, await snapshotFor(ownerAuthed))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/archives') {
      writeJson(res, 200, await archivesFor(ownerAuthed))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/files') {
      // Paired-device sessions are the least-privilege tier: confine their filesystem
      // browsing to the home subtree. The owner token browses unrestricted.
      writeJson(res, 200, await deps.listDirectories(url.searchParams.get('dir') || '', { confineToHome: !ownerAuthed }))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/control-plane') {
      writeJson(res, 200, remoteControlCommandCatalog())
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/doctor') {
      writeJson(res, 200, await deps.getDoctor())
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/device-status') {
      const engine = await deps.getDeviceEngine?.()
      writeJson(res, 200, remoteDeviceStatus(ownerAuthed ? 'owner' : 'paired', currentSecurityRoleFromEnv(process.env), engine?.status ?? null, engine?.performance ?? null))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/clis') {
      if (!deps.listClis) { writeJson(res, 501, { error: 'clis_unavailable' }); return }
      writeJson(res, 200, await deps.listClis())
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/terminals') {
      if (!deps.listTerminals) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
      writeJson(res, 200, deps.listTerminals(url.searchParams.get('projectId') || ''))
      return
    }

    // Ordered event spine replay (VST-034): durable, cursor-based gap-fill for remote
    // clients. The SSE ring above stays the live full-fidelity push; this is the survives-
    // restarts ordering log — a reconnecting client reads what changed since its cursor
    // (lean payloads: ids/states, never transcript text) and refetches precisely.
    if (req.method === 'GET' && url.pathname === '/v1/spine/events') {
      if (!deps.spineReadSince) { writeJson(res, 501, { error: 'spine_unavailable' }); return }
      const sinceRaw = Number(url.searchParams.get('since') ?? 0)
      const limitRaw = Number(url.searchParams.get('limit') ?? 500)
      const since = Number.isFinite(sinceRaw) && sinceRaw >= 0 ? Math.floor(sinceRaw) : 0
      const limit = Math.min(Math.max(1, Number.isFinite(limitRaw) ? Math.floor(limitRaw) : 500), 2000)
      const aggregate = url.searchParams.get('aggregate') || undefined
      const streamId = url.searchParams.get('streamId') || undefined
      writeJson(res, 200, deps.spineReadSince(since, { aggregate: aggregate as SpineAggregate | undefined, streamId, limit }))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/spine/receipt') {
      if (!deps.spineReceipt) { writeJson(res, 501, { error: 'spine_unavailable' }); return }
      const commandId = String(url.searchParams.get('commandId') ?? '').trim()
      if (!commandId) { writeJson(res, 400, { error: 'commandId_required' }); return }
      writeJson(res, 200, { receipt: deps.spineReceipt(commandId) })
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/events') {
      const state = newClientState(ownerAuthed ? 'owner' : 'paired')
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-store',
        connection: 'keep-alive',
      })
      // Track the response before the snapshot is assembled. Heartbeat comments can flow
      // during a slow baseline without exposing the client to live events out of order.
      sseResponses.add(res)
      pendingClients.set(res, state)
      res.flushHeaders()
      const remove = () => removeClient(res, state)
      req.on('close', remove)
      res.on('error', remove)
      // Reconnect with Last-Event-ID (EventSource sends it automatically) / ?since: gap-fill from the
      // buffer instead of resending a full snapshot — as long as the gap is still buffered.
      const lastSeen = Number(req.headers['last-event-id'] ?? url.searchParams.get('since') ?? NaN)
      const oldestBuffered = recent.length ? Number(recent[0].id) : seq + 1
      const canReplay = Number.isFinite(lastSeen) && lastSeen <= seq && oldestBuffered <= lastSeen + 1
      if (canReplay) {
        for (const ev of recent) if (Number(ev.id) > lastSeen) deliverClient(res, state, eventFor(ev, ownerAuthed))
      } else {
        // Fresh client, or the gap is larger than the buffer: send a snapshot baselined at the current
        // seq so the client's subsequent Last-Event-ID lines up with the live stream.
        const snapshot = await snapshotFor(ownerAuthed)
        deliverClient(res, state, { id: String(seq), kind: 'snapshot', createdAt: Date.now(), data: snapshot })
      }
      pendingClients.delete(res)
      if (!state.closed && !res.destroyed && !res.writableEnded) clients.set(res, state)
      return
    }

    if (req.method !== 'POST') {
      writeJson(res, 405, { error: 'method_not_allowed' })
      return
    }

    const body = await readJson(req)
    if (url.pathname === '/v1/projects') {
      const protectedFieldError = rejectsArchiveStateMutation(body)
      if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
      const name = String(body.name || '').trim()
      if (!name) { writeJson(res, 400, { error: 'name_required' }); return }
      // Paired sessions cannot mint SSH workspaces or non-home local roots: either
      // would widen the home-confined browsing tier their token is documented to have.
      if (!ownerAuthed) {
        const requestedCwd = typeof body.cwd === 'string' ? body.cwd : ''
        try {
          const canonical = await resolvePairedWorkspace(requestedCwd, body.workspace as Project['workspace'])
          body.cwd = canonical
          if (body.workspace) body.workspace = { kind: 'local', path: canonical }
        } catch {
          writeJson(res, 403, { error: 'paired_sessions_are_limited_to_home_workspaces' })
          return
        }
      }
      writeJson(res, 201, await deps.createProject(body as Partial<Project> & { name?: string; cwd?: string }, workspaceScope))
      return
    }
    const projectUpdate = /^\/v1\/projects\/([^/]+)\/update$/.exec(url.pathname)
    if (projectUpdate) {
      const protectedFieldError = rejectsArchiveStateMutation(body, true)
      if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
      writeJson(res, 202, await deps.updateProject(decodeURIComponent(projectUpdate[1]), body as Partial<Project>))
      return
    }
    const projectDelete = /^\/v1\/projects\/([^/]+)\/delete$/.exec(url.pathname)
    if (projectDelete) {
      await writeArchiveMutation(res, deps.deleteProject(decodeURIComponent(projectDelete[1])))
      return
    }
    const projectRestore = /^\/v1\/projects\/([^/]+)\/restore$/.exec(url.pathname)
    if (projectRestore) {
      await writeArchiveMutation(res, deps.restoreProject(decodeURIComponent(projectRestore[1])))
      return
    }
    if (url.pathname === '/v1/threads') {
      const protectedFieldError = rejectsArchiveStateMutation(body)
      if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
      const kind = body.kind === 'chat' ? 'chat' : 'task'
      const requestedCwd = typeof body.cwd === 'string' ? body.cwd : ''
      // Paired-device sessions are home-confined when BROWSING files; newly created
      // workspaces must not widen that tier to arbitrary directories (e.g. "/").
      if (!ownerAuthed && kind !== 'chat' && requestedCwd && !await isHomeConfinedPath(requestedCwd)) {
        writeJson(res, 403, { error: 'paired_sessions_are_limited_to_home_workspaces' })
        return
      }
      writeJson(res, 201, await deps.createThread({
        cwd: requestedCwd,
        kind,
        projectId: typeof body.projectId === 'string' ? body.projectId : null,
      }, workspaceScope))
      return
    }
    if (url.pathname === '/v1/rooms') {
      const protectedFieldError = rejectsArchiveStateMutation(body)
      if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
      const projectId = String(body.projectId || '').trim()
      if (!projectId) { writeJson(res, 400, { error: 'projectId_required' }); return }
      writeJson(res, 201, await deps.createRoom({ projectId, name: typeof body.name === 'string' ? body.name : undefined }))
      return
    }
    if (url.pathname === '/v1/tickets') {
      const protectedFieldError = rejectsArchiveStateMutation(body)
      if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
      const projectId = String(body.projectId || '').trim()
      if (!projectId) { writeJson(res, 400, { error: 'projectId_required' }); return }
      writeJson(res, 201, await deps.createTicket(body as Partial<Ticket> & { projectId: string }))
      return
    }
    if (url.pathname === '/v1/terminals') {
      // Terminal control is owner-tier in requiredRoleForRemoteMutation; the process
      // role is 'owner' on a normal desktop, so the BEARER's scope must be checked here
      // or a paired-device token could drive an interactive local PTY.
      if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
      if (!deps.startTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
      const parsed = parseTerminalStart(body)
      if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
      if (parsed.command || parsed.cliId.startsWith('custom:')) { writeJson(res, 403, { error: 'custom_terminal_commands_unavailable' }); return }
      writeJson(res, 201, deps.startTerminal(parsed))
      return
    }
    const terminalWrite = /^\/v1\/terminals\/([^/]+)\/write$/.exec(url.pathname)
    if (terminalWrite) {
      if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
      if (!deps.writeTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
      const parsed = parseTerminalWrite(body)
      if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
      deps.writeTerminal(decodeURIComponent(terminalWrite[1]), parsed.data)
      writeJson(res, 202, { ok: true })
      return
    }
    const terminalResize = /^\/v1\/terminals\/([^/]+)\/resize$/.exec(url.pathname)
    if (terminalResize) {
      if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
      if (!deps.resizeTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
      const parsed = parseTerminalResize(body)
      if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
      deps.resizeTerminal(decodeURIComponent(terminalResize[1]), parsed.cols, parsed.rows)
      writeJson(res, 202, { ok: true })
      return
    }
    const terminalRename = /^\/v1\/terminals\/([^/]+)\/rename$/.exec(url.pathname)
    if (terminalRename) {
      if (!deps.renameTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
      const parsed = parseTerminalRename(body)
      if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
      deps.renameTerminal(decodeURIComponent(terminalRename[1]), parsed.title)
      writeJson(res, 202, { ok: true })
      return
    }
    const terminalKill = /^\/v1\/terminals\/([^/]+)\/kill$/.exec(url.pathname)
    if (terminalKill) {
      if (!deps.killTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
      deps.killTerminal(decodeURIComponent(terminalKill[1]))
      writeJson(res, 202, { ok: true })
      return
    }
    const approvalResolve = /^\/v1\/approvals\/([^/]+)$/.exec(url.pathname)
    if (approvalResolve) {
      if (!deps.approve) { writeJson(res, 501, { error: 'approvals_unavailable' }); return }
      const decision = String(body.decision || '').trim()
      if (decision !== 'allow' && decision !== 'allow_always' && decision !== 'deny') { writeJson(res, 400, { error: 'decision_invalid' }); return }
      await deps.approve(decodeURIComponent(approvalResolve[1]), decision)
      writeJson(res, 202, { ok: true })
      return
    }
    const threadStart = /^\/v1\/threads\/([^/]+)\/start$/.exec(url.pathname)
    if (threadStart) {
      const userText = String(body.userText || '').trim()
      if (!userText) { writeJson(res, 400, { error: 'userText_required' }); return }
      const attachments = normalizeAttachments(body.attachments)
      writeJson(res, 202, await deps.startThread({
        threadId: decodeURIComponent(threadStart[1]),
        userText,
        attachments,
        sourceThreadKey: typeof body.sourceThreadKey === 'string' ? body.sourceThreadKey : undefined,
      }, workspaceScope))
      return
    }
    const threadGuide = /^\/v1\/threads\/([^/]+)\/guide$/.exec(url.pathname)
    if (threadGuide) {
      const userText = String(body.userText || '').trim()
      if (!userText) { writeJson(res, 400, { error: 'userText_required' }); return }
      const attachments = normalizeAttachments(body.attachments)
      writeJson(res, 202, await deps.guideThread({
        threadId: decodeURIComponent(threadGuide[1]),
        userText,
        attachments,
      }, workspaceScope))
      return
    }
    const threadCancel = /^\/v1\/threads\/([^/]+)\/cancel$/.exec(url.pathname)
    if (threadCancel) {
      await deps.cancelThread(decodeURIComponent(threadCancel[1]))
      writeJson(res, 202, { ok: true })
      return
    }
    const threadArchive = /^\/v1\/threads\/([^/]+)\/archive$/.exec(url.pathname)
    if (threadArchive) {
      await writeArchiveMutation(res, deps.deleteThread(decodeURIComponent(threadArchive[1])))
      return
    }
    const threadRestore = /^\/v1\/threads\/([^/]+)\/restore$/.exec(url.pathname)
    if (threadRestore) {
      await writeArchiveMutation(res, deps.restoreThread(decodeURIComponent(threadRestore[1])))
      return
    }
    const roomStart = /^\/v1\/rooms\/([^/]+)\/start$/.exec(url.pathname)
    if (roomStart) {
      const problem = String(body.problem || '').trim()
      if (!problem) { writeJson(res, 400, { error: 'problem_required' }); return }
      const attachments = normalizeAttachments(body.attachments)
      writeJson(res, 202, await deps.startRoom(decodeURIComponent(roomStart[1]), problem, attachments, workspaceScope))
      return
    }
    const roomSteer = /^\/v1\/rooms\/([^/]+)\/steer$/.exec(url.pathname)
    if (roomSteer) {
      const text = typeof body.text === 'string' ? body.text : ''
      if (!text.trim()) { writeJson(res, 400, { error: 'text_required' }); return }
      writeJson(res, 202, await deps.steerRoom(decodeURIComponent(roomSteer[1]), text, workspaceScope))
      return
    }
    const roomRename = /^\/v1\/rooms\/([^/]+)\/rename$/.exec(url.pathname)
    if (roomRename) {
      const name = typeof body.name === 'string' ? body.name.trim() : ''
      if (!name) { writeJson(res, 400, { error: 'name_required' }); return }
      writeJson(res, 202, await deps.renameRoom(decodeURIComponent(roomRename[1]), name))
      return
    }
    const roomMemory = /^\/v1\/rooms\/([^/]+)\/memory$/.exec(url.pathname)
    if (roomMemory) {
      writeJson(res, 202, await deps.mutateRoomMemory(decodeURIComponent(roomMemory[1]), body as RoomManualMemoryCommand))
      return
    }
    const roomUpdate = /^\/v1\/rooms\/([^/]+)\/update$/.exec(url.pathname)
    if (roomUpdate) {
      const protectedFieldError = rejectsArchiveStateMutation(body, true)
      if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
      writeJson(res, 202, await deps.updateRoom(decodeURIComponent(roomUpdate[1]), body as Partial<Room>))
      return
    }
    const roomDelete = /^\/v1\/rooms\/([^/]+)\/delete$/.exec(url.pathname)
    if (roomDelete) {
      await writeArchiveMutation(res, deps.deleteRoom(decodeURIComponent(roomDelete[1])))
      return
    }
    const roomRestore = /^\/v1\/rooms\/([^/]+)\/restore$/.exec(url.pathname)
    if (roomRestore) {
      await writeArchiveMutation(res, deps.restoreRoom(decodeURIComponent(roomRestore[1])))
      return
    }
    const roomCancel = /^\/v1\/rooms\/([^/]+)\/cancel$/.exec(url.pathname)
    if (roomCancel) {
      await deps.cancelRoom(decodeURIComponent(roomCancel[1]))
      writeJson(res, 202, { ok: true })
      return
    }
    const missionStatus = /^\/v1\/missions\/([^/]+)\/status$/.exec(url.pathname)
    if (missionStatus) {
      const status = String(body.status || '')
      if (!isMissionStatus(status)) { writeJson(res, 400, { error: 'valid_status_required' }); return }
      writeJson(res, 202, await deps.setMissionStatus(decodeURIComponent(missionStatus[1]), status, typeof body.message === 'string' ? body.message : undefined))
      return
    }
    const missionHeartbeat = /^\/v1\/missions\/([^/]+)\/heartbeat$/.exec(url.pathname)
    if (missionHeartbeat) {
      writeJson(res, 202, await deps.recordMissionHeartbeat(decodeURIComponent(missionHeartbeat[1]), typeof body.message === 'string' ? body.message : 'Remote heartbeat.'))
      return
    }
    const ticketStart = /^\/v1\/tickets\/([^/]+)\/start$/.exec(url.pathname)
    if (ticketStart) {
      const userText = typeof body.userText === 'string' ? body.userText.trim() : ''
      if (!userText) { writeJson(res, 400, { error: 'userText_required' }); return }
      writeJson(res, 202, await deps.startTicketExecution(decodeURIComponent(ticketStart[1]), {
        hostId: typeof body.hostId === 'string' ? body.hostId : '',
        userText,
      }, workspaceScope))
      return
    }
    const ticketUpdate = /^\/v1\/tickets\/([^/]+)\/update$/.exec(url.pathname)
    if (ticketUpdate) {
      const protectedFieldError = rejectsArchiveStateMutation(body, true)
      if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
      writeJson(res, 202, await deps.updateTicket(decodeURIComponent(ticketUpdate[1]), body as Partial<Ticket>))
      return
    }
    const ticketLifecycle = /^\/v1\/tickets\/([^/]+)\/lifecycle$/.exec(url.pathname)
    if (ticketLifecycle) {
      writeJson(res, 202, await deps.transitionTicket(decodeURIComponent(ticketLifecycle[1]), body as TicketRemoteLifecycleCommand))
      return
    }
    const ticketDelete = /^\/v1\/tickets\/([^/]+)\/delete$/.exec(url.pathname)
    if (ticketDelete) {
      await writeArchiveMutation(res, deps.deleteTicket(decodeURIComponent(ticketDelete[1])))
      return
    }
    const ticketRestore = /^\/v1\/tickets\/([^/]+)\/restore$/.exec(url.pathname)
    if (ticketRestore) {
      await writeArchiveMutation(res, deps.restoreTicket(decodeURIComponent(ticketRestore[1])))
      return
    }

    writeJson(res, 404, { error: 'not_found' })
  }

  async function start(nextHost: RemoteControlSettings['bindHost'], nextPort: number): Promise<void> {
    await stop()
    await new Promise<void>((resolve) => {
      const s = createServer((req, res) => {
        handle(req, res).catch((err: any) => {
          // If a handler already started the response (e.g. an SSE writeHead before getSnapshot()
          // rejected), a second writeHead throws ERR_HTTP_HEADERS_SENT and crashes the process.
          if (res.headersSent || res.writableEnded) { try { res.end() } catch { /* already closing */ } return }
          const code = typeof err?.code === 'number' ? err.code : 500
          writeJson(res, code, { error: code === 500 ? 'remote_control_error' : String(err?.message || err) })
        })
      })
      s.on('error', (err: any) => {
        stopHeartbeats()
        server = null
        host = null
        port = null
        lastError = String(err?.message || err || 'Remote control server failed.')
        deps.onError(lastError)
        resolve()
      })
      s.listen(nextPort, nextHost, () => {
        server = s
        host = nextHost
        const addr = s.address()
        port = typeof addr === 'object' && addr ? addr.port : nextPort
        lastStartedAt = Date.now()
        lastError = null
        deps.onStarted(lastStartedAt, null)
        deps.audit('remote_control.start', { host: nextHost, port })
        startHeartbeats()
        resolve()
      })
    })
  }

  async function syncFromSettings(): Promise<void> {
    const settings = deps.getSettings()
    if (!settings.enabled) { await stop(); return }
    if (!settings.authToken) {
      lastError = 'Remote control token is missing.'
      deps.onError(lastError)
      await stop()
      return
    }
    const nextHost = settingsHost(settings)
    const nextPort = Number.isFinite(settings.port) ? Math.max(1, Math.min(65535, Math.round(settings.port))) : 8765
    if (settings.allowLan || settings.bindHost !== '127.0.0.1') deps.audit('remote_control.bind_normalized', { requestedHost: settings.bindHost, host: nextHost, reason: 'plaintext_server_loopback_only' })
    if (server && host === nextHost && port === nextPort) return
    await start(nextHost, nextPort)
  }

  async function stop(): Promise<void> {
    stopHeartbeats()
    for (const res of sseResponses) {
      try { res.end() } catch { /* ignore */ }
    }
    sseResponses.clear()
    clients.clear()
    pendingClients.clear()
    const s = server
    server = null
    host = null
    port = null
    if (s) await new Promise<void>((resolve) => s.close(() => resolve()))
  }

  function status(): RemoteControlStatus {
    const settings = deps.getSettings()
    return {
      enabled: settings.enabled,
      listening: !!server,
      serverName: settings.serverName,
      bindHost: settingsHost(settings),
      port,
      url: displayUrl(),
      clients: clients.size,
      lastStartedAt: settings.lastStartedAt ?? lastStartedAt,
      lastClientAt: settings.lastClientAt ?? lastClientAt,
      lastError: settings.lastError ?? lastError,
    }
  }

  function publish(event: Omit<RemoteControlEvent, 'id' | 'createdAt'>): void {
    try { deps.observePublish?.(event.kind) } catch { /* observers are fail-open */ }
    // Buffer even with no clients connected, so a client that reconnects can still replay the gap.
    const full: RemoteControlEvent = { id: String(++seq), createdAt: Date.now(), ...event }
    recent.push(full)
    if (recent.length > REPLAY_LIMIT) recent.shift()
    for (const [client, state] of clients) deliverClient(client, state, eventFor(full, state.scope === 'owner'))
  }

  async function issuePairingLink(label: string): Promise<{ credential: string; expiresAt: number }> {
    return await (deps.pairing?.issuePairingLink(label) ?? { credential: '', expiresAt: 0 })
  }
  function listDeviceSessions() {
    return deps.pairing?.listSessions() ?? []
  }
  async function revokeDeviceSession(id: string): Promise<boolean> {
    return await (deps.pairing?.revokeSession(id) ?? false)
  }

  return { syncFromSettings, stop, status, publish, issuePairingLink, listDeviceSessions, revokeDeviceSession }
}
