import path from 'node:path'
import type { DiffLine, TimelineItem, ToolState } from './types'
import { auditPreview } from './security'

export type ExternalAgentProtocol = 'acp'
export type ExternalAgentLane = 'thread' | 'room'
export type ExternalAgentCliKind = 'codex' | 'claude' | 'generic'
export type ExternalAgentEventKind = 'message' | 'tool_call' | 'edit' | 'status' | 'error' | 'approval_request' | 'cancelled'
export type ExternalAgentRunStatus = 'queued' | 'running' | 'waiting_approval' | 'paused' | 'completed' | 'failed' | 'cancelled'
export type ExternalAgentControlAction = 'cancel' | 'guide'
export type ExternalAgentAuditAction = 'start' | 'status' | 'event' | 'guide' | 'cancel' | 'complete' | 'error'

export interface ExternalAgentLaunchInput {
  id?: string
  protocol: ExternalAgentProtocol
  lane: ExternalAgentLane
  command: string
  args?: string[]
  cwd: string
  title?: string | null
  threadId?: string | null
  roomId?: string | null
  providerId?: string | null
  providerAccountId?: string | null
  model?: string | null
  env?: Record<string, string>
  envAllowlist?: string[]
}

export interface ExternalAgentProfileInput {
  kind: ExternalAgentCliKind
  env?: Record<string, string>
  profileHome?: string | null
  profilePath?: string | null
  extraAllowlist?: string[]
}

export interface ExternalAgentProfileScope {
  env: Record<string, string>
  envAllowlist: string[]
  profileKeys: string[]
}

export interface ExternalAgentSession {
  id: string
  protocol: ExternalAgentProtocol
  lane: ExternalAgentLane
  title: string
  command: string
  args: string[]
  cwd: string
  threadId: string | null
  roomId: string | null
  providerId: string | null
  providerAccountId: string | null
  model: string | null
  createdAt: number
}

export interface ExternalAgentSpawnSpec {
  command: string
  args: string[]
  cwd: string
  env: Record<string, string>
}

export interface ExternalAgentRuntimeEvent {
  id?: string
  kind: ExternalAgentEventKind
  runStatus?: ExternalAgentRunStatus
  text?: string
  toolName?: string
  title?: string
  args?: Record<string, unknown>
  output?: string
  path?: string
  diff?: DiffLine[]
  state?: ToolState
  mutating?: boolean
  approvalReason?: string
  createdAt?: number
}

export interface ExternalAgentRuntimeSnapshot {
  session: ExternalAgentSession
  status?: ExternalAgentRunStatus
  eventCount?: number
  pendingApprovals?: number
  lastEventAt?: number
  lastText?: string | null
}

export interface ExternalAgentCommandCentreItem {
  id: string
  sessionId: string
  source: 'external_agent'
  protocol: ExternalAgentProtocol
  lane: ExternalAgentLane
  title: string
  detail: string
  footnote?: string
  status: ExternalAgentRunStatus
  startedAt: number
  lastActivityAt: number
  threadId: string | null
  roomId: string | null
  canCancel: boolean
  canGuide: boolean
  canAudit: boolean
}

export interface ExternalAgentControlCommand {
  id: string
  sessionId: string
  action: ExternalAgentControlAction
  text?: string
  reason?: string
  requestedAt: number
}

export interface ExternalAgentAuditRecord {
  kind: `external_agent.${ExternalAgentAuditAction}`
  sessionId: string
  protocol: ExternalAgentProtocol
  lane: ExternalAgentLane
  threadId: string | null
  roomId: string | null
  providerId: string | null
  providerAccountId: string | null
  model: string | null
  title: string
  cwd: string
  command: string
  status?: ExternalAgentRunStatus
  eventKind?: ExternalAgentEventKind
  toolName?: string
  mutating?: boolean
  requiresApproval?: boolean
  controlAction?: ExternalAgentControlAction
  text?: string
  createdAt: number
}

export function createExternalAgentSession(input: ExternalAgentLaunchInput, now = Date.now()): ExternalAgentSession {
  const cwd = path.resolve(input.cwd || process.cwd())
  const title = input.title?.trim() || `${input.protocol.toUpperCase()} external agent`
  return {
    id: input.id || `ext_${now.toString(36)}`,
    protocol: input.protocol,
    lane: input.lane,
    title,
    command: input.command.trim(),
    args: (input.args ?? []).map(String),
    cwd,
    threadId: input.threadId ?? null,
    roomId: input.roomId ?? null,
    providerId: input.providerId ?? null,
    providerAccountId: input.providerAccountId ?? null,
    model: input.model ?? null,
    createdAt: now,
  }
}

export function buildExternalAgentSpawnSpec(input: ExternalAgentLaunchInput): ExternalAgentSpawnSpec {
  const session = createExternalAgentSession(input)
  return {
    command: session.command,
    args: session.args,
    cwd: session.cwd,
    env: pickEnv(input.env ?? {}, input.envAllowlist ?? []),
  }
}

export function externalAgentProfileScope(input: ExternalAgentProfileInput): ExternalAgentProfileScope {
  const env = { ...(input.env ?? {}) }
  const envAllowlist = [...(input.extraAllowlist ?? [])]
  const profileKeys: string[] = []
  const profileHome = input.profileHome?.trim()
  const profilePath = input.profilePath?.trim()

  if (input.kind === 'codex') {
    if (profileHome) env.CODEX_HOME = profileHome
    envAllowlist.push('CODEX_HOME')
    profileKeys.push('CODEX_HOME')
  } else if (input.kind === 'claude') {
    if (profileHome) {
      env.CLAUDE_CONFIG_DIR = profileHome
      env.CLAUDE_HOME = profileHome
    }
    if (profilePath) env.CLAUDE_CONFIG_FILE = profilePath
    envAllowlist.push('CLAUDE_CONFIG_DIR', 'CLAUDE_HOME', 'CLAUDE_CONFIG_FILE')
    profileKeys.push('CLAUDE_CONFIG_DIR', 'CLAUDE_HOME', 'CLAUDE_CONFIG_FILE')
  }

  return {
    env,
    envAllowlist: [...new Set(envAllowlist.map((item) => item.trim()).filter(Boolean))],
    profileKeys,
  }
}

export function externalAgentEventRequiresApproval(event: ExternalAgentRuntimeEvent): boolean {
  if (event.kind === 'approval_request') return true
  if (event.kind === 'edit') return true
  if (event.kind === 'tool_call' && event.mutating) return true
  return false
}

export function externalAgentStatusFromEvent(event: ExternalAgentRuntimeEvent): ExternalAgentRunStatus {
  if (event.runStatus) return event.runStatus
  if (event.kind === 'cancelled') return 'cancelled'
  if (event.kind === 'error') return 'failed'
  if (externalAgentEventRequiresApproval(event)) return 'waiting_approval'
  return 'running'
}

export function externalAgentCommandCentreItem(snapshot: ExternalAgentRuntimeSnapshot): ExternalAgentCommandCentreItem {
  const { session } = snapshot
  const status = snapshot.status ?? 'running'
  const active = isExternalAgentActive(status)
  const provider = [session.providerId, session.model].filter(Boolean).join(' / ') || session.protocol.toUpperCase()
  const account = session.providerAccountId ? `account:${session.providerAccountId}` : 'default account'
  const events = snapshot.eventCount != null ? `${snapshot.eventCount} event${snapshot.eventCount === 1 ? '' : 's'}` : 'live lane'
  const approvals = snapshot.pendingApprovals ? ` · ${snapshot.pendingApprovals} approval${snapshot.pendingApprovals === 1 ? '' : 's'}` : ''
  return {
    id: `external:${session.id}`,
    sessionId: session.id,
    source: 'external_agent',
    protocol: session.protocol,
    lane: session.lane,
    title: session.title,
    detail: `${provider} · ${account} · ${session.lane}`,
    footnote: `${events}${approvals}`,
    status,
    startedAt: session.createdAt,
    lastActivityAt: snapshot.lastEventAt ?? session.createdAt,
    threadId: session.threadId,
    roomId: session.roomId,
    canCancel: active,
    canGuide: active,
    canAudit: true,
  }
}

export function createExternalAgentControlCommand(
  session: ExternalAgentSession,
  input: { action: ExternalAgentControlAction; text?: string; reason?: string },
  now = Date.now(),
): ExternalAgentControlCommand {
  const action = input.action
  const text = input.text?.trim()
  const reason = input.reason?.trim()
  if (action === 'guide' && !text) throw new Error('Guidance text is required before guiding an external agent.')
  return {
    id: `${session.id}:${action}:${now}`,
    sessionId: session.id,
    action,
    text: action === 'guide' ? text : undefined,
    reason: reason || undefined,
    requestedAt: now,
  }
}

export function externalAgentAuditRecord(
  session: ExternalAgentSession,
  input: {
    action: ExternalAgentAuditAction
    event?: ExternalAgentRuntimeEvent
    control?: ExternalAgentControlCommand
    status?: ExternalAgentRunStatus
    text?: string | null
  },
  now = Date.now(),
): ExternalAgentAuditRecord {
  const event = input.event
  const control = input.control
  const status = input.status ?? (event ? externalAgentStatusFromEvent(event) : undefined)
  return {
    kind: `external_agent.${input.action}`,
    sessionId: session.id,
    protocol: session.protocol,
    lane: session.lane,
    threadId: session.threadId,
    roomId: session.roomId,
    providerId: session.providerId,
    providerAccountId: session.providerAccountId,
    model: session.model,
    title: auditPreview(session.title, 160),
    cwd: auditPreview(session.cwd, 240),
    command: auditPreview([session.command, ...session.args].join(' '), 320),
    status,
    eventKind: event?.kind,
    toolName: event?.toolName,
    mutating: event?.mutating,
    requiresApproval: event ? externalAgentEventRequiresApproval(event) : undefined,
    controlAction: control?.action,
    text: auditPreview(input.text ?? control?.text ?? control?.reason ?? event?.text ?? event?.output ?? '', 400),
    createdAt: now,
  }
}

export function externalAgentEventToTimelineItem(session: ExternalAgentSession, event: ExternalAgentRuntimeEvent): TimelineItem {
  const createdAt = event.createdAt ?? Date.now()
  const id = event.id || `${session.id}:${event.kind}:${createdAt}`
  const meta = `${session.protocol} · ${session.lane}${session.providerAccountId ? ` · account:${session.providerAccountId}` : ''}`
  if (event.kind === 'message') {
    return { id, kind: 'assistant', text: event.text ?? '', title: session.title, meta, createdAt }
  }
  if (event.kind === 'error') {
    return { id, kind: 'error', text: event.text || event.output || 'External agent failed.', title: session.title, meta, createdAt }
  }
  if (event.kind === 'status' || event.kind === 'cancelled') {
    return { id, kind: 'notice', text: event.text || (event.kind === 'cancelled' ? 'External agent cancelled.' : 'External agent status updated.'), title: session.title, meta, status: event.kind, createdAt }
  }
  if (event.kind === 'edit') {
    return {
      id,
      kind: 'tool',
      tool: 'external_agent_edit',
      title: event.title || 'External agent proposed edits',
      path: event.path,
      diff: event.diff,
      state: externalAgentEventRequiresApproval(event) ? 'awaiting' : (event.state ?? 'pending'),
      approval: event.approvalReason || 'External agent edit requires TensorFold approval.',
      output: event.output,
      meta,
      createdAt,
    }
  }
  return {
    id,
    kind: 'tool',
    tool: event.toolName || 'external_agent_tool',
    title: event.title || event.toolName || 'External agent tool request',
    args: event.args,
    output: event.output,
    state: externalAgentEventRequiresApproval(event) ? 'awaiting' : (event.state ?? 'pending'),
    approval: externalAgentEventRequiresApproval(event) ? (event.approvalReason || 'External agent tool request requires TensorFold approval.') : undefined,
    meta,
    createdAt,
  }
}

function isExternalAgentActive(status: ExternalAgentRunStatus): boolean {
  return status === 'queued' || status === 'running' || status === 'waiting_approval' || status === 'paused'
}

function pickEnv(source: Record<string, string>, allowlist: string[]): Record<string, string> {
  const out: Record<string, string> = {}
  for (const key of [...new Set(allowlist.map((item) => item.trim()).filter(Boolean))]) {
    if (Object.prototype.hasOwnProperty.call(source, key)) out[key] = source[key]
  }
  return out
}
