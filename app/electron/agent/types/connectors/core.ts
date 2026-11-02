import type { TraceContext } from '../timeline'
import type { ProjectMemoryConfidence, ProjectMemoryKind, ProjectMemoryPrivacyScope } from '../workspace'

// ---- Assistant connector spine (Zora-inspired local contracts) ----
export type ConnectorId = 'email' | 'slack' | 'discord' | 'mattermost' | 'matrix' | 'googlechat' | 'whatsapp' | 'imessage' | 'docs' | 'calendar' | 'telegram' | 'controller' | 'webhook' | 'browser'

export interface SyncCursor {
  position: number
  lastId: string | null
  lastImapUid?: number
  updatedAt: number
}

export interface ConnectorReplyContext {
  id: string | null
  author: string | null
  text: string | null
}

export interface ConnectorChannelPrompt {
  connectorId: ConnectorId
  sourceId: string | null
  channelPrompt?: string
}

export interface ChannelApprovalRequest {
  title: string
  summary: string
  command?: string | null
}

export type ChannelApprovalResult =
  | { ok: true; decision: ConnectorApprovalDecision }
  | { ok: false; error?: string; fallbackToApp?: boolean }

export interface MessageLengthLimit {

  maxUnits: number | null
}

export type MessageLengthLimits = Record<string, MessageLengthLimit>

export type ConnectorStatus = 'connected' | 'mock' | 'pending' | 'error'

export type ConnectorPermissionStatus = 'granted' | 'degraded' | 'denied' | 'not_configured' | 'unknown'

export type ConnectorDirection = 'inbound' | 'outbound' | 'internal'

export type ConnectorPriority = 'critical' | 'high' | 'normal' | 'low'

export type ConnectorTrustLevel = 'notify' | 'suggest' | 'approve' | 'auto_logged' | 'auto_silent'

export type ConnectorClassification = 'communication' | 'document' | 'schedule' | 'approval' | 'system' | 'unknown'

export type ConnectorApprovalStatus = 'pending' | 'approved' | 'denied'

export type ConnectorApprovalDecision = 'approve' | 'deny'

export type RunBudgetClass = 'interactive' | 'backlog' | 'background' | 'mission' | 'overnight' | 'unbounded_with_heartbeat'

export interface ConnectorToolSchema {
  name: string
  description: string
  approvalRequired?: boolean
  parameters: Record<string, unknown>
}

export interface ConnectorDefinition {
  id: ConnectorId
  name: string
  kind: 'communication' | 'document' | 'calendar' | 'system'
  status: ConnectorStatus
  description: string
  transport: string | null
  tools: ConnectorToolSchema[]
  unread: number
  pendingApprovals: number
  lastSyncAt: number | null
  lastEventAt: number | null
  permissionStatus: ConnectorPermissionStatus
  permissionDetail: string
  error?: string
}

export interface ConnectorOutboundSnapshot {
  version: 1
  connectorId: 'email' | 'whatsapp' | 'telegram' | 'imessage'
  accountId: string | null
  // Host binding changes when the selected transport identity or source scope changes.
  accountRevision?: string
  recipient: string
  body: string
  subject?: string
  threadKey: string
  inReplyTo?: string
}

export interface ConnectorEvent {
  id: string
  connectorId: ConnectorId
  pluginName: string
  action: string
  direction: ConnectorDirection
  classification: ConnectorClassification
  priority: ConnectorPriority
  trustLevel: ConnectorTrustLevel
  requiresApproval: boolean
  approvalId: string | null
  approvalStatus?: ConnectorApprovalStatus | null
  resolvedAt?: number | null
  resolvedBy?: string | null
  title: string
  summary: string
  actor: string
  threadKey: string
  payloadKeys: string[]
  payload?: Record<string, unknown>

  reply?: ConnectorReplyContext

  mediaPaths?: string[]
  // Exact immutable outbound content shown for a one-time owner approval.
  outbound?: ConnectorOutboundSnapshot
  reviewDigest?: string
  outboundError?: string
  delivery?: {
    transport: string
    status: 'sending' | 'delivered' | 'failed' | 'uncertain' | 'queued'
    attemptId?: string
    reviewDigest?: string
    attemptedAt?: number
    sentId?: string | null
    path?: string | null
    deliveredAt?: number | null
    error?: string | null
  }
  sources?: ConnectorSourceRef[]
  trace?: TraceContext
  createdAt: number
}

export interface ConnectorSourceRef {
  // Host-authored binding for cached account/location scoped material.
  sourceRevision?: string
  id: string
  connectorId: ConnectorId
  sourceId?: string | null
  itemId?: string | null
  title: string
  path?: string | null
  excerpt?: string | null
  createdAt?: number | null
}

export type ConnectorFeedAction = 'Review' | 'Compare' | 'Plan' | 'Summarize' | 'Attach'

export interface ConnectorAttentionItem {
  id: string
  eventId: string
  connectorId: ConnectorId
  threadKey: string
  connectorAction: string
  title: string
  summary: string
  actor: string
  action: ConnectorFeedAction
  priority: ConnectorPriority
  requiresApproval: boolean
  sourceIds: string[]
  sources: ConnectorSourceRef[]
  createdAt: number

  count?: number
  burst?: boolean
}

export interface ConnectorBriefing {
  generatedAt: number
  summary: string
  lines: string[]
  sourceIds: string[]
  sources: ConnectorSourceRef[]
  eventCount: number
  pendingApprovalCount: number
  latestEventAt: number | null
}

export interface ConnectorMemoryCandidateDraft {
  id: string
  eventId: string
  connectorId: ConnectorId
  projectId: string
  kind: ProjectMemoryKind
  text: string
  sourceType: 'connector'
  sourceRef: string
  sourceIds?: string[]
  confidence: ProjectMemoryConfidence
  privacyScope: ProjectMemoryPrivacyScope
  reason: string
  createdAt: number
}

export interface ConnectorMissionTrigger {
  id: string
  missionId: string
  missionTitle: string
  eventId: string
  connectorId: ConnectorId
  title: string
  summary: string
  action: ConnectorFeedAction
  sourceIds: string[]
  sources: ConnectorSourceRef[]
  createdAt: number
}

export interface ConnectorInteractionInput {
  connectorId: ConnectorId
  pluginName?: string
  action: string
  direction?: ConnectorDirection
  target?: string
  input?: Record<string, unknown>
  output?: Record<string, unknown>
}
