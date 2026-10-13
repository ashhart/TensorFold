import type { ApprovalMode, ApprovalRuleScope, Attachment, Provider, ProviderAccountKind } from './base'
import type { ControlPlaneGap, RunBudgetDefinition } from './capabilities'
import type {
  ConnectorAttentionItem,
  ConnectorBriefing,
  ConnectorDefinition,
  ConnectorEvent,
  ConnectorMemoryCandidateDraft,
  ConnectorMissionTrigger,
  ConnectorToolSchema,
  RunBudgetClass,
} from './connectors'
import type { PlanStep, Thread, TimelineItem, TraceContext } from './timeline'
import type { AgentRoleProfileId, FanOutBudget, RoomState, SeatHeartbeat } from './workspace'

// ---- Durable jobs / run leases (shared supervisor kernel) ----
export type JobKind = 'thread' | 'mission' | 'room' | 'ticket' | 'automation' | 'connector' | 'pr_loop'
export type JobStatus = 'queued' | 'running' | 'waiting_approval' | 'paused' | 'stalled' | 'completed' | 'failed' | 'cancelled' | 'dead_letter'
export type JobExecutionVerification = 'live' | 'unverifiable' | 'exited'
export type JobRestartResumePolicy = 'never' | 'ask' | 'read_only' | 'low_risk' | 'full_policy'
export type JobResumeAfterRestart = JobRestartResumePolicy | 'pending_approval'
export type JobApprovalDecision = 'allow' | 'allow_always' | 'deny'

export interface JobApprovalPolicySnapshot {
  scope: ApprovalRuleScope
  key: string
  label: string
  explanation: string
}

export interface JobPendingApproval {
  id: string
  callId: string
  threadId: string | null
  itemId: string | null
  tool: string
  policy: JobApprovalPolicySnapshot
  requestedAt: number
  resolvedAt: number | null
  decision: JobApprovalDecision | null
}

export interface Job {
  id: string
  kind: JobKind
  status: JobStatus
  attempt: number
  owner: string
  projectId: string | null
  missionId: string | null
  roomId: string | null
  threadId: string | null
  ticketId?: string | null
  automationId?: string | null
  connectorEventId?: string | null
  currentRunId: string | null
  currentThreadId: string | null
  nextWakeAt: number | null
  lastUsefulProgressAt: number | null
  blockedReason: string | null
  resumePlan: string | null
  resumeAfterRestart: JobResumeAfterRestart
  pendingApprovals: JobPendingApproval[]
  leaseOwner: string | null
  leaseExpiresAt: number | null
  /** Remote execution identity and evidence state. Transport loss never implies process failure. */
  executionHostId?: string | null
  executionVerification?: JobExecutionVerification | null
  executionVerificationReason?: string | null
  executionVerifiedAt?: number | null
  startedAt?: number | null
  finishedAt?: number | null
  metadata?: Record<string, unknown>
  createdAt: number
  updatedAt: number
  // Synthesized by paired snapshot reconciliation when the host no longer has this row.
  archivedAt?: number | null
}

export type RoomSupervisorTicketPhase = 'build' | 'review'
export type RoomSupervisorPhaseStatus = 'idle' | 'queued' | 'running' | 'blocked' | 'completed' | 'failed'

export interface RoomSupervisorPhaseSnapshot {
  phase: RoomSupervisorTicketPhase
  status: RoomSupervisorPhaseStatus
  ticketId: string | null
  ticketTitle?: string | null
  seat?: string | null
  threadId?: string | null
  jobId?: string | null
  paths?: string[]
  summary?: string | null
  updatedAt: number | null
}

export interface RoomSupervisorJobMetadata extends Record<string, unknown> {
  schema: 'vontra.room-supervisor/1'
  supervisor: 'room'
  missionId?: string | null
  roomId: string
  roomName: string
  projectId: string
  projectName?: string | null
  problem: string
  roomState: RoomState
  workspaceCwd?: string | null
  scratchpadCwd?: string | null
  rootCwd?: string | null
  seatCount: number
  eventCount: number
  ticketCounts: Record<string, number>
  currentTurnSeat?: string | null
  lastEventKind?: string | null
  lastEventSeat?: string | null
  lastEventAt?: number | null
  lastHeartbeatAt: number
  // Per-seat liveness (BUG #53). Keyed by seat id; a non-null missedSince means stalled.
  seatHeartbeats?: Record<string, SeatHeartbeat>
  workerBudget: { maxActive: number; maxStarts: number; started: number; active: number }
  phases: Record<RoomSupervisorTicketPhase, RoomSupervisorPhaseSnapshot>
}

// ---- Durable Missions (resumable command-centre operations) ----
export type MissionStatus =
  | 'draft'
  | 'queued'
  | 'running'
  | 'waiting_approval'
  | 'paused'
  | 'stalled'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type MissionJournalEventKind =
  | 'created'
  | 'started'
  | 'resumed'
  | 'paused'
  | 'heartbeat'
  | 'checkpoint'
  | 'approval_requested'
  | 'approval_resolved'
  | 'status'
  | 'note'
  | 'error'
  | 'completed'
  | 'cancelled'

export type MissionJournalActor = 'system' | 'agent' | 'user' | 'connector'
export type MissionApprovalDecision = 'allow' | 'deny'
export type MissionRepeatMode = 'until_condition' | 'interval' | 'self_paced' | 'fixed_count' | 'forever'
export type MissionContextMode = 'fresh' | 'build_on_previous' | 'fresh_with_notes'

export interface MissionBudgetSnapshot {
  executionClass: RunBudgetClass
  label: string
  maxWallSeconds: number
  heartbeatIntervalSeconds: number
  stallGraceSeconds: number
  requiresHeartbeat: boolean
  startedAt: number | null
  deadlineAt: number | null
}

export interface MissionHeartbeat {
  intervalSeconds: number
  stallGraceSeconds: number
  required: boolean
  count: number
  lastBeatAt: number | null
  nextDueAt: number | null
  missedSince: number | null
}

export interface MissionCheckpoint {
  id: string
  sequence: number
  label: string
  summary: string
  status: MissionStatus
  createdAt: number
  data?: Record<string, unknown>
  artifactRefs?: string[]
}

export interface MissionJournalEvent {
  id: string
  kind: MissionJournalEventKind
  actor: MissionJournalActor
  message: string
  createdAt: number
  status?: MissionStatus
  checkpointId?: string
  approvalId?: string
  data?: Record<string, unknown>
}

export interface MissionPendingApproval {
  id: string
  callId: string
  itemId?: string
  title: string
  summary: string
  requestedAt: number
  expiresAt: number | null
  data?: Record<string, unknown>
}

export type MissionScratchMemoryKind = 'note' | 'decision' | 'finding' | 'blocker'

export interface MissionScratchMemoryEntry {
  id: string
  kind: MissionScratchMemoryKind
  text: string
  source: MissionJournalActor
  runIndex: number | null
  checkpointId: string | null
  createdAt: number
  updatedAt: number
}

export interface Mission {
  id: string
  title: string
  objective: string
  stopCondition: string
  repeatMode: MissionRepeatMode
  contextMode: MissionContextMode
  roleProfileId: AgentRoleProfileId
  fanOutBudget: FanOutBudget
  maxRuns: number | null
  costCapUsd: number | null
  expiresAt: number | null
  destructiveApprovalRequired: boolean
  status: MissionStatus
  runBudgetClass: RunBudgetClass
  budget: MissionBudgetSnapshot
  heartbeat: MissionHeartbeat
  checkpoints: MissionCheckpoint[]
  journal: MissionJournalEvent[]
  scratchMemory: MissionScratchMemoryEntry[]
  pendingApprovals: MissionPendingApproval[]
  threadId: string | null
  projectId: string | null
  cwd: string | null
  providerId: string | null
  providerAccountId?: string | null
  model: string | null
  createdAt: number
  updatedAt: number
  startedAt: number | null
  finishedAt: number | null
  metadata?: Record<string, unknown>
}

export interface CreateMissionInput {
  title: string
  objective: string
  stopCondition?: string
  repeatMode?: MissionRepeatMode
  contextMode?: MissionContextMode
  roleProfileId?: AgentRoleProfileId
  fanOutBudget?: FanOutBudget
  maxRuns?: number | null
  costCapUsd?: number | null
  expiresAt?: number | null
  destructiveApprovalRequired?: boolean
  runBudgetClass?: RunBudgetClass
  threadId?: string | null
  projectId?: string | null
  cwd?: string | null
  providerId?: string | null
  providerAccountId?: string | null
  model?: string | null
  metadata?: Record<string, unknown>
}

export interface CreatePrLoopInput {
  title: string
  objective: string
  cwd?: string | null
  projectId?: string | null
  baseBranch?: string | null
  requiredApprovals?: number | null
  autoStart?: boolean
  providerId?: string | null
  providerAccountId?: string | null
  model?: string | null
  // Opt-in (default false): after the local reviewer approves, push the branch and open a
  // GitHub draft PR via `gh` (the local loop otherwise stays local-only).
  publishRemote?: boolean
}

export interface CreatePrLoopResult {
  localPrId: string
  branch: string
  baseBranch: string | null
  worktree: string
  mission: Mission
  implementationThread: Thread
  reviewThread: Thread
}

export interface MissionRuntimeSnapshot {
  generatedAt: number
  missions: Mission[]
  activeMissionIds: string[]
  pendingApprovalCount: number
}

export interface MissionInspectionSnapshot {
  generatedAt: number
  mission: Mission
  terminal: boolean
  active: boolean
  rerunRequired: false
  checkpointCount: number
  journalCount: number
  scratchMemoryCount: number
  pendingApprovalCount: number
  latestCheckpointId: string | null
  latestJournalId: string | null
}

export type ModelRouterCooldownKind = 'rate_limit' | 'quota_exhausted'

export interface ModelRouterCooldownSnapshot {
  providerId: string
  providerName: string
  providerKind: Provider['kind'] | 'cloud'
  providerTransport: Provider['transport'] | null
  accountId: string | null
  accountName: string
  accountKind: ProviderAccountKind | 'provider-default'
  model: string
  kind: ModelRouterCooldownKind
  observedAt: number
  resetAt: number
  source: string
}

export interface ModelRouterHistoryEntry {
  id: string
  scope: 'thread' | 'ticket' | 'automation' | 'mission' | 'room' | 'local_pr'
  subjectId: string | null
  subjectLabel: string | null
  providerId: string
  providerName: string
  fromAccountId: string | null
  fromAccountName: string
  toAccountId: string | null
  toAccountName: string
  model: string
  reason: string
  exhaustionKind: ModelRouterCooldownKind | null
  resetAt: number | null
  createdAt: number
}

export interface ModelRouterSnapshot {
  generatedAt: number
  activeCooldowns: ModelRouterCooldownSnapshot[]
  recentlyReset: ModelRouterCooldownSnapshot[]
  history: ModelRouterHistoryEntry[]
}

export interface AssistantControlSnapshot {
  generatedAt: number
  connectors: ConnectorDefinition[]
  events: ConnectorEvent[]
  attention: ConnectorAttentionItem[]
  briefing: ConnectorBriefing
  memoryCandidates: ConnectorMemoryCandidateDraft[]
  missionTriggers: ConnectorMissionTrigger[]
  gaps: ControlPlaneGap[]
  transports: string[]
  tools: ConnectorToolSchema[]
  budgets: RunBudgetDefinition[]
  pendingApprovals: number
}

// ---- App update detection ----
export type AppUpdateState = 'not_configured' | 'not_checked' | 'checking' | 'current' | 'available' | 'downloading' | 'ready' | 'installing' | 'error'
export type AppUpdateIntegrityState =
  | 'not_checked'
  | 'manifest_checksum'
  | 'missing_checksum'
  | 'signature_verified'
  | 'missing_signature'
  | 'signature_failed'
  | 'artifact_verified'
  | 'artifact_verification_failed'

export interface AppUpdateChecksum {
  algorithm: 'sha256'
  value: string
}

export interface AppUpdateStatus {
  state: AppUpdateState
  currentVersion: string
  latestVersion: string | null
  releaseName: string | null
  releaseNotes: string | null
  downloadUrl: string | null
  checksum: AppUpdateChecksum | null
  signature: string | null
  signatureVerified: boolean
  integrity: AppUpdateIntegrityState
  feedUrl: string | null
  checkedAt: number | null
  error: string | null
  automaticUpdate?: { zipUrl: string; checksum: AppUpdateChecksum; arch: 'arm64' | 'x64' | 'universal' } | null
  automaticInstall?: { supported: boolean; reason: string | null }
  downloadProgress?: { receivedBytes: number; totalBytes: number | null } | null
}

export interface StartRunPayload {
  threadId: string
  /** Trusted host scope; never copied from a remote request body. */
  workspaceBoundary?: 'home'
  userText: string
  // #33 — per-run approval override. Loop/room role threads run autonomously
  // ('auto') so the loop never stalls waiting for a human to approve a tool call.
  approvalMode?: ApprovalMode
  // Documents (extracted text) + images (data URLs) attached from the composer.
  attachments?: Attachment[]
  // Connector conversation this run is acting on (a threadKey, e.g. "slack:<src>:<chan>:<ts>").
  // When set, the source's per-channel ephemeral prompt (channelPrompt) is injected for this
  // run only — never persisted into the system-prompt snapshot.
  sourceThreadKey?: string
}

export interface CueRunPayload {
  threadId: string
  userText: string
  attachments?: Attachment[]
}

export interface CueRunResult {
  queued: boolean
  queueDepth: number
}

export interface AgentCue {
  id: string
  itemId: string
  text: string
  createdAt: number
}

// Events streamed main -> renderer
export type AgentEvent = (
  | { runId: string; threadId: string; type: 'status'; status: 'thinking' | 'streaming' | 'tool' | 'awaiting' | 'done' | 'error' }
  | { runId: string; threadId: string; type: 'lifecycle'; phase: 'run_start' | 'turn_start' | 'model_start' | 'model_end' | 'tool_start' | 'tool_end' | 'turn_end' | 'run_end'; ts: number; step?: number; callId?: string; tool?: string; ok?: boolean }
  | { runId: string; threadId: string; type: 'item_add'; item: TimelineItem }
  | { runId: string; threadId: string; type: 'item_delta'; id: string; field: 'text' | 'output'; chunk: string }
  | { runId: string; threadId: string; type: 'item_update'; id: string; patch: Partial<TimelineItem> }
  | { runId: string; threadId: string; type: 'thread_update'; patch: Partial<Pick<Thread, 'providerId' | 'providerAccountId' | 'model' | 'modelRoute' | 'activeRunContract' | 'lastRunReceipt'>> }
  | { runId: string; threadId: string; type: 'approval_request'; callId: string; itemId: string }
  | { runId: string; threadId: string; type: 'title'; title: string }
  | { runId: string; threadId: string; type: 'plan'; plan: PlanStep[]; goalStatus?: 'active' | 'complete' | 'paused' }
  | { runId: string; threadId: string; type: 'done' }
  | { runId: string; threadId: string; type: 'error'; message: string }
) & { trace?: TraceContext; orchestratorHostId?: string | null }

// ---- Live orchestration lifecycle events (additive push channel) ----
// Typed lifecycle events the supervisors push the instant a mission/room/job changes
// state, so the renderer can update live instead of only learning from the next
// snapshot poll (the poll path stays as the reconciling fallback). The broker that
// emits these lives in electron/main/host/orchestration-events.ts; the shapes are kept
// here in the shared type layer so the renderer can consume them type-only.
export type MissionLifecyclePhase = 'created' | 'started' | 'checkpoint' | 'status_changed' | 'finalized'
export type RoomLifecyclePhase = 'started' | 'turn' | 'finalized'

export interface MissionLifecycleEvent {
  scope: 'mission'
  phase: MissionLifecyclePhase
  missionId: string
  status?: MissionStatus
  title?: string
  repeatMode?: MissionRepeatMode
  runIndex?: number
  reason?: string
  ts: number
}

export interface RoomLifecycleEvent {
  scope: 'room'
  phase: RoomLifecyclePhase
  roomId: string
  state?: RoomState
  seat?: string
  ts: number
}

export interface JobLifecycleEvent {
  scope: 'job'
  phase: 'status_changed'
  jobId: string
  status: JobStatus
  kind: JobKind
  missionId?: string | null
  roomId?: string | null
  ts: number
}

// Discriminated by `scope`; new scopes can be added without breaking consumers.
export type OrchestrationEvent = MissionLifecycleEvent | RoomLifecycleEvent | JobLifecycleEvent
