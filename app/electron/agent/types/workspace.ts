export type {
  Seat,
  SeatHeartbeat,
  RoomSeatLane,
  AgentRoleProfileId,
  FanOutBudget,
  RoomCollaborationMode,
  RoomDeliberationEffort,
  RoomWorkerBudget,
  RoomMemoryEntry,
  RoomMemoryDiff,
  RoomManualMemoryCommand,
  RoomState,
  RoomEventItem,
  Room,
  RoomMergeStatus,
  RoomMergeConflictInfo,
  RoomMergeResult,
  RoomStreamEvent,
} from './rooms'
export type { RoomMode, RoomArtifactKind, RoomOutcomeProfile } from '../room-outcomes'
import type { ExportRedaction, ProjectKanbanSettings } from './timeline'
import type { ReasoningEffortSetting } from './settings'

export type ProjectExecutionTarget = 'local' | 'orchestrator'

export interface Project {
  id: string
  name: string
  cwd: string
  // Default remote workspace for chats/rooms/tickets under this project (overridable per task).
  workspace?: WorkspaceLocation | null
  /** Host-owned confinement retained for projects created by a paired device. */
  workspaceBoundary?: 'home'
  memory: string
  team: string
  // 'solo' = one model does the work (chats/tasks) · 'room' = multi-agent room
  mode?: 'solo' | 'room'
  // model policy: a guardrail/default for chats + rooms created in this project
  localOnly?: boolean // restrict to local providers (e.g. a "Secure" project)
  executionTarget?: ProjectExecutionTarget
  orchestratorId?: string | null
  defaultProviderId?: string | null
  defaultProviderAccountId?: string | null
  defaultModel?: string | null
  kanban?: ProjectKanbanSettings
  // User-defined tags for filtering (e.g. 'work', 'personal'). Blank = untagged.
  tags?: string[]
  archivedAt?: number | null
  createdAt: number
  updatedAt: number
}

// Typed, source-attributed project memory (build.md P1 / Pillar 6), distinct from the
// freeform Project.memory "charter" blob. Durable facts an agent reuses across runs.
export type ProjectMemoryKind = 'architecture' | 'convention' | 'api' | 'decision' | 'gotcha' | 'summary' | 'person' | 'preference'
export type ProjectMemorySourceType = 'manual' | 'file' | 'agent' | 'connector'
export type ProjectMemoryConfidence = 'low' | 'medium' | 'high'
export type ProjectMemoryPrivacyScope = 'project' | 'personal' | 'team'

// One superseded state of a memory entry (VST-009): pushed when an approval OVERWRITES an
// existing entry, so the change is reversible. Bounded per entry (newest first).
export interface ProjectMemoryRevision {
  text: string
  kind: ProjectMemoryKind
  sourceType?: ProjectMemorySourceType
  confidence: ProjectMemoryConfidence
  privacyScope: ProjectMemoryPrivacyScope
  sourceRef: string | null
  sourceIds?: string[]
  owner?: string | null
  lastVerifiedAt?: number | null
  sourceThreadId?: string | null
  sourceTurn?: number | null
  // When this superseded state was replaced, and by whom.
  replacedAt: number
  replacedBy: string
}

export interface ProjectMemoryEntry {
  id: string
  projectId: string
  kind: ProjectMemoryKind
  text: string
  sourceType: ProjectMemorySourceType
  sourceRef: string | null
  sourceIds?: string[]
  confidence: ProjectMemoryConfidence
  privacyScope: ProjectMemoryPrivacyScope
  owner: string | null
  lastVerifiedAt: number | null
  // Provenance to the exact conversation that noticed the pattern (VST-009).
  sourceThreadId?: string | null
  sourceTurn?: number | null
  // Host-owned lifecycle state; absent in legacy records means active, revision 1.
  trashedAt?: number | null
  stateRevision?: number
  // Superseded states, newest first, bounded — the rollback ledger (VST-009).
  revisions?: ProjectMemoryRevision[]
  createdAt: number
  updatedAt: number
}

export type ProjectMemoryCandidateStatus = 'pending' | 'approved' | 'rejected' | 'rolled_back'
export interface ProjectMemoryCandidate {
  id: string
  projectId: string
  kind: ProjectMemoryKind
  text: string
  sourceType: ProjectMemorySourceType
  sourceRef: string | null
  sourceIds?: string[]
  confidence: ProjectMemoryConfidence
  privacyScope: ProjectMemoryPrivacyScope
  owner: string | null
  reason: string | null
  status: ProjectMemoryCandidateStatus
  targetEntryId: string | null
  reviewedAt: number | null
  reviewedBy: string | null
  durableEntryId: string | null
  // Provenance to the exact conversation/turn that proposed this (VST-009).
  sourceThreadId?: string | null
  sourceTurn?: number | null
  createdAt: number
  updatedAt: number
}

// What approving a candidate would DO to durable memory (VST-009): shown in the review UI
// so approval is an informed action, never a blind one.
export interface ProjectMemoryCandidateDiff {
  candidateId: string
  action: 'create' | 'update'
  targetEntryId: string | null
  changes: Array<{ field: 'text' | 'kind' | 'confidence' | 'privacyScope' | 'sourceType' | 'sourceRef' | 'sourceIds' | 'owner' | 'lastVerifiedAt' | 'sourceThreadId' | 'sourceTurn'; before: string | null; after: string | null }>
}

export type {
  AppendSubAgentMessageInput,
  CreateSubAgentInput,
  ReadSubAgentMessagesInput,
  SubAgent,
  SubAgentEvent,
  SubAgentEventKind,
  SubAgentLaunchInput,
  SubAgentLaunchSource,
  SubAgentParentGroup,
  SubAgentParentKind,
  SubAgentParentRef,
  SubAgentPatch,
  SubAgentMessage,
  SubAgentMessageAudience,
  SubAgentMessageKind,
  SubAgentMessageSender,
  SubAgentStatus,
} from '../subagents'

export type TicketStatus = 'todo' | 'ready' | 'in_progress' | 'needs_clarification' | 'review' | 'completed'
export type TicketRisk = 'low' | 'medium' | 'high'
export type TicketCapability = 'read' | 'write' | 'shell' | 'network' | 'browser' | 'connector' | 'mcp'
export type TicketEvidenceKind = 'file' | 'diff' | 'command' | 'test' | 'review' | 'source' | 'note'
export type TicketDependencyFailurePolicy = 'block' | 'continue'

export interface TicketEvidence {
  id: string
  kind: TicketEvidenceKind
  summary: string
  ref?: string | null
  createdBy?: string | null
  createdAt: number
}

// How a kanban ticket is executed when you press Build (#55): a single agent on
// the configured model, a single agent forced onto the best frontier (cloud)
// model, or a multi-agent room. Defaults to 'solo' — building never silently
// spins up a room.
// Where a ticket EXECUTES when you press Build (VST-018): a solo agent on the configured
// model, a solo agent on the best frontier cloud model, a multi-agent room, the project's
// remote Orchestrator host, or an external CLI engine (codex/claude/… CLI transport).
export type TicketExecutionMode = 'solo' | 'frontier' | 'room' | 'remote' | 'cli'

// Execution tracking for a ticket (VST-018): what actually ran, where, and what it proved.
// One record per ticket, updated at run start and settle — the kanban card's ground truth
// for branch / run / checks / blockers / review verdict.
export interface TicketExecutionRecord {
  target: TicketExecutionMode
  hostId?: string | null
  providerId?: string | null
  model?: string | null
  reasoningEffort?: ReasoningEffortSetting
  branch?: string | null
  threadId?: string | null
  /** Host-issued run identity for lifecycle correlation. */
  runId?: string | null
  /** Timeline length frozen before that run began; older output is never completion evidence. */
  timelineBaseline?: number
  startedAt: number
  finishedAt: number | null
  checks: { command: string; passed: boolean }[]
  blockers: string[]
  // 'pending' = built, awaiting review; set by the review phase (or a room's review seat).
  reviewVerdict: 'pending' | 'approved' | 'changes_requested' | null
}

export interface Ticket {
  id: string
  projectId: string
  roomId?: string | null
  executionMode?: TicketExecutionMode
  botId?: string | null
  title: string
  body: string
  type: 'epic' | 'task' | 'subticket' | 'bug'
  status: TicketStatus
  priority: 'low' | 'medium' | 'high' | 'urgent'
  effort: 'small' | 'medium' | 'large'
  tags: string[]
  risk?: TicketRisk
  allowedCapabilities?: TicketCapability[]
  requiredEvidence?: string[]
  acceptanceCriteria?: string[]
  evidence?: TicketEvidence[]
  execution?: TicketExecutionRecord | null
  dependsOn?: string[]
  dependencyFailurePolicy?: TicketDependencyFailurePolicy
  parentId: string | null
  assignee: string | null
  threadId: string | null
  runState: 'queued' | 'running' | 'done' | 'cancelled' | null
  startedAt?: number | null
  order: number
  createdAt: number
  updatedAt: number
  // Remote snapshot tombstone. Tickets are normally removed outright on the host;
  // reconnect projections synthesize this marker so paired clients remove stale rows.
  archivedAt?: number | null
  /** Origin host for an Orchestrator-owned renderer projection/archive record. */
  orchestratorHostId?: string | null
}

export type TicketRemoteLifecycleCommand =
  | { action: 'begin'; threadId: string; hostId: string }
  | { action: 'sync' }
  | { action: 'start_failed'; message?: string }
  | { action: 'cancel' }

// ---- Automations (model-pinned, scheduled headless runs) ----
export type AutomationSchedule = 'interval' | 'daily' | 'manual'
// 'chat'/'agent' run a model; 'digest' composes a deterministic daily briefing from recent
// host activity (no model call) and delivers it via deliverTo.
export type AutomationMode = 'chat' | 'agent' | 'digest'
export type AutomationDeliver = 'inbox' | 'telegram' | 'email' | 'whatsapp'

export interface AutomationDeliveryTarget {
  connectorId: Exclude<AutomationDeliver, 'inbox'>
  accountId: string | null
  recipient: string
  subject?: string
}

export interface AutomationSourceScope {
  connectorId: import('./connectors').ConnectorId
  sourceId: string
}

// The safety-relevant posture frozen at a run's START (VST-038): a later edit to the
// automation definition can't change what an in-flight run was permitted to do.
export interface AutomationPermissionSnapshot {
  approvalMode: string
  sandbox: string | null
  worktreeMode: string
  acknowledgedRisks: string[]
  completionPolicyVersion: number
  capturedAt: number
}

// The completion verdict a finished run recorded (VST-038): for an ai_evaluated policy, the
// model's judgement of whether the objective is met; for others, the deterministic verdict.
export type AutomationCompletionVerdict = 'complete' | 'continue' | 'needs_review'

export interface AutomationRun {
  id: string
  startedAt: number
  finishedAt: number | null
  status: 'running' | 'ok' | 'error'
  output: string
  error?: string
  deliveryEventId?: string
  deliveryStatus?: 'inbox' | 'pending_approval' | 'blocked'
  deliveryError?: string
  // Frozen permission posture + completion verdict (VST-038; optional → older runs lack them).
  permissions?: AutomationPermissionSnapshot
  completion?: { verdict: AutomationCompletionVerdict; reason: string }
}

// Reusable instruction invoked with $ in the composer (a SKILL.md file).
export interface Skill {
  id: string
  name: string
  description: string
  body: string // the instruction the agent should follow
  enabled: boolean
  // When to reach for this skill (VST-010): travels with the skill summary so the agent
  // selects well BEFORE viewing the body. References/scripts/templates/examples/tests live
  // as markdown sections inside the body (parseSkillSections lifts them for rendering).
  trigger?: string
  // 'global' (default) applies everywhere; 'project' skills exist only inside their
  // project and require per-content trust (repo-authored — see skill-trust.ts).
  scope?: 'global' | 'project'
  projectId?: string | null
  createdAt: number
  updatedAt: number
}

// A repo-authored skill as surfaced to the trust UI (VST-010): the skill plus its live
// content hash and review state. Only 'trusted' ones join a run's context.
export interface ProjectSkillView {
  skill: Skill
  contentHash: string
  trust: 'trusted' | 'untrusted' | 'changed'
}

// What accepting a skill suggestion would DO (VST-010): mirrors ProjectMemoryCandidateDiff.
export interface SkillSuggestionDiff {
  suggestionId: string
  action: 'create' | 'update'
  targetSkillId: string | null
  changes: Array<{ field: 'name' | 'description' | 'body' | 'argHint'; before: string | null; after: string | null }>
}

// First-class artifacts (VST-011): typed outputs a run produced, with provenance and a
// redaction status — documents, PDFs, sheets, decks, images, generated files, reports,
// screenshots, logs, and run exports. Content lives on disk (path) or inline (bounded).
export type ArtifactKind = 'document' | 'pdf' | 'spreadsheet' | 'presentation' | 'image' | 'file' | 'report' | 'screenshot' | 'log' | 'export'
export type ArtifactRedactionStatus = 'unredacted' | 'redacted' | 'not_required'

export interface ArtifactProvenance {
  threadId: string
  runId?: string | null
  turn?: number | null
  tool?: string | null
  // Snapshot of the filesystem that owned a path-backed artifact when it was
  // registered. Optional only for records written before workspace provenance
  // was introduced; callers must treat an absent legacy value as untrusted.
  workspace?: WorkspaceLocation | null
  // Free-form pointer to the source data (a command, a URL, an input file).
  sourceRef?: string | null
}

export interface Artifact {
  id: string
  title: string
  kind: ArtifactKind
  // Exactly one of: a file path (content stays on disk) or bounded inline text.
  path: string | null
  inlineText: string | null
  inlineTruncated: boolean
  mimeType: string | null
  sizeBytes: number | null
  provenance: ArtifactProvenance
  redaction: ArtifactRedactionStatus
  redactionReport: ExportRedaction[]
  createdAt: number
  updatedAt: number
}

// Bounded artifact preview (VST-011): text excerpt for textual kinds, metadata line for
// binary ones.
export interface ArtifactPreview {
  kind: ArtifactKind
  title: string
  excerpt: string | null
  truncated: boolean
  meta: string
}

// Custom slash command invoked with / — expands to its prompt (+ any args).
export interface Command {
  id: string
  name: string // invoked as /name
  description: string
  prompt: string
  argHint?: string
  createdAt: number
  updatedAt: number
}

// A consent-first suggestion the agent files after a reusable workflow. It is NEVER
// created automatically — the user accepts it into a real Skill or Command.
export type SkillSuggestionKind = 'skill' | 'command'
export type SkillSuggestionStatus = 'pending' | 'accepted' | 'dismissed'
export interface SkillSuggestion {
  id: string
  kind: SkillSuggestionKind
  name: string
  description: string
  body: string // skill instruction OR command prompt
  argHint?: string // commands only
  rationale: string // why the agent thinks this is worth saving (shown to the user)
  sourceThreadId?: string | null
  // When set, this is an in-place IMPROVEMENT of an existing skill (kind:'skill'):
  // accepting it rewrites that skill's body/description rather than minting a new one.
  targetSkillId?: string | null
  status: SkillSuggestionStatus
  createdAt: number
  updatedAt: number
  acceptedId?: string | null // id of the Skill/Command created on accept
}

export interface Automation {
  id: string
  name: string
  prompt: string
  // schedule
  scheduleKind: AutomationSchedule
  everyMinutes: number // interval cadence
  atHour: number // daily wall-clock hour 0-23
  atMinute: number // daily wall-clock minute 0-59
  // Explicit IANA timezone for new daily schedules. Absent only for legacy host-local
  // schedules, whose interpretation must be labelled rather than silently migrated.
  timeZone?: string
  // Host-owned scheduling diagnostic. An invalid persisted timezone prevents dispatch.
  scheduleError?: string | null
  // execution — pin to a specific provider+model (local or cloud)
  providerId: string | null
  providerAccountId?: string | null
  model: string | null
  mode: AutomationMode // chat = one completion · agent = full tool loop in cwd
  cwd: string | null
  deliverTo: AutomationDeliver
  deliveryTarget?: AutomationDeliveryTarget
  /** Personal routines are inbox-first previews until explicitly configured otherwise. */
  previewOnly?: boolean
  botId?: string
  sourceScope?: AutomationSourceScope[]
  routineKind?: 'morning-brief' | 'meeting-prep' | 'follow-up'
  enabled: boolean
  // bookkeeping
  lastRunAt: number | null
  nextRunAt: number | null
  lastStatus: 'ok' | 'error' | null
  runs: AutomationRun[] // most-recent first, capped
  // Industrial-grade fields (VST-038; all optional → older automations behave as before):
  // misfire policy for a host that was asleep across fire times ('coalesce' default → run
  // once; 'run_each' → catch up each, capped), and risk gating (a run whose required risks
  // aren't all acknowledged is refused — e.g. `local-checkout` edits your real working copy).
  misfire?: 'coalesce' | 'run_each'
  requiredRisks?: string[]
  acknowledgedRisks?: string[]
  // AI-evaluated (or deterministic) completion policy (VST-038). Absent → a single pass
  // ('none'). Versioned: editing the policy bumps `version`, and a run that started under an
  // older version forces a needs_review verdict rather than trusting a changed contract.
  completionPolicy?: AutomationCompletionPolicy
  createdAt: number
  updatedAt: number
  // Synthesized by paired snapshot reconciliation when the host no longer has this row.
  archivedAt?: number | null
}

export type AutomationCompletionPolicy =
  | { type: 'none'; version: number }
  | { type: 'max_iterations'; version: number; max: number }
  | { type: 'on_signal'; version: number; signal: string }
  | { type: 'ai_evaluated'; version: number; prompt: string }

// ---- SSH remote workspaces ----
// A host the agent can run its file/shell/git tools on, over SSH. The model
// still runs against the configured provider; only tool execution moves to the
// remote box. No secrets are persisted here — passphrases/passwords are prompted
// at connect time and held only in memory.

export interface SshHost {
  id: string
  label: string // user-facing; defaults to the alias/hostName
  configAlias?: string | null // ~/.ssh/config Host alias this came from, if any
  hostName: string
  user?: string
  port: number
  identityFiles: string[] // paths (may contain ~), tried in order; empty => agent/default
  proxyJump?: string
  forwardAgent?: boolean
  lastRemoteDir?: string | null // remembered remote project dir for re-connect
  platform?: string | null // detected on first connect (e.g. 'linux'); informational
  origin: 'config' | 'manual' // mirrored from ~/.ssh/config vs added in-app
  createdAt: number
  updatedAt: number
}

export type WorkspaceLocationKind = 'local' | 'ssh'

// Where a thread/project/room's tools execute. Absent on an entity => local fs.
export interface WorkspaceLocation {
  kind: WorkspaceLocationKind
  hostId?: string | null // SshHost.id when kind === 'ssh'
  path: string // local fs path, or absolute remote path on the host
}

// Pinned server host key (StrictHostKeyChecking). Fingerprint only — no secret.
export interface TrustedSshHostKey {
  host: string // hostName the key was pinned for
  port: number
  keyType: string // e.g. 'ssh-ed25519'
  fingerprintSha256: string // base64 SHA-256 of the server public key
  trustedAt: number
}

export type SshConnectionPhase =
  | 'idle'
  | 'connecting'
  | 'authenticating'
  | 'awaiting_password' // passphrase/password prompt pending in the UI
  | 'awaiting_trust' // unknown host key, awaiting a trust decision
  | 'connected'
  | 'error'
  | 'disconnected'

export interface SshConnectionState {
  hostId: string
  phase: SshConnectionPhase
  message?: string | null
  remotePlatform?: string | null
  connectedAt?: number | null
  lastError?: string | null
  droppedAt?: number | null
}

// Prompt surfaced to the renderer when ssh2 needs a secret. The secret is
// supplied back over IPC and never written to disk.
export interface SshSecretPrompt {
  promptId: string
  hostId: string
  kind: 'passphrase' | 'password'
  label: string // e.g. "Passphrase for ~/.ssh/id_ed25519"
  retry: boolean // true after a failed attempt
}

// Unknown/changed host-key decision surfaced to the renderer.
export interface SshTrustPrompt {
  promptId: string
  hostId: string
  host: string
  port: number
  keyType: string
  fingerprintSha256: string
  changed: boolean // a different key is already pinned (possible MITM) — warn loudly
}
