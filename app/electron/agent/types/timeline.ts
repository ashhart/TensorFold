import type { ApiMessage, Attachment, ModelFailoverResetPolicy } from './base'
import type { Project, Room, Ticket, WorkspaceLocation } from './workspace'
import type { SubAgent } from '../subagents'
import type { ContextFragmentStat } from '../context-governance'
export type { ContextFragment, ContextFragmentPriority, ContextFragmentSource, ContextFragmentStat } from '../context-governance'

// ---- Visual artifacts (rendered inline in chat/code timelines) ----
export type VisualizationKind = 'flow' | 'bar' | 'stats' | 'timeline'
export type VisualizationTone = 'accent' | 'muted' | 'positive' | 'warning' | 'danger'

export interface VisualizationNode {
  id: string
  title: string
  subtitle?: string
  tone?: VisualizationTone
  column?: number
  row?: number
}

export interface VisualizationEdge {
  from: string
  to: string
  label?: string
}

export interface VisualizationBar {
  label: string
  value: number
  max?: number
  detail?: string
  tone?: VisualizationTone
}

export interface VisualizationMetric {
  label: string
  value: string
  detail?: string
  tone?: VisualizationTone
}

export interface VisualizationTimelineEvent {
  title: string
  time?: string
  detail?: string
  tone?: VisualizationTone
}

export interface VisualizationLegendItem {
  label: string
  tone?: VisualizationTone
}

export interface VisualizationSpec {
  kind: VisualizationKind
  title: string
  subtitle?: string
  caption?: string
  nodes?: VisualizationNode[]
  edges?: VisualizationEdge[]
  bars?: VisualizationBar[]
  metrics?: VisualizationMetric[]
  events?: VisualizationTimelineEvent[]
  legend?: VisualizationLegendItem[]
}

// ---- Trace context (renderer/main/tool/MCP/connectors/remote/webhook) ----
export type TraceSurface =
  | 'renderer'
  | 'main'
  | 'model'
  | 'tool'
  | 'mcp'
  | 'connector'
  | 'remote'
  | 'webhook'
  | 'mission'
  | 'room'
  | 'automation'

export interface TraceContext {
  traceId: string
  parentTraceId?: string | null
  runId?: string | null
  threadId?: string | null
  itemId?: string | null
  callId?: string | null
  surface: TraceSurface
  origin?: string | null
  at?: number | null
}

// ---- UI timeline (what the renderer renders) ----
export type TimelineKind = 'user' | 'cue' | 'assistant' | 'reasoning' | 'tool' | 'visualization' | 'error' | 'notice'
export type ToolState = 'pending' | 'awaiting' | 'running' | 'ok' | 'error' | 'denied'

export interface TimelineItem {
  question?: import('./user-question').UserQuestion
  id: string
  kind: TimelineKind
  text?: string // user / assistant markdown, reasoning, error
  // tool fields:
  tool?: string
  /** Raw provider tool-call ID, retained for model protocol and persisted tool responses. */
  callId?: string
  /** Opaque app approval identity, independent of provider IDs shared by concurrent workers. */
  approvalId?: string
  title?: string
  args?: Record<string, unknown>
  command?: string
  path?: string
  output?: string
  approval?: string
  visualization?: VisualizationSpec
  diff?: DiffLine[]
  /** Workspace-relative paths this completed tool demonstrably mutated. */
  changedPaths?: string[]
  state?: ToolState
  meta?: string // e.g. "exit 0 · 240ms"
  subAgent?: SubAgent // for launch_subagent/delegate lineage
  subAgentId?: string
  parentAgentId?: string
  status?: string
  trace?: TraceContext
  eventReceipt?: AgentEventReceipt
  runContract?: AgentRunContractSnapshot
  runReceipt?: AgentRunReceiptSnapshot
  createdAt?: number
  appAction?: AppAction // #42 — natural-language app control (create/navigate/setting)
  attachments?: Attachment[] // user message: documents + images attached from the composer
  /** run_command: what the sandbox captured, kept out of the model-visible output. */
  artifacts?: ToolArtifactSummary
  /** Final assistant item of a run: measured facts about the run's model calls. */
  stats?: AssistantTurnStats
}

/** Compact sandbox artifact record for a command's timeline item (the full bundle stays in main). */
export interface ToolArtifactSummary {
  id: string
  profile: string
  implementation: string
  inputFiles: number
  outputFiles: number
  screenshots: string[]
  diffs: Array<{ path: string; status: 'added' | 'changed' | 'deleted' }>
  patch: boolean
  logBytes: number
}

/** Real measurements for one run; tokens and timings are absent unless the server reported them. */
export interface AssistantTurnStats {
  model: string
  providerId: string | null
  modelCalls: number
  toolCalls: number
  /** Output tokens summed from the usage events of the calls that sent one. */
  completionTokens?: number
  /** Decode rate: the engine's own timings when it sent them, else the calls that streamed at least two tokens. */
  decodeTokensPerSecond?: number
  /** Seconds behind decodeTokensPerSecond. */
  decodeSeconds?: number
  /** New prompt tokens (prompt minus cached) over the engine's prefill seconds; absent when nothing new was read. */
  prefillTokensPerSecond?: number
  /** Prompt tokens summed over the calls whose usage reported them. */
  promptTokens?: number
  /** Of promptTokens, the ones the server reused from its cache. */
  cachedTokens?: number
  /** Request to first token for the run's first model call: the engine's figure when sent, else measured here. */
  firstTokenMs?: number
  /** Present when the built-in engine ran Cortheon for this run: its web searches, pages read and sources. */
  cortheon?: { task: string | null; searches: number; pagesRead: number; sources: string[] }
}

// An app-control action the chat agent requested. The renderer executes it:
// create_* and open_view apply directly; change_setting raises a confirm chip
// (the user approves before it's applied). `state` tracks confirm-first actions.
export interface AppAction {
  kind: 'create_project' | 'create_room' | 'create_kanban' | 'open_view' | 'change_setting'
  payload: Record<string, any>
  state?: 'pending' | 'applied' | 'dismissed'
}

export interface DiffLine {
  type: 'add' | 'del' | 'ctx' | 'hunk'
  text: string
  oldNo?: number
  newNo?: number
}

export type PlanStepStatus = 'pending' | 'in_progress' | 'done' | 'blocked' | 'dropped'

export interface PlanStep {
  id: string
  text: string
  done: boolean
  /** Granular status; `done` stays the canonical boolean for existing consumers. */
  status?: PlanStepStatus
  /** Present when status === 'blocked': what the step is waiting on. */
  blocker?: string
  /** 1 when this step is a subtask nested under the previous depth-0 step. */
  depth?: number
}

export interface SystemPromptSnapshot {
  sections: {
    base: string
    persona: string
    goal: string
    capabilities: string
  }
  fingerprint: string
  createdAt: number
  updatedAt: number
}

export type AgenticSurface = 'chat' | 'code' | 'assistant' | 'room' | 'ticket' | 'automation' | 'remote'
export type RunTargetKind = 'local' | 'remote'
export type RunVerificationVerdict = 'not_applicable' | 'unverified' | 'verified' | 'failed' | 'blocked' | 'cancelled'
export type AgenticSessionLifecycle = 'idle' | 'starting' | 'running' | 'awaiting_approval' | 'queued_followup' | 'suspended' | 'blocked' | 'complete' | 'error'
export type AgentEventReceiptKind = 'run_contract_saved' | 'tool_result_saved' | 'diff_finalized' | 'turn_complete' | 'job_complete' | 'run_receipt_saved'

export interface AgenticGuardPolicySnapshot {
  surface: AgenticSurface
  maxSteps: number
  repeatLimit: number
  searchLimit: number
  toolCadenceLimit: number
  noProgressLimit: number
  textLoopBreaker: boolean
  verifyOnStop: boolean
}

export interface AgenticSessionSnapshot {
  key: string
  source: 'thread' | 'connector' | 'room' | 'ticket' | 'automation' | 'remote'
  state: AgenticSessionLifecycle
  resumePending: boolean
  queueDepth: number
  pendingCueCount: number
  guidedCueCount: number
  resumeCursor: string | null
  resetPolicy: 'resume' | 'start_fresh'
  abortBehavior: 'preserve_queue' | 'mark_undrained'
  compactionCount: number
  lastCueAt: number | null
}

export interface AgentEventReceipt {
  id: string
  kind: AgentEventReceiptKind
  runId: string
  threadId: string
  itemId?: string | null
  state: 'pending' | 'ok' | 'error' | 'blocked' | 'cancelled'
  summary: string
  evidenceRefs: string[]
  createdAt: number
}

export interface AgentRunReceiptSnapshot {
  id: string
  runId: string
  threadId: string
  verdict: RunVerificationVerdict
  summary: string
  changedFiles: string[]
  commands: string[]
  tests: string[]
  approvals: number
  diffs: number
  artifacts: string[]
  errors: number
  eventReceipts: AgentEventReceipt[]
  gaps: string[]
  warnings: string[]
  startedAt: number
  finishedAt: number
  needsReview: boolean
}

export interface AgentRunContractSnapshot {
  id: string
  runId: string
  objective: string
  surface: AgenticSurface
  threadId: string
  projectId: string | null
  cwd: string
  targetKind: RunTargetKind
  targetLabel: string
  branch: string | null
  providerId: string | null
  providerAccountId: string | null
  model: string | null
  approvalMode: string
  permissions: string[]
  contextEpoch: number
  budget: { maxSteps: number; maxWallMs: number | null }
  guardPolicy: AgenticGuardPolicySnapshot
  session: AgenticSessionSnapshot
  createdAt: number
}

export interface Thread {
  bot?: import('./settings').BotThreadConfig
  // Once phone access is granted, its derived conversation stays on a local model.
  phoneLocalOnly?: boolean
  id: string
  title: string
  kind: 'task' | 'chat'
  projectId?: string | null
  cwd: string
  // Resume cursor for a persistent Codex app-server session, so a relaunched app continues the same
  // provider conversation (prompt cache, history) instead of starting fresh. Absent => no session yet.
  codexResumeCursor?: { threadId: string } | null
  // Same idea for Claude Code: the stream-json session id persisted so the next turn (or a
  // future app run) resumes the same `claude` session via `--resume` instead of re-flattening.
  claudeResumeCursor?: { sessionId: string } | null
  // Same idea for OpenCode: the `opencode serve` session id persisted so the next turn
  // reuses the same server-side session instead of creating a fresh one.
  openCodeResumeCursor?: { sessionId: string } | null
  // Same idea for Pi: the session id persisted so the next turn passes --session-id
  // (create-or-resume) and pi reattaches its own session file.
  piResumeCursor?: { sessionId: string } | null
  // Same idea for Cursor Agent: the chat id persisted so the next turn passes --resume.
  cursorResumeCursor?: { sessionId: string } | null
  // OMP engine (VST-050): run this thread ON the installed omp coding agent via its
  // headless rpc mode instead of TensorFold's native loop. Absent => native engine.
  engine?: 'native' | 'omp'
  // omp persists its own session file; resume it on the next run with --resume.
  ompResumeCursor?: { sessionId: string; sessionPath?: string } | null
  // CLIENT-side origin marker (VST-041): the remote Orchestrator host profile this thread
  // lives on. Stamped when a thread is created on / mirrored from a host, so runs, guides,
  // and cancels route to the thread's REAL home even if the project pin or the global
  // execution mode changes later. Absent = this machine owns the thread.
  orchestratorHostId?: string | null
  // Optional remote workspace binding; absent => local fs at cwd.
  workspace?: WorkspaceLocation | null
  /** Host-owned confinement inherited from a paired creation or project. */
  workspaceBoundary?: 'home'
  // Lineage (VST-049): the thread this one was forked or handed off from. Absent
  // => a normal first-class thread. Set by createForkThread and the thread:handoff
  // handler so the UI can show a breadcrumb back to the source. forkedAt is the
  // epoch ms of the fork/handoff; the kind (fork vs handoff) is read from the
  // checkpoint ledger / spine event, not stored here, to keep the Thread shape flat.
  parentId?: string | null
  forkedAt?: number | null
  providerId: string | null
  providerAccountId?: string | null
  modelRoute?: ModelRouteMemory | null
  model: string | null
  createdAt: number
  updatedAt: number
  status: 'idle' | 'running' | 'error'
  apiMessages: ApiMessage[]
  // Monotonic epoch for an AUTHORITATIVE conversation replacement (for example a
  // checkpoint revert). Ordinary streaming updates keep the same epoch and are merged
  // defensively in the renderer; a higher epoch tells projections to accept truncation.
  conversationRevision?: number
  systemPromptSnapshot?: SystemPromptSnapshot | null
  timeline: TimelineItem[]
  activeRunContract?: AgentRunContractSnapshot | null
  lastRunReceipt?: AgentRunReceiptSnapshot | null
  // Goal mode (right-panel Goal + Progress)
  goal?: string
  goalStatus?: 'active' | 'complete' | 'paused'
  plan?: PlanStep[]
  // Right-panel Notepad: free-form per-thread notes, persisted with the
  // thread so they survive restarts and travel with handoffs.
  notepad?: string
  // Thread annotations (VST-040): pinned messages + markers + notes, serialized (the pure
  // core is thread-annotations.ts). Persisted with the thread; pruned when a revert removes
  // the items they point at.
  annotations?: string
  // Incognito thread: lives only in memory. Persistence (threads.json) and the
  // event spine skip it entirely, so it vanishes on relaunch and leaves no
  // journal trace. Set at creation from the composer's Incognito toggle.
  incognito?: boolean
  archivedAt?: number | null
  // Compaction ledger: a record of each time older turns were folded into a summary.
  compactions?: CompactionLedgerEntry[]
  // Context-fragment ledger (VST-016): text-free stats for the bounded blocks assembled
  // into this thread's context at the last run start (+ tool-output spills as they happen).
  contextFragments?: ContextFragmentStat[]
  // Workbench done-gate (VST-017): when the user last marked this thread's diffs as
  // reviewed. Only counts while fresher than the last run receipt — a new run resets it.
  diffReviewedAt?: number | null
  // Anti-thrash: consecutive compaction passes that saved well under the worthwhile
  // threshold, for THIS thread. compactThread uses it to back off once compaction
  // stops helping, instead of re-paying for a fold that won't shrink much. Reset to
  // 0/undefined the moment a pass saves a useful amount again.
  compactionIneffectiveStreak?: number
}

/** Fields the renderer is allowed to edit through the generic thread patch IPC. */
export type RendererThreadPatch = Partial<Pick<Thread,
  'goal' | 'goalStatus' | 'plan' | 'notepad' | 'annotations' | 'diffReviewedAt' | 'incognito'
>>

export interface ModelRouteMemory {
  preferredProviderAccountId: string | null
  failoverProviderAccountId: string | null
  resetPolicy: ModelFailoverResetPolicy
  failedOverAt: number
  resetAt: number | null
  reason: string
}

export interface TicketWorkOptions {
  providerId?: string | null
  providerAccountId?: string | null
  model?: string | null
  phase?: 'build' | 'review'
}

export interface KanbanPhaseModel {
  providerId?: string | null
  providerAccountId?: string | null
  model?: string | null
}

export interface ProjectKanbanSettings {
  modelMode?: 'single' | 'phased'
  single?: KanbanPhaseModel
  draft?: KanbanPhaseModel
  ticketReview?: KanbanPhaseModel
  review?: KanbanPhaseModel
  build?: KanbanPhaseModel
  finalGate?: KanbanPhaseModel
}

// One folded-context event. This keeps context folding auditable.
export interface CompactionLedgerEntry {
  id: string
  at: number
  trigger: 'auto' | 'manual'
  messagesFolded: number
  tokensBefore: number
  tokensAfter: number
  summaryExcerpt: string
  kindsDropped: string[]
  summaryMethod?: 'verified-model' | 'verifier-rewrite' | 'deterministic-fallback'
  verificationNote?: string
  // Context epoch AFTER this fold (1-based; VST-016). Optional: pre-epoch ledgers persist.
  epoch?: number
}

// ── Phase 7: Run export & sharing ────────────────────────────────────────────
// A portable, redacted snapshot of a thread's run history (prompts, tool calls,
// diffs, outcomes). Secret redaction is ON by default; share/eval modes also
// anonymize identifiers. See electron/agent/export.ts + export-redaction.ts.
export type ExportMode = 'full' | 'share' | 'eval'
export type ExportRedactionKind =
  | 'private_key' | 'jwt' | 'api_key' | 'aws_key' | 'github_token' | 'slack_token'
  | 'google_key' | 'bearer' | 'assignment' | 'opaque_secret' | 'email' | 'home_path' | 'ip'
export interface ExportRedaction { kind: ExportRedactionKind; count: number }
export interface RunExportMessage { role: string; content: string }
export interface RunExportEvent {
  id?: string
  kind: TimelineKind
  text?: string
  tool?: string
  title?: string
  command?: string
  path?: string
  output?: string
  diff?: DiffLine[]
  state?: ToolState
  trace?: TraceContext
  at?: number
}
export interface EvidenceFileRef { path: string; itemIds: string[]; at?: number }
export interface EvidenceDiffRef { itemId: string; path?: string; summary?: string; lines: DiffLine[]; at?: number }
export interface EvidenceCommandRef { itemId: string; command: string; output?: string; state?: ToolState; meta?: string; at?: number }
export interface EvidenceScreenshotRef { itemId: string; path?: string; title?: string; at?: number }
export interface EvidenceCitationRef { itemId: string; url: string; title?: string; at?: number }
export interface EvidenceApprovalRef { itemId: string; tool?: string; state?: ToolState; text?: string; at?: number }
export interface EvidenceTraceRef extends TraceContext { itemId: string; kind: TimelineKind; tool?: string; title?: string; at?: number }
export interface EvidenceAssistantClaimRef {
  itemId: string
  claim: string
  sourceItemIds: string[]
  supported: boolean
  at?: number
}
export interface EvidenceGap {
  kind: 'claim_without_evidence'
  itemId: string
  claim: string
  reason: string
  at?: number
}
export interface EvidenceBundle {
  touchedFiles: EvidenceFileRef[]
  diffs: EvidenceDiffRef[]
  commandOutputs: EvidenceCommandRef[]
  screenshots: EvidenceScreenshotRef[]
  citations: EvidenceCitationRef[]
  approvals: EvidenceApprovalRef[]
  traceIds: EvidenceTraceRef[]
  assistantClaims: EvidenceAssistantClaimRef[]
  gaps: EvidenceGap[]
}
export interface RunChronicleEntry {
  itemId: string
  kind: TimelineKind
  label: string
  detail?: string
  state?: ToolState
  trace?: TraceContext
  at?: number
}
export interface RunChronicle {
  summary: string
  startedAt?: number
  finishedAt?: number
  entries: RunChronicleEntry[]
}
export interface RunExportBundle {
  schema: 'vontra.run-export/1'
  mode: ExportMode
  exportedAt: number
  redacted: boolean
  anonymized: boolean
  thread: {
    id: string
    title: string
    kind: 'task' | 'chat'
    cwd?: string
    provider?: string | null
    model?: string | null
    createdAt: number
    updatedAt: number
    status: string
    goal?: string
    goalStatus?: string
  }
  messages: RunExportMessage[]
  timeline: RunExportEvent[]
  compactions?: CompactionLedgerEntry[]
  stats: { messages: number; toolCalls: number; diffs: number; denied: number; errors: number }
  redactionReport: ExportRedaction[]
  evidence?: EvidenceBundle
  chronicle?: RunChronicle
}
export interface ExportOptions { mode?: ExportMode; redact?: boolean; anonymize?: boolean; includeReasoning?: boolean }
export interface ExportResult { bundle: RunExportBundle; json: string }
export interface ExportSaveResult { saved: boolean; path?: string; bytes?: number; canceled?: boolean }
export interface ExportImportResult { bundle: RunExportBundle | null; error?: string }

// ── Phase 6: Production incident mode ────────────────────────────────────────
// A narrow, auditable incident lane: a read-only first-pass scan, an append-only
// timeline, approval-gated mitigations, and a generated postmortem. An Incident
// is its own entity (not a Room) so it never runs mutating tools by surprise.
export type IncidentSeverity = 'sev1' | 'sev2' | 'sev3' | 'sev4'
export type IncidentStatus = 'open' | 'investigating' | 'mitigating' | 'resolved' | 'closed'
export type IncidentEventKind = 'symptom' | 'log' | 'metric' | 'traffic' | 'command' | 'mitigation' | 'note' | 'scan'
export type IncidentScanStatus = 'idle' | 'running' | 'done' | 'error'
export type IncidentMitigationStatus = 'proposed' | 'approved' | 'rejected' | 'applied'
export interface IncidentEvent {
  id: string
  kind: IncidentEventKind
  text: string
  at: number
  source: 'user' | 'scan' | 'agent'
  meta?: string
}
export interface IncidentMitigation {
  id: string
  text: string
  rationale?: string
  status: IncidentMitigationStatus
  proposedAt: number
  decidedAt?: number
  // Set once an approved mitigation is executed as a guarded mission (mutations
  // still require per-step approval via the mission flow).
  missionId?: string
}
export interface Incident {
  id: string
  title: string
  severity: IncidentSeverity
  status: IncidentStatus
  cwd: string
  summary?: string
  createdAt: number
  updatedAt: number
  timeline: IncidentEvent[]
  mitigations: IncidentMitigation[]
  scan: { status: IncidentScanStatus; findings?: string; ranAt?: number; error?: string }
  postmortem?: string
  postmortemAt?: number
}
export interface CreateIncidentInput { title: string; severity?: IncidentSeverity; cwd?: string; summary?: string }

// Working-tree git state for the right-panel Git tools section.
export interface GitFileChange { path: string; status: 'A' | 'M' | 'D' | 'R' | '?'; additions: number; deletions: number }
export interface GitStatus { branch: string | null; files: GitFileChange[]; additions: number; deletions: number }
export interface GitDiffFile extends GitFileChange { diff: string }
export type GitHygieneSeverity = 'critical' | 'high' | 'medium' | 'low'
export type GitHygieneSource = 'working_tree' | 'git_history' | 'commit_message'
export type GitHygieneKind =
  | 'secret'
  | 'env_secret'
  | 'private_key'
  | 'email'
  | 'absolute_path'
  | 'personal_term'
  | 'ai_attribution'
  | 'sensitive_filename'
  | 'private_network'
  | 'large_or_binary'

export interface GitHygieneFinding {
  id: string
  source: GitHygieneSource
  kind: GitHygieneKind
  severity: GitHygieneSeverity
  file: string | null
  line: number | null
  commit: string | null
  excerpt: string
  match: string
  suggestion: string
  autoFixable: boolean
}

export interface GitHygieneSummary {
  critical: number
  high: number
  medium: number
  low: number
}

export interface GitHygieneReport {
  cwd: string
  scannedAt: number
  scannedFiles: number
  scannedBytes: number
  scannedCommits: number
  findings: GitHygieneFinding[]
  blocked: boolean
  summary: GitHygieneSummary
  notes: string[]
}

export interface GitHygieneSanitizeResult {
  ok: boolean
  changedFiles: string[]
  skippedFindings: string[]
}

export interface GitCommitResult {
  ok: boolean
  hygiene: GitHygieneReport
  message: string
}

export type ArchiveKind = 'project' | 'thread' | 'room' | 'ticket'

export interface ArchiveSnapshot {
  projects: Project[]
  threads: Thread[]
  rooms: Room[]
  tickets: Ticket[]
}
export interface GitRepositoryState {
  cwd: string
  exists: boolean
  insideWorkTree: boolean
  topLevel: string | null
  branch: string | null
  hasCommits: boolean
  message?: string
}

// ---- Instruction Graph (@-import resolution; build.md P0) ----
export type InstructionNodeKind = 'shared' | 'provider' | 'local' | 'imported'
export interface InstructionNode {
  id: string
  kind: InstructionNodeKind
  path: string
  content: string
  imports: string[]
  ignored: boolean
  tokenEstimate: number
}
export interface ResolvedInstructionGraph {
  rootPath: string
  providerId: string | null
  nodes: InstructionNode[]
  cycles: string[]
  resolvedContent: string
  tokenEstimate: number
}

// ---- Local token telemetry (Claude / Codex / OpenCode history) ----
export type TokenUsageAccountId = 'claude' | 'codex' | 'opencode'
export type TokenUsageWindowKey = 'five_hour' | 'today' | 'week' | 'month' | 'all'

export interface TokenUsageBreakdown {
  total: number
  input: number
  output: number
  cacheCreation: number
  cacheRead: number
  cached: number
  reasoning: number
}

export interface TokenUsageModel extends TokenUsageBreakdown {
  model: string
}

export interface TokenUsageAccount extends TokenUsageBreakdown {
  id: TokenUsageAccountId
  name: string
  latestAt: number | null
  models: TokenUsageModel[]
}

export interface TokenUsageWindow {
  key: TokenUsageWindowKey
  label: string
  startAt: number
  total: number
  accounts: TokenUsageAccount[]
  models: TokenUsageModel[]
}

export interface TokenUsagePoint {
  key: string
  ts: number
  total: number
  accounts: Record<TokenUsageAccountId, number>
}

export interface TokenUsageSnapshot {
  generatedAt: number
  latestAt: number | null
  active: boolean
  windows: Record<TokenUsageWindowKey, TokenUsageWindow>
  daily: TokenUsagePoint[]
  hourly: TokenUsagePoint[]
}

// ---- Local model context / KV pressure telemetry ----
export type ContextPressureStatus = 'idle' | 'healthy' | 'warm' | 'pressured' | 'critical'
export type ContextPressureAction = 'continue' | 'summarize_next' | 'compact_now' | 'reduce_pins' | 'split_thread'
export type ContextRegistrySource = 'known' | 'model_name' | 'settings' | 'fallback' | 'engine'

export interface ModelContextProfile {
  id: string
  providerId: string
  providerName: string
  model: string
  kind: 'local' | 'cloud'
  nominalContextTokens: number
  recommendedWorkingTokens: number
  kvCacheTokens: number
  source: ContextRegistrySource
  note: string
}

export interface ContextPressureGuidance {
  pinnedTokens: number
  recentTokens: number
  summarizableTokens: number
  toolTokens: number
  recommendedAction: ContextPressureAction
  reason: string
  cacheHint: string
  pinned: string[]
  summarizable: string[]
}

export interface ContextPressureThread {
  id: string
  title: string
  cwd: string
  providerId: string | null
  providerName: string
  model: string | null
  kind: 'local' | 'cloud'
  estimatedTokens: number
  contextWindowTokens: number
  workingLimitTokens: number
  pressureRatio: number
  status: ContextPressureStatus
  guidance: ContextPressureGuidance
  updatedAt: number
}

export interface ContextPressureSnapshot {
  generatedAt: number
  activeProviderId: string | null
  activeModel: string | null
  activeProfile: ModelContextProfile | null
  registry: ModelContextProfile[]
  threads: ContextPressureThread[]
  totals: {
    localThreads: number
    estimatedTokens: number
    pinnedTokens: number
    summarizableTokens: number
    maxPressureRatio: number
  }
}
