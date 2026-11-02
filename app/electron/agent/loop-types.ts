// Loop Mode contracts and observed execution results.

export type LoopMode = 'off' | 'build' | 'ship'
export type LoopPhaseId = 'product' | 'architect' | 'dev' | 'qa' | 'security' | 'browser' | 'docs' | 'review'
export type LoopAutonomy = 'read_only' | 'low' | 'medium' | 'high'
export type LoopContractSource = 'solo' | 'room' | 'ticket'

export interface RunContract {
  source: LoopContractSource
  goal: string
  deliverables: string[]
  acceptanceCriteria: string[]
  verificationCommands: string[]
  forbiddenShortcuts: string[]
  requiredEvidence: string[]
  interruptPolicy: string[]
  autonomy: LoopAutonomy
  maxCycles: number
}

export interface Verdict {
  passed: boolean
  confidence: number
  summary: string
  issues: string[]
}

export type VerificationOutcome = 'passed' | 'failed' | 'cancelled' | 'timed_out' | 'error'

export interface VerificationCheck {
  command: string
  outcome: VerificationOutcome
  exitCode: number | null
  output: string
}

export interface VerificationReport {
  checks: VerificationCheck[]
  cancelled: boolean
}

export interface LoopShellResult {
  code: number | null
  stdout: string
  stderr: string
  cancelled?: boolean
  timedOut?: boolean
  signal?: string | null
}

export type BlockerKind =
  | 'FIXABLE_BY_AGENT'
  | 'NEEDS_USER_SECRET'
  | 'NEEDS_PRODUCT_DECISION'
  | 'EXTERNAL_SERVICE_DOWN'
  | 'UNSAFE_ACTION'

export interface Blocker {
  kind: BlockerKind
  issue: string
  action: string
}

export interface PhaseResult {
  phase: LoopPhaseId
  ok: boolean
  // Tools that ran AND succeeded (state === 'ok'). Drives `ok`.
  toolEvents: number
  // Attempts distinguish a failed operation from a phase that never used a tool.
  toolAttempts: number
  summary: string
  evidenceRefs: string[]
}

// Default ordered phase stacks for surfaces without seats.
export const BUILD_PHASES: LoopPhaseId[] = ['architect', 'dev', 'qa', 'review']
export const SHIP_PHASES: LoopPhaseId[] = ['product', 'architect', 'dev', 'qa', 'security', 'browser', 'docs', 'review']
export const DEFAULT_MAX_CYCLES = { build: 5, ship: 7 } as const
export const HARD_MAX_CYCLES = 12
export const CRITIC_PASS_CONFIDENCE = 0.6

// Planning phases may complete without invoking tools.
export const EVIDENCE_OPTIONAL_PHASES: LoopPhaseId[] = ['product', 'architect']

export type LoopStatus = 'running' | 'shipped' | 'blocked' | 'capped' | 'failed' | 'stopped'

// One persisted record per Loop run.
export interface LoopRecord {
  id: string
  source: LoopContractSource
  surfaceId: string // threadId | roomId | ticketId | projectId
  goal: string
  mode: Exclude<LoopMode, 'off'>
  status: LoopStatus
  cwd: string
  branch: string
  phase: LoopPhaseId | 'verify' | null
  cycle: number
  maxCycles: number
  checkpoints: number
  lastVerdict: Verdict | null
  blockers: Blocker[]
  log: string[]
  roleThreadIds: string[]
  createdAt: number
  updatedAt: number
}

// IPC payload to start a Loop run.
export interface LoopStartInput {
  source: LoopContractSource
  surfaceId: string
  goal: string
  mode: Exclude<LoopMode, 'off'>
  baseCwd: string
  projectId?: string | null
  criteria?: string[]
  verify?: string[]
  workerModel?: { providerId: string; model: string }
  criticModel?: { providerId: string; model: string }
  ticket?: { acceptanceCriteria?: string[]; requiredEvidence?: string[]; allowedCapabilities?: string[]; risk?: string }
}
