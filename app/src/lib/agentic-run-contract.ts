import type { GitStatus, Settings, Thread, TimelineItem } from './types'

export type AgenticSurface = 'chat' | 'code' | 'assistant' | 'room' | 'ticket' | 'automation' | 'remote'
export type RunTargetKind = 'local' | 'remote'
export type RunVerificationVerdict = 'not_applicable' | 'unverified' | 'verified' | 'failed' | 'blocked'
export type AgenticSessionLifecycle = 'idle' | 'running' | 'awaiting_approval' | 'queued_followup' | 'suspended' | 'blocked' | 'complete' | 'error'

export interface AgenticGuardPolicy {
  surface: AgenticSurface
  maxSteps: number
  repeatLimit: number
  searchLimit: number
  toolCadenceLimit: number
  noProgressLimit: number
  textLoopBreaker: boolean
  verifyOnStop: boolean
}

export interface AgenticSessionState {
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

export interface AgenticRunReceipt {
  verdict: RunVerificationVerdict
  summary: string
  changedFiles: string[]
  commandCount: number
  testCount: number
  approvalCount: number
  diffCount: number
  artifactCount: number
  errorCount: number
  lastEvidenceAt: number | null
  needsReview: boolean
  gaps: string[]
  warnings: string[]
}

export interface AgenticRunContract {
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
  approvalMode: Settings['approvalMode']
  permissions: string[]
  changedFiles: string[]
  tests: string[]
  artifacts: string[]
  costTokens: {
    prompt: number
    output: number
    total: number
  }
  pendingApprovals: number
  compactions: number
  guardPolicy: AgenticGuardPolicy
  session: AgenticSessionState
  receipt: AgenticRunReceipt
}

export interface BuildAgenticRunContractInput {
  thread: Thread
  git?: GitStatus | null
  status?: string | null
  settings?: Pick<Settings, 'providers' | 'activeProviderId' | 'activeProviderAccountId' | 'activeModel' | 'approvalMode' | 'allowTaskNetwork' | 'autoCompact'> | null
  pendingApprovals?: number
  surface?: AgenticSurface
}

interface TimelineEvidence {
  changedFiles: string[]
  tests: string[]
  artifacts: string[]
  commandCount: number
  testCount: number
  approvalCount: number
  diffCount: number
  artifactCount: number
  errorCount: number
  pendingTimelineApprovals: number
  lastEvidenceAt: number | null
  lastChangeOrder: number
  lastSuccessfulTestOrder: number
  warnings: string[]
}

const CODE_SURFACES = new Set<AgenticSurface>(['code', 'ticket', 'room', 'automation', 'remote'])
const WRITE_TOOLS = new Set(['write_file', 'edit_file', 'apply_patch', 'patch_file'])
const TEST_COMMAND_RE =
  /\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|typecheck|lint|build|verify|check)\b|\b(?:vitest|jest|mocha|ava|pytest|ruff|eslint|tsc|cargo\s+test|go\s+test|dotnet\s+test|mvn\s+test|gradle\s+test)\b/i

export function buildAgenticRunContract(input: BuildAgenticRunContractInput): AgenticRunContract {
  const { thread, git = null, status = null, settings = null } = input
  const persisted = thread.activeRunContract ?? null
  const surface = input.surface ?? persisted?.surface ?? surfaceFromThread(thread)
  const evidence = collectTimelineEvidence(thread.timeline)
  const changedFiles = mergeUnique([...(git?.files.map((file) => file.path) ?? []), ...evidence.changedFiles])
  const pendingApprovals = Math.max(input.pendingApprovals ?? 0, evidence.pendingTimelineApprovals)
  const receipt = thread.lastRunReceipt && !status ? receiptFromSnapshot(thread.lastRunReceipt) : deriveRunReceipt({ thread, git, status, surface, pendingApprovals, evidence, changedFiles })
  const provider = settings?.providers.find((candidate) => candidate.id === (thread.providerId ?? settings.activeProviderId)) ?? null
  const approvalMode = settings?.approvalMode ?? 'suggest'
  const targetKind = thread.workspace?.kind === 'ssh' ? 'remote' : 'local'

  return {
    objective: persisted?.objective ?? objectiveFromThread(thread),
    surface,
    threadId: thread.id,
    projectId: thread.projectId ?? null,
    cwd: thread.cwd,
    targetKind: persisted?.targetKind ?? targetKind,
    targetLabel: persisted?.targetLabel ?? targetLabel(thread),
    branch: persisted?.branch ?? git?.branch ?? null,
    providerId: persisted?.providerId ?? thread.providerId ?? settings?.activeProviderId ?? null,
    providerAccountId: persisted?.providerAccountId ?? thread.providerAccountId ?? settings?.activeProviderAccountId ?? null,
    model: persisted?.model ?? thread.model ?? settings?.activeModel ?? null,
    approvalMode,
    permissions: permissionSummary({ approvalMode, targetKind, provider, settings }),
    changedFiles,
    tests: evidence.tests,
    artifacts: evidence.artifacts,
    costTokens: estimateCostTokens(thread),
    pendingApprovals,
    compactions: thread.compactions?.length ?? 0,
    guardPolicy: persisted?.guardPolicy ?? guardPolicyForSurface(surface),
    session: deriveSessionState(thread, { status, pendingApprovals, surface }),
    receipt,
  }
}

export function surfaceFromThread(thread: Thread): AgenticSurface {
  const surfaces = thread.timeline.map((item) => item.trace?.surface).filter(Boolean)
  if (surfaces.includes('room')) return 'room'
  if (surfaces.includes('automation')) return 'automation'
  if (surfaces.includes('connector')) return 'assistant'
  if (/^kanban ticket:/i.test(thread.goal ?? thread.title)) return 'ticket'
  if (thread.workspace?.kind === 'ssh' && thread.kind !== 'chat') return 'code'
  return thread.kind === 'chat' ? 'chat' : 'code'
}

export function guardPolicyForSurface(surface: AgenticSurface): AgenticGuardPolicy {
  const common = {
    surface,
    repeatLimit: 3,
    searchLimit: 6,
    toolCadenceLimit: 16,
    noProgressLimit: 6,
    textLoopBreaker: true,
    verifyOnStop: CODE_SURFACES.has(surface),
  }
  if (surface === 'chat') return { ...common, maxSteps: 24, searchLimit: 4, toolCadenceLimit: 10, verifyOnStop: false }
  if (surface === 'assistant') return { ...common, maxSteps: 48, searchLimit: 5, verifyOnStop: false }
  if (surface === 'room') return { ...common, maxSteps: 64, toolCadenceLimit: 20 }
  if (surface === 'ticket') return { ...common, maxSteps: 80 }
  if (surface === 'automation') return { ...common, maxSteps: 40, searchLimit: 4 }
  return { ...common, maxSteps: 100 }
}

export function deriveSessionState(
  thread: Thread,
  input: { status?: string | null; pendingApprovals?: number; surface?: AgenticSurface } = {},
): AgenticSessionState {
  const pendingCues = thread.timeline.filter((item) => item.kind === 'cue' && (!item.state || ['pending', 'running', 'awaiting'].includes(item.state)))
  const guidedCues = thread.timeline.filter((item) => item.kind === 'cue' && item.state === 'ok')
  const lastCueAt = latestAt(thread.timeline.filter((item) => item.kind === 'cue'))
  const status = input.status ?? null
  const pendingApprovals = input.pendingApprovals ?? 0
  const surface = input.surface ?? surfaceFromThread(thread)
  const resumeCursor = thread.codexResumeCursor?.threadId ?? null

  let state: AgenticSessionLifecycle = 'idle'
  if (thread.status === 'error' || status === 'error') state = 'error'
  else if (thread.goalStatus === 'paused' || status === 'suspended') state = 'suspended'
  else if (pendingApprovals > 0 || status === 'awaiting') state = 'awaiting_approval'
  else if (pendingCues.length > 0) state = 'queued_followup'
  else if (thread.status === 'running' || ['thinking', 'streaming', 'tool'].includes(status ?? '')) state = 'running'
  else if (thread.goalStatus === 'complete') state = 'complete'

  return {
    key: `${surface}:${thread.id}`,
    source: sessionSource(surface, thread),
    state,
    resumePending: Boolean(resumeCursor && ['running', 'awaiting_approval', 'queued_followup', 'suspended'].includes(state)),
    queueDepth: pendingCues.length,
    pendingCueCount: pendingCues.length,
    guidedCueCount: guidedCues.length,
    resumeCursor,
    resetPolicy: resumeCursor ? 'resume' : 'start_fresh',
    abortBehavior: 'mark_undrained',
    compactionCount: thread.compactions?.length ?? 0,
    lastCueAt,
  }
}

function deriveRunReceipt(input: {
  thread: Thread
  git: GitStatus | null
  status: string | null
  surface: AgenticSurface
  pendingApprovals: number
  evidence: TimelineEvidence
  changedFiles: string[]
}): AgenticRunReceipt {
  const { thread, status, surface, pendingApprovals, evidence, changedFiles } = input
  const requiresVerification = CODE_SURFACES.has(surface) && changedFiles.length > 0
  const hasFreshVerification = evidence.lastSuccessfulTestOrder > evidence.lastChangeOrder && evidence.testCount > 0
  const gaps: string[] = []
  const warnings = [...evidence.warnings]
  let verdict: RunVerificationVerdict

  if (pendingApprovals > 0 || status === 'awaiting') {
    verdict = 'blocked'
    gaps.push('approval pending')
  } else if (thread.status === 'error' || status === 'error' || evidence.errorCount > 0) {
    verdict = 'failed'
    gaps.push('runtime error')
  } else if (requiresVerification && !hasFreshVerification) {
    verdict = 'unverified'
    gaps.push('fresh test, build, lint, or typecheck evidence missing')
  } else if (requiresVerification) {
    verdict = 'verified'
  } else {
    verdict = 'not_applicable'
  }

  if (changedFiles.length > 0 && evidence.diffCount === 0) warnings.push('changed files have no captured diff evidence')
  if (requiresVerification && evidence.testCount === 0) warnings.push('no verification command recorded')

  return {
    verdict,
    summary: receiptSummary(verdict, changedFiles.length, evidence.testCount, pendingApprovals),
    changedFiles,
    commandCount: evidence.commandCount,
    testCount: evidence.testCount,
    approvalCount: evidence.approvalCount,
    diffCount: evidence.diffCount,
    artifactCount: evidence.artifactCount,
    errorCount: evidence.errorCount,
    lastEvidenceAt: evidence.lastEvidenceAt,
    needsReview: verdict === 'unverified' || verdict === 'failed' || verdict === 'blocked',
    gaps,
    warnings: mergeUnique(warnings),
  }
}

function receiptFromSnapshot(snapshot: NonNullable<Thread['lastRunReceipt']>): AgenticRunReceipt {
  return {
    verdict: snapshot.verdict === 'cancelled' ? 'blocked' : snapshot.verdict,
    summary: snapshot.summary,
    changedFiles: snapshot.changedFiles,
    commandCount: snapshot.commands.length,
    testCount: snapshot.tests.length,
    approvalCount: snapshot.approvals,
    diffCount: snapshot.diffs,
    artifactCount: snapshot.artifacts.length,
    errorCount: snapshot.errors,
    lastEvidenceAt: snapshot.finishedAt,
    needsReview: snapshot.needsReview,
    gaps: snapshot.gaps,
    warnings: snapshot.warnings,
  }
}

function collectTimelineEvidence(items: TimelineItem[]): TimelineEvidence {
  const changedFiles: string[] = []
  const tests: string[] = []
  const artifacts: string[] = []
  const warnings: string[] = []
  let commandCount = 0
  let testCount = 0
  let approvalCount = 0
  let diffCount = 0
  let artifactCount = 0
  let errorCount = 0
  let pendingTimelineApprovals = 0
  let lastEvidenceAt: number | null = null
  let lastChangeOrder = -1
  let lastSuccessfulTestOrder = -1

  items.forEach((item, index) => {
    const order = index
    if (item.kind === 'error' || item.state === 'error') errorCount++
    if (item.kind !== 'tool') {
      if (item.kind === 'visualization') {
        artifactCount++
        artifacts.push(item.title ?? item.visualization?.title ?? 'visualization')
      }
      return
    }

    if (item.state && ['ok', 'error', 'denied'].includes(item.state)) lastEvidenceAt = newestDate(lastEvidenceAt, item.createdAt ?? null)
    if (item.diff?.length) diffCount++
    if (item.approval || item.state === 'awaiting') approvalCount++
    if (item.state === 'awaiting') pendingTimelineApprovals++
    // First-class artifacts (VST-011): a successful save_artifact registration is
    // receipt-visible evidence, exactly like a visualization.
    if (item.tool === 'save_artifact' && item.state === 'ok') {
      artifactCount++
      artifacts.push((item.title ?? 'artifact').replace(/^Artifact:\s*/, ''))
    }

    const path = filePathFromItem(item)
    if ((item.tool && WRITE_TOOLS.has(item.tool)) || item.diff?.length) {
      if (path) changedFiles.push(path)
      lastChangeOrder = Math.max(lastChangeOrder, order)
    }

    if (item.tool === 'run_command' || item.command) {
      commandCount++
      const command = item.command ?? String(item.args?.command ?? '')
      if (isTestCommand(command)) {
        testCount++
        tests.push(command)
        if (!item.state || item.state === 'ok') lastSuccessfulTestOrder = Math.max(lastSuccessfulTestOrder, order)
      }
    }

    if (isArtifactTool(item)) {
      artifactCount++
      artifacts.push(item.title ?? item.path ?? item.tool ?? 'artifact')
    }
    if (item.output && item.output.length > 16_000) warnings.push('large tool output was captured in the timeline')
  })

  return {
    changedFiles: mergeUnique(changedFiles),
    tests: mergeUnique(tests),
    artifacts: mergeUnique(artifacts),
    commandCount,
    testCount,
    approvalCount,
    diffCount,
    artifactCount,
    errorCount,
    pendingTimelineApprovals,
    lastEvidenceAt,
    lastChangeOrder,
    lastSuccessfulTestOrder,
    warnings,
  }
}

function objectiveFromThread(thread: Thread): string {
  const latestUser = [...thread.timeline].reverse().find((item) => (item.kind === 'user' || item.kind === 'cue') && item.text?.trim())
  return clean(thread.goal) || clean(latestUser?.text) || clean(thread.title) || 'Untitled run'
}

function targetLabel(thread: Thread): string {
  if (thread.workspace?.kind === 'ssh') {
    const host = thread.workspace.hostId ? `SSH ${thread.workspace.hostId}` : 'SSH host'
    return `${host} -> ${thread.workspace.path || thread.cwd || '/'}`
  }
  return thread.cwd ? `Local -> ${thread.cwd}` : 'Local'
}

function permissionSummary(input: {
  approvalMode: Settings['approvalMode']
  targetKind: RunTargetKind
  provider: Settings['providers'][number] | null
  settings: BuildAgenticRunContractInput['settings']
}): string[] {
  const { approvalMode, targetKind, provider, settings } = input
  const permissions = [
    `approval:${approvalMode}`,
    `target:${targetKind}`,
    `network:${settings?.allowTaskNetwork ? 'allowed' : 'blocked'}`,
    `auto-compact:${settings?.autoCompact === false ? 'off' : 'on'}`,
  ]
  if (provider?.cliSandbox) permissions.push(`sandbox:${provider.cliSandbox}`)
  else permissions.push('sandbox:app-boundary')
  if (provider?.cliApprovalPolicy) permissions.push(`cli-approval:${provider.cliApprovalPolicy}`)
  return permissions
}

function estimateCostTokens(thread: Thread): AgenticRunContract['costTokens'] {
  let promptChars = 0
  let outputChars = 0
  for (const message of thread.apiMessages) {
    const text = typeof message.content === 'string'
      ? message.content
      : Array.isArray(message.content)
        ? message.content.map((part) => part.type === 'text' ? part.text : '[image]').join('\n')
        : ''
    if (message.role === 'assistant' || message.role === 'tool') outputChars += text.length
    else promptChars += text.length
  }
  for (const item of thread.timeline) {
    if (item.kind === 'assistant' || item.kind === 'tool' || item.kind === 'reasoning') outputChars += (item.text?.length ?? 0) + (item.output?.length ?? 0)
    else promptChars += item.text?.length ?? 0
  }
  const prompt = Math.round(promptChars / 4)
  const output = Math.round(outputChars / 4)
  return { prompt, output, total: prompt + output }
}

function sessionSource(surface: AgenticSurface, thread: Thread): AgenticSessionState['source'] {
  if (thread.workspace?.kind === 'ssh') return 'remote'
  if (surface === 'assistant') return 'connector'
  if (surface === 'room') return 'room'
  if (surface === 'ticket') return 'ticket'
  if (surface === 'automation') return 'automation'
  if (surface === 'remote') return 'remote'
  return 'thread'
}

function filePathFromItem(item: TimelineItem): string | null {
  const value = item.path ?? item.args?.path ?? item.args?.file ?? item.args?.target
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function isTestCommand(command: string): boolean {
  return TEST_COMMAND_RE.test(command)
}

function isArtifactTool(item: TimelineItem): boolean {
  // First-class registrations (save_artifact, VST-011) are counted by their own precise
  // branch above — successful ones only, with clean names — so the heuristic skips them.
  if (item.tool === 'save_artifact') return false
  const label = `${item.tool ?? ''} ${item.title ?? ''} ${item.path ?? ''}`.toLowerCase()
  return /screenshot|artifact|export|image|video|pdf|document|report/.test(label)
}

function receiptSummary(verdict: RunVerificationVerdict, changedCount: number, testCount: number, pendingApprovals: number): string {
  if (verdict === 'blocked') return `${pendingApprovals || 1} approval or blocker is waiting.`
  if (verdict === 'failed') return 'The run recorded an error before a clean final verdict.'
  if (verdict === 'unverified') return `${changedCount} changed file${changedCount === 1 ? '' : 's'} need fresh verification.`
  if (verdict === 'verified') return `${changedCount} changed file${changedCount === 1 ? '' : 's'} verified by ${testCount} check${testCount === 1 ? '' : 's'}.`
  return 'No code-change verification required.'
}

function latestAt(items: TimelineItem[]): number | null {
  return items.reduce<number | null>((latest, item) => newestDate(latest, item.createdAt ?? null), null)
}

function newestDate(a: number | null, b: number | null): number | null {
  if (a == null) return b
  if (b == null) return a
  return Math.max(a, b)
}

function mergeUnique(values: string[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const value of values) {
    const cleanValue = clean(value)
    if (!cleanValue || seen.has(cleanValue)) continue
    seen.add(cleanValue)
    out.push(cleanValue)
  }
  return out
}

function clean(value: string | null | undefined): string {
  return (value ?? '').replace(/\s+/g, ' ').trim()
}
