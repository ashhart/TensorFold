import { randomUUID } from 'node:crypto'
import type {
  AgentEventReceipt,
  AgentEventReceiptKind,
  AgenticGuardPolicySnapshot,
  AgenticSessionSnapshot,
  AgentRunContractSnapshot,
  AgentRunReceiptSnapshot,
  ApprovalMode,
  Provider,
  Thread,
  TimelineItem,
} from './types'

const WRITE_TOOLS = new Set(['write_file', 'edit_file', 'apply_patch', 'patch_file'])
const TEST_COMMAND_RE =
  /\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|typecheck|lint|build|verify|check)\b|\b(?:vitest|jest|mocha|ava|pytest|ruff|eslint|tsc|cargo\s+test|go\s+test|dotnet\s+test|mvn\s+test|gradle\s+test)\b/i
const CODE_SURFACES = new Set(['code', 'ticket', 'room', 'automation', 'remote'])

export function agenticSurfaceFromThread(thread: Thread): AgentRunContractSnapshot['surface'] {
  const surfaces = thread.timeline.map((item) => item.trace?.surface).filter(Boolean)
  if (surfaces.includes('room')) return 'room'
  if (surfaces.includes('automation')) return 'automation'
  if (surfaces.includes('connector')) return 'assistant'
  if (/^kanban ticket:/i.test(thread.goal ?? thread.title)) return 'ticket'
  if (thread.workspace?.kind === 'ssh' && thread.kind !== 'chat') return 'code'
  return thread.kind === 'chat' ? 'chat' : 'code'
}

export function guardPolicyForThread(thread: Thread): AgenticGuardPolicySnapshot {
  const surface = agenticSurfaceFromThread(thread)
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

export function buildAgentRunContract(input: {
  thread: Thread
  runId: string
  objective: string
  provider: Pick<Provider, 'id' | 'cliSandbox' | 'cliApprovalPolicy' | 'kind'> | null
  model: string | null
  providerAccountId?: string | null
  approvalMode: ApprovalMode
  maxWallMs?: number | null
  createdAt?: number
}): AgentRunContractSnapshot {
  const { thread, runId, provider, approvalMode } = input
  const guardPolicy = guardPolicyForThread(thread)
  const createdAt = input.createdAt ?? Date.now()
  const targetKind = thread.workspace?.kind === 'ssh' ? 'remote' : 'local'
  return {
    id: randomUUID(),
    runId,
    objective: clean(input.objective) || clean(thread.goal) || clean(thread.title) || 'Untitled run',
    surface: guardPolicy.surface,
    threadId: thread.id,
    projectId: thread.projectId ?? null,
    cwd: thread.cwd,
    targetKind,
    targetLabel: targetKind === 'remote'
      ? `${thread.workspace?.hostId ? `SSH ${thread.workspace.hostId}` : 'SSH host'} -> ${thread.workspace?.path || thread.cwd || '/'}`
      : (thread.cwd ? `Local -> ${thread.cwd}` : 'Local'),
    branch: null,
    providerId: provider?.id ?? thread.providerId ?? null,
    providerAccountId: input.providerAccountId ?? thread.providerAccountId ?? null,
    model: input.model ?? thread.model ?? null,
    approvalMode,
    permissions: permissionSummary(provider, approvalMode, targetKind),
    contextEpoch: thread.compactions?.length ?? 0,
    budget: { maxSteps: guardPolicy.maxSteps, maxWallMs: input.maxWallMs ?? null },
    guardPolicy,
    session: buildSessionSnapshot(thread, guardPolicy.surface, 'starting'),
    createdAt,
  }
}

export function buildSessionSnapshot(
  thread: Thread,
  surface = agenticSurfaceFromThread(thread),
  state: AgenticSessionSnapshot['state'] = thread.status === 'running' ? 'running' : thread.status === 'error' ? 'error' : 'idle',
): AgenticSessionSnapshot {
  const pendingCues = thread.timeline.filter((item) => item.kind === 'cue' && (!item.state || ['pending', 'running', 'awaiting'].includes(item.state)))
  const guidedCues = thread.timeline.filter((item) => item.kind === 'cue' && item.state === 'ok')
  const lastCueAt = pendingCues.concat(guidedCues).reduce<number | null>((latest, item) => Math.max(latest ?? 0, item.createdAt ?? 0), null)
  const resumeCursor = thread.codexResumeCursor?.threadId ?? null
  return {
    key: `${surface}:${thread.id}`,
    source: thread.workspace?.kind === 'ssh' ? 'remote' : surface === 'assistant' ? 'connector' : surface === 'room' ? 'room' : surface === 'ticket' ? 'ticket' : surface === 'automation' ? 'automation' : 'thread',
    state,
    resumePending: Boolean(resumeCursor && ['starting', 'running', 'awaiting_approval', 'queued_followup', 'suspended'].includes(state)),
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

export function buildEventReceipt(input: {
  kind: AgentEventReceiptKind
  runId: string
  threadId: string
  itemId?: string | null
  state?: AgentEventReceipt['state']
  summary: string
  evidenceRefs?: string[]
  createdAt?: number
}): AgentEventReceipt {
  return {
    id: randomUUID(),
    kind: input.kind,
    runId: input.runId,
    threadId: input.threadId,
    itemId: input.itemId ?? null,
    state: input.state ?? 'ok',
    summary: input.summary,
    evidenceRefs: input.evidenceRefs ?? [],
    createdAt: input.createdAt ?? Date.now(),
  }
}

export function buildAgentRunReceipt(input: {
  thread: Thread
  contract: AgentRunContractSnapshot
  timelineStart: number
  aborted?: boolean
  finishedAt?: number
}): AgentRunReceiptSnapshot {
  const finishedAt = input.finishedAt ?? Date.now()
  const items = input.thread.timeline.slice(input.timelineStart)
  const eventReceipts = items.flatMap((item) => item.eventReceipt ? [item.eventReceipt] : [])
  const changedFiles: string[] = []
  const commands: string[] = []
  const tests: string[] = []
  const artifacts: string[] = []
  const warnings: string[] = []
  const gaps: string[] = []
  let approvals = 0
  let pendingApprovals = 0
  let deniedActions = 0
  let diffs = 0
  let errors = 0
  let lastChangeOrder = -1
  let lastSuccessfulTestOrder = -1

  items.forEach((item, index) => {
    const order = index
    if (item.kind === 'error' || item.state === 'error') errors++
    if (item.kind === 'notice' && /guardrail|model kept repeating|Run .* reached/i.test(item.text ?? item.title ?? '')) warnings.push(clean(item.text ?? item.title))
    if (item.kind === 'visualization') artifacts.push(item.title ?? item.visualization?.title ?? 'visualization')
    if (item.kind !== 'tool') return
    if (item.state === 'awaiting' || item.approval) approvals++
    if (item.state === 'awaiting') pendingApprovals++
    if (item.state === 'denied') deniedActions++
    const completed = item.state === 'ok' || item.state === 'error'
    if (completed && item.diff?.length) diffs++
    const path = pathFromItem(item)
    if ((item.state === 'ok' && item.tool && WRITE_TOOLS.has(item.tool)) || (completed && item.diff?.length)) {
      if (path) changedFiles.push(path)
      lastChangeOrder = Math.max(lastChangeOrder, order)
    }
    if (completed && item.changedPaths?.length) {
      changedFiles.push(...item.changedPaths)
      lastChangeOrder = Math.max(lastChangeOrder, order)
    }
    if (item.command || item.tool === 'run_command') {
      const command = item.command ?? String(item.args?.command ?? '')
      if (command) commands.push(command)
      if (completed && TEST_COMMAND_RE.test(command)) {
        tests.push(command)
        if (item.state === 'ok') lastSuccessfulTestOrder = Math.max(lastSuccessfulTestOrder, order)
      }
    }
    if (isArtifactTool(item)) artifacts.push(item.title ?? item.path ?? item.tool ?? 'artifact')
  })

  const uniqueChangedFiles = unique(changedFiles)
  const requiresVerification = input.contract.guardPolicy.verifyOnStop && uniqueChangedFiles.length > 0
  const freshVerification = lastSuccessfulTestOrder > lastChangeOrder && tests.length > 0
  let verdict: AgentRunReceiptSnapshot['verdict']
  if (input.aborted) {
    verdict = 'cancelled'
    gaps.push('run cancelled')
  } else if (pendingApprovals > 0) {
    verdict = 'blocked'
    gaps.push('approval pending')
  } else if (errors > 0) {
    verdict = 'failed'
    gaps.push('runtime error')
  } else if (deniedActions > 0) {
    verdict = 'blocked'
    gaps.push('action denied')
  } else if (requiresVerification && !freshVerification) {
    verdict = 'unverified'
    gaps.push('fresh test, build, lint, or typecheck evidence missing')
    warnings.push('verify-on-stop did not find fresh verification after code changes')
  } else if (requiresVerification) {
    verdict = 'verified'
  } else {
    verdict = 'not_applicable'
  }

  return {
    id: randomUUID(),
    runId: input.contract.runId,
    threadId: input.thread.id,
    verdict,
    summary: deniedActions > 0 && pendingApprovals === 0 && verdict === 'blocked'
      ? `${deniedActions} action${deniedActions === 1 ? ' was' : 's were'} denied.`
      : receiptSummary(verdict, uniqueChangedFiles.length, tests.length, pendingApprovals),
    changedFiles: uniqueChangedFiles,
    commands: unique(commands),
    tests: unique(tests),
    approvals,
    diffs,
    artifacts: unique(artifacts),
    errors,
    eventReceipts,
    gaps: unique(gaps),
    warnings: unique(warnings),
    startedAt: input.contract.createdAt,
    finishedAt,
    needsReview: ['unverified', 'failed', 'blocked', 'cancelled'].includes(verdict),
  }
}

function permissionSummary(provider: Pick<Provider, 'cliSandbox' | 'cliApprovalPolicy' | 'kind'> | null, approvalMode: ApprovalMode, targetKind: string): string[] {
  const out = [`approval:${approvalMode}`, `target:${targetKind}`]
  out.push(provider?.cliSandbox ? `sandbox:${provider.cliSandbox}` : 'sandbox:app-boundary')
  if (provider?.cliApprovalPolicy) out.push(`cli-approval:${provider.cliApprovalPolicy}`)
  if (provider?.kind) out.push(`provider:${provider.kind}`)
  return out
}

function pathFromItem(item: TimelineItem): string | null {
  const value = item.path ?? item.args?.path ?? item.args?.file ?? item.args?.target
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function isArtifactTool(item: TimelineItem): boolean {
  return /screenshot|artifact|export|image|video|pdf|document|report/i.test(`${item.tool ?? ''} ${item.title ?? ''} ${item.path ?? ''}`)
}

function receiptSummary(verdict: AgentRunReceiptSnapshot['verdict'], changedCount: number, testCount: number, approvals: number): string {
  if (verdict === 'cancelled') return 'The run was cancelled before a final verdict.'
  if (verdict === 'blocked') return `${approvals || 1} approval${approvals === 1 ? ' is' : 's are'} waiting.`
  if (verdict === 'failed') return 'The run recorded an error before a clean final verdict.'
  if (verdict === 'unverified') return `${changedCount} changed file${changedCount === 1 ? '' : 's'} need fresh verification.`
  if (verdict === 'verified') return `${changedCount} changed file${changedCount === 1 ? '' : 's'} verified by ${testCount} check${testCount === 1 ? '' : 's'}.`
  return 'No code-change verification required.'
}

function unique(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const item = clean(value)
    if (!item || seen.has(item)) continue
    seen.add(item)
    out.push(item)
  }
  return out
}

function clean(value: string | null | undefined): string {
  return (value ?? '').replace(/\s+/g, ' ').trim()
}
