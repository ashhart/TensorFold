import { buildRunExport } from './export'
import type { CreatePrLoopInput, PlanStep, Thread, ToolState } from './types'

export interface LocalPrLoopDraft {
  title: string
  objective: string
  stopCondition: string
  implementationTitle: string
  reviewTitle: string
  implementationGoal: string
  reviewGoal: string
  implementationPlan: PlanStep[]
  reviewPlan: PlanStep[]
  requiredApprovals: number
}

export interface LocalPrLoopSummaryContext {
  localPrId: string
  branch: string
  baseBranch: string | null
  worktree: string
  implementationThreadId: string
  reviewThreadId?: string | null
}

export interface LocalPrLoopEvidenceLink {
  itemId: string
  href: string
  label: string
}

export interface LocalPrLoopDiffLink extends LocalPrLoopEvidenceLink {
  path: string
  lineCount: number
  summary: string
}

export interface LocalPrLoopCheckLink extends LocalPrLoopEvidenceLink {
  command: string
  state?: ToolState
  passed: boolean
  meta?: string
}

export interface LocalPrLoopSummary {
  markdown: string
  changedFiles: string[]
  diffRefs: LocalPrLoopDiffLink[]
  checkRefs: LocalPrLoopCheckLink[]
  risks: string[]
  reviewInstructions: string[]
  readyForReview: boolean
}

export interface LocalPrReviewComment {
  id: string
  text: string
  sourceItemId: string
}

export interface LocalPrReviewApproval {
  id: string
  text: string
  sourceItemId: string
  reviewerId: string
  reviewPassId: string
  evidenceItemIds: string[]
}

export interface LocalPrReviewOutcome {
  approved: boolean
  comments: LocalPrReviewComment[]
  approvals: LocalPrReviewApproval[]
  sourceItemIds: string[]
  summary: string
}

export interface LocalPrApprovalEvidence {
  approved: boolean
  requiredApprovals: number
  recordedApprovals: number
  missingApprovals: number
  sourceItemIds: string[]
  reviewPassIds: string[]
  reviewerIds: string[]
  evidenceItemIds: string[]
  summary: string
}

export interface LocalPrRemotePublishGate {
  allowed: boolean
  requested: boolean
  approved: boolean
  reason: string
}

export function normalizeRequiredApprovals(value: unknown): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return 1
  return Math.max(1, Math.min(6, Math.round(n)))
}

export function buildLocalPrLoopDraft(input: CreatePrLoopInput, context: {
  localPrId: string
  projectName: string | null
  cwd: string
  worktree: string
  branch: string
  baseBranch: string | null
}): LocalPrLoopDraft {
  const title = clean(input.title) || 'Local PR loop'
  const objective = cleanLong(input.objective) || title
  const requiredApprovals = normalizeRequiredApprovals(input.requiredApprovals)
  const projectLine = context.projectName ? `Project: ${context.projectName}` : 'Project: local workspace'
  const base = context.baseBranch || 'current HEAD'
  const branchLines = [
    projectLine,
    `Local PR: ${context.localPrId}`,
    `Base: ${base}`,
    `Branch: ${context.branch}`,
    `Worktree: ${context.worktree}`,
    `Required evidence-bound review passes: ${requiredApprovals}`,
  ].join('\n')
  const stopCondition = [
    `The local PR branch ${context.branch} has all requested changes implemented.`,
    'The reviewer thread has reviewed the local diff and completed an evidence-bound pass.',
    requiredApprovals > 1 ? `At least ${requiredApprovals} separate evidence-bound review passes are recorded.` : 'At least 1 evidence-bound review pass is recorded.',
    'All relevant tests or checks have passed, or any skipped checks are clearly justified.',
  ].join(' ')

  const implementationGoal = [
    'You are the implementation thread in a TensorFold local PR loop.',
    branchLines,
    '',
    'Objective:',
    objective,
    '',
    'Work only inside the local PR worktree. Make focused changes, keep commits and comments clean, and do not publish remote branches or open remote PRs.',
    'Remote publish, push, or opening a hosted PR is outside this local loop and requires a separate explicit approval path.',
    'When the implementation is ready, produce a local PR summary with changed files, tests run, risks, and review instructions.',
    'After reviewer comments arrive through Guide or the mission journal, address every comment, rerun relevant checks, and post an updated local PR summary.',
    'Continue until the reviewer approves and the stop condition is met.',
  ].join('\n')

  const reviewGoal = [
    'You are the independent reviewer thread in a TensorFold local PR loop.',
    branchLines,
    '',
    'Review the local PR only after the implementation thread reports that it is ready.',
    `Compare ${base} to ${context.branch} from the worktree, inspect the changed files, and review for correctness, security, tests, regressions, and scope creep.`,
    'Do not edit files. Leave concrete review comments for the implementation thread.',
    'Approve only when the local diff is ready to merge locally and the evidence is good enough.',
    'Do not ask the builder to push, publish, or open a hosted PR from this loop.',
  ].join('\n')

  return {
    title,
    objective,
    stopCondition,
    requiredApprovals,
    implementationTitle: `Build local PR: ${title}`.slice(0, 96),
    reviewTitle: `Review local PR: ${title}`.slice(0, 96),
    implementationGoal,
    reviewGoal,
    implementationPlan: [
      step('Confirm local PR branch and worktree'),
      step('Implement the requested change'),
      step('Run relevant checks'),
      step('Post local PR summary for review'),
      step('Address reviewer comments until approved'),
    ],
    reviewPlan: [
      step('Wait for implementation ready signal'),
      step('Review local diff against base'),
      step('Leave actionable comments or approval'),
    ],
  }
}

export function buildLocalPrLoopSummary(thread: Thread, context: LocalPrLoopSummaryContext, now = Date.now()): LocalPrLoopSummary {
  const bundle = buildRunExport(thread, { mode: 'full', redact: true, anonymize: false, includeReasoning: false }, now)
  const evidence = bundle.evidence
  const diffRefs: LocalPrLoopDiffLink[] = (evidence?.diffs ?? [])
    .filter((diff) => diff.itemId)
    .map((diff) => {
      const path = clean(diff.path) || 'diff'
      const lineCount = diff.lines.length
      return {
        itemId: diff.itemId,
        href: timelineEvidenceHref(thread.id, diff.itemId),
        label: path,
        path,
        lineCount,
        summary: clean(diff.summary) || `${lineCount} diff line${lineCount === 1 ? '' : 's'}`,
      }
    })
    .slice(0, 12)

  const checkRefs: LocalPrLoopCheckLink[] = (evidence?.commandOutputs ?? [])
    .filter((command) => isCheckCommand(command.command))
    .map((command) => ({
      itemId: command.itemId,
      href: timelineEvidenceHref(thread.id, command.itemId),
      label: command.command,
      command: command.command,
      state: command.state,
      passed: command.state === 'ok' && !commandLooksFailed(`${command.command}\n${command.output ?? ''}\n${command.meta ?? ''}`),
      meta: clean(command.meta),
    }))
    .slice(-8)

  const changedFiles = unique([
    ...diffRefs.map((diff) => diff.path),
    ...(evidence?.touchedFiles ?? []).map((file) => file.path),
  ].filter(Boolean)).slice(0, 24)

  const risks = [
    ...(!diffRefs.length ? ['No diff evidence is recorded yet.'] : []),
    ...(!checkRefs.length ? ['No check command evidence is recorded yet.'] : []),
    ...checkRefs.filter((check) => !check.passed).map((check) => `Check may need attention: ${check.command}`),
    ...(evidence?.gaps ?? []).map((gap) => gap.reason),
  ].slice(0, 8)

  const reviewInstructions = [
    'Open every linked diff and check before approving.',
    `Compare ${context.baseBranch || 'base HEAD'} to ${context.branch} inside ${context.worktree}.`,
    'Review for correctness, regressions, scope creep, security, and missing tests.',
    'Leave actionable comments for the builder or explicitly approve when the evidence is sufficient.',
  ]

  const readyForReview = diffRefs.length > 0 && checkRefs.some((check) => check.passed) && risks.length === 0
  const markdown = [
    `## Local PR Summary: ${context.localPrId}`,
    '',
    `Branch: \`${context.branch}\``,
    `Base: \`${context.baseBranch || 'base HEAD'}\``,
    `Implementation thread: \`${context.implementationThreadId}\``,
    context.reviewThreadId ? `Review thread: \`${context.reviewThreadId}\`` : '',
    '',
    '### Changed Files',
    changedFiles.length ? changedFiles.map((file) => `- \`${file}\``).join('\n') : '- No changed file evidence recorded.',
    '',
    '### Diff Evidence',
    diffRefs.length ? diffRefs.map((diff) => `- [${diff.path}](${diff.href}) - ${diff.summary}; ${diff.lineCount} line${diff.lineCount === 1 ? '' : 's'}`).join('\n') : '- No diff evidence recorded.',
    '',
    '### Checks',
    checkRefs.length ? checkRefs.map((check) => `- [${check.command}](${check.href}) - ${check.passed ? 'passed' : 'needs attention'}${check.meta ? `; ${check.meta}` : ''}`).join('\n') : '- No check command evidence recorded.',
    '',
    '### Risks',
    risks.length ? risks.map((risk) => `- ${risk}`).join('\n') : '- No evidence gaps detected in the recorded timeline.',
    '',
    '### Review Instructions',
    reviewInstructions.map((instruction) => `- ${instruction}`).join('\n'),
  ].filter((line) => line !== '').join('\n')

  return { markdown, changedFiles, diffRefs, checkRefs, risks, reviewInstructions, readyForReview }
}

export function shouldAutoStartLocalPrReview(summary: Pick<LocalPrLoopSummary, 'readyForReview'>): boolean {
  return summary.readyForReview === true
}

export function buildLocalPrReviewPrompt(summary: Pick<LocalPrLoopSummary, 'readyForReview'>): string {
  return shouldAutoStartLocalPrReview(summary)
    ? 'Review the local PR using the implementation summary above. Open the linked diff/check evidence before approving.'
    : 'Review is waiting for stronger builder evidence. Inspect the implementation summary and ask the builder for missing diffs or checks.'
}

export function extractLocalPrReviewOutcome(reviewThread: Thread): LocalPrReviewOutcome {
  const sourceItems = reviewThread.timeline.filter((item) => item.kind === 'assistant' && item.text?.trim())
  const approvals: LocalPrReviewApproval[] = []
  const comments: LocalPrReviewComment[] = []
  const reviewerId = [reviewThread.providerId || 'provider', reviewThread.providerAccountId || 'default', reviewThread.model || 'model'].join(':')
  const latestPassId = [...sourceItems].reverse().map((item) => item.trace?.runId?.trim()).find(Boolean) ?? null
  const seenPasses = new Set<string>()
  for (const item of sourceItems) {
    const text = item.text ?? ''
    const reviewPassId = item.trace?.runId?.trim() || ''
    const passEvidence = reviewPassId ? reviewEvidenceForPass(reviewThread, reviewPassId, item.createdAt ?? Number.MAX_SAFE_INTEGER) : []
    const summaryEvidence = latestSummaryEvidence(reviewThread, item.createdAt ?? Number.MAX_SAFE_INTEGER)
    if (isExplicitApproval(text) && reviewPassId && passEvidence.length > 0 && summaryEvidence && !seenPasses.has(reviewPassId)) {
      seenPasses.add(reviewPassId)
      approvals.push({
        id: `${item.id}:approval`,
        text: firstUsefulLine(text) || 'Reviewer approved the local PR.',
        sourceItemId: item.id,
        reviewerId,
        reviewPassId,
        evidenceItemIds: [summaryEvidence, ...passEvidence],
      })
    }
    // Only the latest authenticated model run can create current blocking comments. Older
    // comments have either been addressed by a later pass or remain visible in the timeline.
    if (!reviewPassId || reviewPassId !== latestPassId) continue
    for (const line of actionableReviewLines(text)) {
      comments.push({
        id: `${item.id}:comment:${comments.length + 1}`,
        text: line,
        sourceItemId: item.id,
      })
    }
  }
  const latestItems = latestPassId ? sourceItems.filter((item) => item.trace?.runId === latestPassId) : []
  const hasBlockingLanguage = latestItems.some((item) => hasReviewBlockingLanguage(item.text ?? ''))
  const latestApproved = !!latestPassId && approvals.some((approval) => approval.reviewPassId === latestPassId)
  return {
    approved: latestApproved && comments.length === 0 && !hasBlockingLanguage,
    comments,
    approvals,
    sourceItemIds: sourceItems.map((item) => item.id),
    summary: summarizeReviewOutcome(approvals, comments, hasBlockingLanguage),
  }
}

export function shouldResumeBuilderFromReview(outcome: Pick<LocalPrReviewOutcome, 'approved' | 'comments'>): boolean {
  return !outcome.approved && outcome.comments.length > 0
}

export function localPrApprovalEvidence(outcome: Pick<LocalPrReviewOutcome, 'approved' | 'approvals' | 'sourceItemIds'>, requiredApprovals: unknown): LocalPrApprovalEvidence {
  const required = normalizeRequiredApprovals(requiredApprovals)
  const distinct = [...new Map(outcome.approvals.filter((approval) => approval.reviewPassId && approval.evidenceItemIds.length > 1).map((approval) => [approval.reviewPassId, approval])).values()]
  const recorded = distinct.length
  const missing = Math.max(0, required - recorded)
  const approved = outcome.approved && missing === 0
  return {
    approved,
    requiredApprovals: required,
    recordedApprovals: recorded,
    missingApprovals: missing,
    sourceItemIds: outcome.sourceItemIds,
    reviewPassIds: distinct.map((approval) => approval.reviewPassId),
    reviewerIds: unique(distinct.map((approval) => approval.reviewerId)),
    evidenceItemIds: unique(distinct.flatMap((approval) => approval.evidenceItemIds)),
    summary: approved
      ? `${recorded}/${required} required evidence-bound review pass${required === 1 ? '' : 'es'} recorded.`
      : `${recorded}/${required} required evidence-bound review pass${required === 1 ? '' : 'es'} recorded; ${missing} still required.`,
  }
}

function reviewEvidenceForPass(thread: Thread, runId: string, before: number): string[] {
  return unique(thread.timeline.filter((item) => (
    item.kind === 'tool' &&
    item.trace?.runId === runId &&
    (item.createdAt ?? 0) <= before &&
    item.state === 'ok' &&
    item.eventReceipt?.state === 'ok'
  )).map((item) => item.id))
}

function latestSummaryEvidence(thread: Thread, before: number): string | null {
  return [...thread.timeline].reverse().find((item) => (
    item.tool === 'local_pr_summary' && item.state === 'ok' && (item.createdAt ?? 0) <= before
  ))?.id ?? null
}

export function localPrRemotePublishGate(input: { requested?: unknown; approved?: unknown } = {}): LocalPrRemotePublishGate {
  const requested = input.requested === true
  const approved = input.approved === true
  return {
    requested,
    approved,
    allowed: requested && approved,
    reason: requested && approved
      ? 'Remote publish has an explicit approval record.'
      : 'Remote publish/push/open-PR is outside the local PR loop and requires a separate explicit approval.',
  }
}

export function buildLocalPrBuilderResumePrompt(outcome: Pick<LocalPrReviewOutcome, 'comments' | 'summary'>): string {
  const comments = outcome.comments.map((comment, index) => `${index + 1}. ${comment.text}`).join('\n')
  return [
    'Reviewer comments arrived for the local PR.',
    outcome.summary,
    '',
    'Address every comment below, rerun relevant checks, and post an updated local PR summary.',
    comments || 'No actionable comments were extracted. Ask for clarification before changing code.',
  ].join('\n').trim()
}

function step(text: string): PlanStep {
  return { id: text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40), text, done: false }
}

function clean(value: unknown): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

function cleanLong(value: unknown): string {
  return String(value ?? '').replace(/\r\n/g, '\n').trim()
}

function timelineEvidenceHref(threadId: string, itemId: string): string {
  return `vontra://thread/${encodeURIComponent(threadId)}/timeline/${encodeURIComponent(itemId)}`
}

function isCheckCommand(command: string): boolean {
  return /\b(test|typecheck|tsc|lint|vitest|jest|playwright|npm run test|pnpm test|yarn test|cargo test|go test)\b/i.test(command)
}

function commandLooksFailed(text: string): boolean {
  return /\b(exit\s+[1-9]|failed|failure|not ok|error:|errors?)\b/i.test(text)
}

function unique(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const cleanValue = clean(value)
    if (!cleanValue || seen.has(cleanValue)) continue
    seen.add(cleanValue)
    out.push(cleanValue)
  }
  return out
}

function isExplicitApproval(text: string): boolean {
  const lower = text.toLowerCase()
  return /\b(approved|approval:\s*approved|lgtm|looks good to me|ready to merge|ready for merge|ship it)\b/.test(lower)
}

function hasReviewBlockingLanguage(text: string): boolean {
  return /\b(changes requested|not approved|blocker|blocking|must fix|needs? changes?|do not merge|cannot approve)\b/i.test(text)
}

function actionableReviewLines(text: string): string[] {
  const lines = text.split(/\r?\n/)
    .map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s+/, '').trim())
    .filter(Boolean)
  return unique(lines.filter((line) => {
    if (isExplicitApproval(line)) return false
    return /\b(fix|change|blocker|missing|risk|regression|test|coverage|please|should|must|needs?|unsafe|bug|clarify)\b/i.test(line)
  })).slice(0, 12)
}

function firstUsefulLine(text: string): string {
  return text.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? ''
}

function summarizeReviewOutcome(approvals: LocalPrReviewApproval[], comments: LocalPrReviewComment[], blocked: boolean): string {
  if (approvals.length && !comments.length && !blocked) return 'Reviewer approval recorded.'
  if (comments.length) return `${comments.length} actionable reviewer comment${comments.length === 1 ? '' : 's'} extracted.`
  if (blocked) return 'Reviewer reported blockers but no actionable bullet comments were extracted.'
  return 'Review completed without explicit approval or actionable comments.'
}
