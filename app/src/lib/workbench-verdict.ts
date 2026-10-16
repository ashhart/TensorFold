// Workbench done-gate (VST-017): "done" for a coding surface MEANS tests passed + diff
// reviewed — an evidence verdict, not a vibe. Tests-passed comes from the run receipt
// (fresh successful checks AFTER the last change → 'verified'); diff-reviewed comes from an
// explicit review signal: the PR-loop review approval, or the user's "mark reviewed" action,
// which only counts while FRESHER than the run it reviews (a new run invalidates old marks).
// Pure derivation over the agentic run contract; ThreadPanel renders it.

import type { AgenticRunReceipt, AgenticSurface } from './agentic-run-contract'
import type { Job } from './types'

export interface WorkbenchDoneVerdict {
  // The gate only applies to code-mode surfaces; chat/assistant runs have no diff to review.
  applicable: boolean
  done: boolean
  testsPassed: boolean
  diffReviewed: boolean
  reviewRequired: boolean
  // Precisely what is missing, in the order to fix it. Empty when done (or not applicable).
  missing: string[]
  detail: string
}

const CODE_SURFACES = new Set<AgenticSurface>(['code', 'ticket', 'room', 'automation', 'remote'])

export function workbenchDoneVerdict(input: {
  surface: AgenticSurface
  receipt: Pick<AgenticRunReceipt, 'verdict' | 'diffCount' | 'changedFiles'>
  // When the user last marked the diffs reviewed (Thread.diffReviewedAt).
  diffReviewedAt?: number | null
  // When the run this receipt describes finished — the freshness bar for the mark.
  receiptFinishedAt?: number | null
  // PR-loop review outcome, when this thread is a local-PR implementation thread.
  reviewApproved?: boolean
  // Local PR implementation threads cannot substitute a manual mark for the configured
  // configured count of distinct evidence-bound review passes.
  reviewRequired?: boolean
}): WorkbenchDoneVerdict {
  if (!CODE_SURFACES.has(input.surface)) {
    return { applicable: false, done: false, testsPassed: false, diffReviewed: false, reviewRequired: false, missing: [], detail: 'Not a coding surface — the done gate does not apply.' }
  }
  const receipt = input.receipt
  const hasChanges = receipt.diffCount > 0 || receipt.changedFiles.length > 0

  // A run that ended badly is never done, changes or not. (The renderer receipt folds the
  // host's 'cancelled' verdict into 'blocked' — see receiptFromSnapshot.)
  if (receipt.verdict === 'failed' || receipt.verdict === 'blocked') {
    return {
      applicable: true, done: false, testsPassed: false, diffReviewed: false, reviewRequired: input.reviewRequired === true,
      missing: [`a clean run (last run ${receipt.verdict})`],
      detail: `Not done: the last run ${receipt.verdict === 'blocked' ? 'is blocked (approval pending or cancelled)' : 'failed'}.`,
    }
  }

  // Nothing changed → nothing to verify or review. Done, and the detail says WHY (never
  // silently healthy: "no changes" is stated, not implied).
  if (!hasChanges && input.reviewRequired !== true) {
    return { applicable: true, done: true, testsPassed: true, diffReviewed: true, reviewRequired: false, missing: [], detail: 'Done: no code changes recorded — nothing to verify or review.' }
  }

  const testsPassed = !hasChanges || receipt.verdict === 'verified'
  const markFresh = input.diffReviewedAt != null && input.diffReviewedAt >= (input.receiptFinishedAt ?? 0)
  const reviewRequired = input.reviewRequired === true
  const diffReviewed = input.reviewApproved === true || (!reviewRequired && markFresh)

  const missing: string[] = []
  if (!testsPassed) missing.push('fresh passing checks after the last change (run tests/verify)')
  if (!diffReviewed) missing.push(reviewRequired ? 'required PR review approval' : 'diff review (approve in the review loop, or mark the diffs reviewed)')

  return {
    applicable: true,
    done: missing.length === 0,
    testsPassed,
    diffReviewed,
    reviewRequired,
    missing,
    detail: missing.length === 0
      ? `Done: checks passed and the diff was reviewed${input.reviewApproved ? ' (review approved)' : ''}.`
      : `Not done — missing: ${missing.join('; ')}.`,
  }
}

/** True when a "mark reviewed" action would change the verdict (there IS something to mark). */
export function canMarkDiffsReviewed(verdict: WorkbenchDoneVerdict): boolean {
  return verdict.applicable && !verdict.reviewRequired && !verdict.diffReviewed && !verdict.done
}

export interface WorkbenchPrReviewState {
  required: boolean
  approved: boolean
  requiredApprovals: number
  recordedApprovals: number
}

/** Resolve the durable local-PR reviewer evidence attached to an implementation thread. */
export function workbenchPrReviewState(threadId: string, jobs: Array<Pick<Job, 'kind' | 'metadata' | 'updatedAt'>>): WorkbenchPrReviewState {
  const job = [...jobs]
    .filter((candidate) => candidate.kind === 'pr_loop' && stringMeta(candidate.metadata, 'implementationThreadId') === threadId)
    .sort((a, b) => b.updatedAt - a.updatedAt)[0]
  if (!job) return { required: false, approved: false, requiredApprovals: 0, recordedApprovals: 0 }

  const evidence = objectMeta(job.metadata, 'localPrApprovalEvidence')
  const requiredApprovals = Math.max(1, finiteCount(evidence?.requiredApprovals) ?? finiteCount(job.metadata?.requiredApprovals) ?? 1)
  const recordedApprovals = finiteCount(evidence?.recordedApprovals) ?? 0
  const missingApprovals = finiteCount(evidence?.missingApprovals) ?? Math.max(0, requiredApprovals - recordedApprovals)
  return {
    required: true,
    approved: evidence?.approved === true && recordedApprovals >= requiredApprovals && missingApprovals === 0,
    requiredApprovals,
    recordedApprovals,
  }
}

function stringMeta(metadata: Record<string, unknown> | undefined, key: string): string | null {
  const value = metadata?.[key]
  return typeof value === 'string' && value ? value : null
}

function objectMeta(metadata: Record<string, unknown> | undefined, key: string): Record<string, any> | null {
  const value = metadata?.[key]
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, any> : null
}

function finiteCount(value: unknown): number | null {
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? Math.floor(number) : null
}
