import type { Thread } from './types'
import type { RunOutcomeReviewInput } from '../../shared/run-recovery-api'

export function assertRecoveryAllowsNewRun(thread: Thread): void {
  if (thread.runRecovery && !thread.runRecovery.canStartNewRun) throw new Error('Wait for started operations to settle before starting another run.')
  if (thread.runRecovery?.retry === 'requires_review' && thread.runRecoveryReviewedId !== thread.runRecovery.runId) throw new Error('Review the previous run outcome before sending another request; a tool or provider may already have accepted it.')
}
export function reviewedRunOutcome(thread: Thread, input: RunOutcomeReviewInput, active: boolean): Thread {
  if (!input || input.threadId !== thread.id || !thread.runRecovery || input.runId !== thread.runRecovery.runId) throw new Error('The run changed; review the current outcome.')
  if (active || !thread.runRecovery.canStartNewRun) throw new Error('Started work is still draining; wait before reviewing its final outcome.')
  if (thread.runRecovery.retry !== 'requires_review') throw new Error('This run does not need an uncertain-outcome review.')
  return { ...thread, runRecoveryReviewedId: input.runId }
}
