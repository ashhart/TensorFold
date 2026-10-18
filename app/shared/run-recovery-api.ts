import type { RunRecoverySnapshot } from './run-recovery'
export interface RunOutcomeReviewInput { threadId: string; runId: string }
export interface RunRecoveryApi { reviewRunOutcome(input: RunOutcomeReviewInput): Promise<{ recovery: RunRecoverySnapshot; reviewedRunId: string }> }
