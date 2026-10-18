export type RunRecoveryState = 'running' | 'stop_requested' | 'stopping' | 'stopped' | 'paused' | 'needs_review' | 'failed' | 'completed'
export type RunDeliveryState = 'not_sent' | 'refused' | 'unknown' | 'accepted' | 'partial' | 'complete'
export type RunPauseReason = 'user_stop' | 'repeated_action' | 'repeated_output' | 'step_budget' | 'time_budget' | 'partial_response' | 'acceptance_unknown' | 'tool_uncertain' | 'provider_refused' | 'run_error' | 'restart_interrupted'
export interface RunRecoverySnapshot {
  runId: string
  state: RunRecoveryState
  reason?: RunPauseReason
  message: string
  completedTools: number
  completedMutations: number
  pendingOperations: number
  pendingMutations: number
  delivery: RunDeliveryState
  retry: 'safe' | 'requires_review' | 'unavailable'
  retryReason: string
  /** Physical drain barrier only; retry policy still applies after work drains. */
  canStartNewRun: boolean
  updatedAt: number
}
export interface RunRetryEvidence { delivery: RunDeliveryState; delegated?: boolean; visibleOutput?: boolean; uncertainEffects?: boolean }
export interface RunRetryVerdict { allowed: boolean; reason: string }

/** Acceptance is evidence, not a guess from a generic transient-error label. */
export function runRetryVerdict(evidence: RunRetryEvidence): RunRetryVerdict {
  if (evidence.delegated) return { allowed: false, reason: 'The delegated engine may have completed actions; inspect its work before starting another request.' }
  if (evidence.uncertainEffects) return { allowed: false, reason: 'An action may have completed; reconcile its outcome before retrying.' }
  if (evidence.visibleOutput || evidence.delivery === 'partial') return { allowed: false, reason: 'The provider returned partial output; another request may be billable and requires an explicit choice.' }
  if (evidence.delivery === 'complete') return { allowed: false, reason: 'The request already completed; continue with an explicit new request instead of replaying it.' }
  if (evidence.delivery === 'unknown' || evidence.delivery === 'accepted') return { allowed: false, reason: 'Request acceptance or completion is uncertain; check the provider before sending another request.' }
  return { allowed: true, reason: evidence.delivery === 'not_sent' ? 'The request was not dispatched.' : 'The provider explicitly refused this request before output.' }
}

/** A saved active run cannot be labelled stopped after a process restart without reconciliation. */
export function interruptedRunRecovery(snapshot: RunRecoverySnapshot, now = Date.now()): RunRecoverySnapshot {
  if (!['running', 'stop_requested', 'stopping'].includes(snapshot.state)) return snapshot
  return { ...snapshot, state: 'needs_review', reason: 'restart_interrupted', message: 'The app closed before the run drained; verify any external actions before continuing.', pendingOperations: 0, pendingMutations: 0, delivery: 'unknown', retry: 'requires_review', retryReason: 'The prior process could not confirm completion.', canStartNewRun: true, updatedAt: now }
}
