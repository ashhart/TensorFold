import type { RunRecoverySnapshot } from '../../shared/run-recovery'

const ACTIVE = new Set(['running', 'stop_requested', 'stopping'])
const FIELDS = ['runId', 'state', 'reason', 'message', 'completedTools', 'completedMutations', 'pendingOperations', 'pendingMutations', 'delivery', 'retry', 'retryReason', 'canStartNewRun', 'updatedAt']
const REASONS = ['user_stop', 'repeated_action', 'repeated_output', 'step_budget', 'time_budget', 'partial_response', 'acceptance_unknown', 'tool_uncertain', 'provider_refused', 'run_error', 'restart_interrupted']
function invalid(): never { throw new Error('Invalid or unsupported backup run recovery metadata.') }

export function validateBackupRunRecovery(input: unknown): asserts input is RunRecoverySnapshot {
  if (!input || typeof input !== 'object' || Array.isArray(input)) invalid()
  const value = input as Record<string, unknown>
  if (Object.keys(value).some(key => !FIELDS.includes(key))) invalid()
  if (typeof value.runId !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}$/u.test(value.runId)) invalid()
  for (const key of ['message', 'retryReason']) if (typeof value[key] !== 'string' || (value[key] as string).length > 4_000) invalid()
  for (const key of ['completedTools', 'completedMutations', 'pendingOperations', 'pendingMutations', 'updatedAt']) if (!Number.isSafeInteger(value[key]) || Number(value[key]) < 0) invalid()
  if (Number(value.completedMutations) > Number(value.completedTools) || Number(value.pendingMutations) > Number(value.pendingOperations)) invalid()
  if (!['running', 'stop_requested', 'stopping', 'stopped', 'paused', 'needs_review', 'failed', 'completed'].includes(String(value.state))
    || !['not_sent', 'refused', 'unknown', 'accepted', 'partial', 'complete'].includes(String(value.delivery))
    || !['safe', 'requires_review', 'unavailable'].includes(String(value.retry)) || typeof value.canStartNewRun !== 'boolean'
    || value.reason !== undefined && !REASONS.includes(String(value.reason))) invalid()
  if (value.state === 'needs_review' && value.retry !== 'requires_review') invalid()
  if (!ACTIVE.has(String(value.state)) && ['unknown', 'accepted', 'partial'].includes(String(value.delivery)) && value.retry !== 'requires_review') invalid()
  if (value.retry === 'safe' && ['tool_uncertain', 'partial_response', 'acceptance_unknown', 'restart_interrupted'].includes(String(value.reason))) invalid()
}

/** A backup carries historical uncertainty, never a live process/controller or a reviewed decision. */
export function portableRunRecovery(input: unknown): RunRecoverySnapshot {
  validateBackupRunRecovery(input)
  const interrupted = ACTIVE.has(input.state) || input.pendingOperations > 0 || !input.canStartNewRun
  const uncertain = interrupted || input.state === 'needs_review' || input.retry === 'requires_review' || ['unknown', 'accepted', 'partial'].includes(input.delivery)
  if (interrupted) return { ...input, state: 'needs_review', reason: 'restart_interrupted', message: 'This history was restored without its original process; verify any external action outcomes before continuing.',
    pendingOperations: 0, pendingMutations: 0, delivery: 'unknown', retry: 'requires_review', retryReason: 'Original operations and controller ownership were not imported.', canStartNewRun: true }
  return { ...input, ...(uncertain ? { state: 'needs_review' as const, retry: 'requires_review' as const } : {}), pendingOperations: 0, pendingMutations: 0, canStartNewRun: true }
}
