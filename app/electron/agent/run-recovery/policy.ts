import { extractErrorStatus } from '../api-error-classifier'
import type { RunDeliveryState, RunPauseReason } from '../../../shared/run-recovery'

/** Only connection-establishment errors prove that a request was not delivered. */
export function failedRequestDelivery(error: unknown, receivedOutput: boolean): RunDeliveryState {
  if (receivedOutput) return 'partial'
  const record = error && typeof error === 'object' ? error as Record<string, any> : {}
  if (record.delivery === 'not_sent' || record.delivery === 'refused') return record.delivery
  const code = String(record.code ?? record.cause?.code ?? '')
  if (['ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'ENETUNREACH', 'EHOSTUNREACH', 'UND_ERR_CONNECT_TIMEOUT'].includes(code)) return 'not_sent'
  const status = extractErrorStatus(error)
  // HTTP rejection from a completed error response, before any output. Gateway timeouts,
  // conflicts and generic server errors do not prove that an upstream generation was refused.
  if (status != null && [400, 401, 403, 404, 413, 422, 425, 429, 503, 529].includes(status)) return 'refused'
  return 'unknown'
}

export function budgetPauseReason(reason: string): RunPauseReason { return reason === 'time budget' ? 'time_budget' : reason === 'repeated action limit' ? 'repeated_action' : 'step_budget' }
