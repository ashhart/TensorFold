// Bound retries by delivery evidence, transport side effects, and compaction progress.

import { classifyApiError, jitteredBackoff, supportedEffort } from './api-error-classifier'
import { runRetryVerdict, type RunRetryEvidence } from '../../shared/run-recovery'

export const STREAM_MAX_RETRIES = 3

export const STREAM_RETRY_BASE_MS = 600

export const STREAM_RETRY_MAX_MS = 8_000

const STREAM_MAX_COMPACT_RETRIES = 2

export const MAX_STEER_RESTARTS = 2

// Remember the server's accepted effort by provider and model across turns.
export const serverEfforts = new Map<string, string>()

export type StreamRetryAction =
  | { kind: 'retry'; delayMs: number; reason: string }
  | { kind: 'compact' }
  | { kind: 'effort'; effort: string }
  | { kind: 'fail' }

const DELEGATED_TRANSPORTS = new Set(['codex-cli', 'claude-cli', 'cursor-cli', 'gemini-cli', 'pi-cli', 'opencode', 'kilo'])

export function isDelegatedTransport(transport?: string | null): boolean {
  return !!transport && DELEGATED_TRANSPORTS.has(transport)
}

export function streamRetryDecision(
  error: unknown,
  attempt: number,
  opts: { compactRetries: number; canCompact: boolean; delegated?: boolean; maxRetries?: number; maxCompactRetries?: number; random?: () => number; acceptance?: RunRetryEvidence; effortRetried?: boolean },
): StreamRetryAction {
  // Delegated engines may already have performed effects; uncertain delivery and visible output also forbid replay.
  if (opts.delegated || opts.acceptance && !runRetryVerdict(opts.acceptance).allowed) return { kind: 'fail' }
  const effort = opts.effortRetried ? null : supportedEffort(error)
  if (effort) return { kind: 'effort', effort }
  const maxRetries = opts.maxRetries ?? STREAM_MAX_RETRIES
  const maxCompact = opts.maxCompactRetries ?? STREAM_MAX_COMPACT_RETRIES
  const c = classifyApiError(error)

  if (c.shouldCompress && opts.canCompact && opts.compactRetries < maxCompact) {
    return { kind: 'compact' }
  }
  if (c.retryable && attempt <= maxRetries) {
    return { kind: 'retry', delayMs: jitteredBackoff(attempt, STREAM_RETRY_BASE_MS, STREAM_RETRY_MAX_MS, { random: opts.random }), reason: c.reason }
  }
  return { kind: 'fail' }
}

// Resolve early on abort; callers must recheck the signal before making another request.
export function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  if (ms <= 0 || signal.aborted) return Promise.resolve()
  return new Promise<void>((resolve) => {
    const timer = setTimeout(done, ms)
    function done() {
      clearTimeout(timer)
      signal.removeEventListener('abort', done)
      resolve()
    }
    signal.addEventListener('abort', done, { once: true })
  })
}
