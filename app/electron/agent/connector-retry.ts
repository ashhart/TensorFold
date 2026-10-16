// Transparent retry/backoff for connector HTTP calls — the one piece Hermes's connector
// layer has that TensorFold's lacked. Adapted to TensorFold's idiom (a drop-in `fetch` replacement,
// not a Python base-class method) so a connector keeps its existing `const res = await
// fetch(...); if (!res.ok) ...` shape and just swaps `fetch` → `fetchWithRetry`. By the time
// the caller sees `!res.ok`, the transient retries (429 / 5xx / dropped socket) have already
// happened with jittered backoff and Retry-After awareness.
//
// Deliberately reuses the SAME transient/auth classification + jittered backoff the model
// stream path uses (api-error-classifier.ts) instead of a second, drifting copy — a 429 from
// Slack and a 429 from the LLM provider are "wait and retry" for the exact same reasons.

import { classifyApiError, jitteredBackoff } from './api-error-classifier.ts'

export interface RetryInfo {
  // 1-based attempt that just FAILED and is being retried.
  attempt: number
  // How long we're about to wait before the next attempt (ms).
  delayMs: number
  // HTTP status when the failure was a response (absent for a thrown network error).
  status?: number
  // Short human reason (for logging/telemetry).
  reason: string
}

export interface ConnectorRetryOptions {
  // Total attempts INCLUDING the first. 1 disables retry. Default 3.
  maxAttempts?: number
  // Backoff base / cap (ms) for the jittered exponential curve. Defaults match the model path.
  baseDelayMs?: number
  maxDelayMs?: number
  // If the server's Retry-After asks for longer than this, give up rather than block the run
  // (hand the failing response back so the caller surfaces it). Default 30s.
  retryAfterCapMs?: number
  // Cancel in-flight retries when the run is aborted.
  signal?: AbortSignal
  // Injectable jitter source (tests) — forwarded to jitteredBackoff.
  random?: () => number
  // Injectable sleep (tests) so no real time passes.
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>
  // Injectable fetch (tests / non-global environments).
  fetchImpl?: typeof fetch
  // Observe each retry (logging/telemetry); never throws back into the loop.
  onRetry?: (info: RetryInfo) => void
}

function abortError(): Error {
  return Object.assign(new Error('Aborted'), { name: 'AbortError' })
}

function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(abortError()); return }
    const timer = setTimeout(resolve, Math.max(0, ms))
    signal?.addEventListener?.('abort', () => { clearTimeout(timer); reject(abortError()) }, { once: true })
  })
}

/**
 * Decide whether an HTTP status warrants a transparent retry. Delegates to the shared
 * classifier so the retryable set stays identical to the model stream path (429 + the
 * 5xx family + retryable 4xx like 408/409/425). Auth (401/403), not-found, 413, and any
 * 2xx are NOT retried.
 */
export function shouldRetryStatus(status: number): boolean {
  if (!Number.isFinite(status)) return false
  return classifyApiError({ status }).retryable
}

/**
 * Parse a `Retry-After` header into milliseconds. Accepts delta-seconds (integer or
 * fractional, e.g. "2" or "0.5" as some gateways/Discord send) and an HTTP-date, which is
 * measured against `now`. A date in the past clamps to 0. Returns null for a missing or
 * unparseable value (caller falls back to the backoff curve). Pure.
 */
export function parseRetryAfter(headerValue: string | null | undefined, now: number): number | null {
  if (headerValue == null) return null
  const v = String(headerValue).trim()
  if (!v) return null
  if (/^\d+(\.\d+)?$/.test(v)) return Math.max(0, Math.round(Number(v) * 1000))
  const dateMs = Date.parse(v)
  if (Number.isFinite(dateMs)) return Math.max(0, dateMs - now)
  return null
}

// Release a discarded response's body so undici doesn't keep the socket open while we wait to
// retry (an unconsumed body is a real resource leak). Best-effort: never throws.
async function cancelBody(res: any): Promise<void> {
  try { await res?.body?.cancel?.() } catch { /* best effort */ }
}

/**
 * `fetch` with transparent transient-error retry. Returns the final `Response` (success, a
 * non-retryable status, or the last failing response after exhausting attempts) so existing
 * `if (!res.ok)` connector code is unchanged. Rethrows a non-transient error immediately and a
 * transient network error only after attempts are exhausted.
 */
export async function fetchWithRetry(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1] = {},
  opts: ConnectorRetryOptions = {},
): Promise<Response> {
  const maxAttempts = Math.max(1, Math.floor(opts.maxAttempts ?? 3))
  const baseDelayMs = opts.baseDelayMs ?? 500
  const maxDelayMs = opts.maxDelayMs ?? 15_000
  const retryAfterCapMs = opts.retryAfterCapMs ?? 30_000
  const fetchImpl = opts.fetchImpl ?? fetch
  const sleep = opts.sleep ?? defaultSleep
  const backoffOpts = opts.random ? { random: opts.random } : {}
  const signal = opts.signal ?? (init as any)?.signal ?? undefined
  // Thread the abort signal into the underlying fetch without mutating the caller's init.
  const fetchInit = signal ? { ...(init || {}), signal } : init

  let lastError: unknown
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    if (signal?.aborted) throw abortError()

    let res: Response
    try {
      res = await fetchImpl(input, fetchInit)
    } catch (err) {
      lastError = err
      const c = classifyApiError(err)
      if (!c.retryable || attempt >= maxAttempts || signal?.aborted) throw err
      const delayMs = jitteredBackoff(attempt, baseDelayMs, maxDelayMs, backoffOpts)
      opts.onRetry?.({ attempt, delayMs, reason: c.reason })
      await sleep(delayMs, signal)
      continue
    }

    // A response we got back. Non-retryable status (incl. success) → return as-is for the
    // caller's own !res.ok handling. Last attempt → also return the failing response.
    if (!shouldRetryStatus(res.status) || attempt >= maxAttempts) return res

    // Retryable status with attempts left. Honor Retry-After, but refuse to block the run for
    // an absurd duration — hand the failing response back instead.
    const retryAfterMs = parseRetryAfter(res.headers?.get?.('retry-after'), Date.now())
    if (retryAfterMs != null && retryAfterMs > retryAfterCapMs) return res

    await cancelBody(res)
    const delayMs = retryAfterMs != null ? retryAfterMs : jitteredBackoff(attempt, baseDelayMs, maxDelayMs, backoffOpts)
    opts.onRetry?.({ attempt, delayMs, status: res.status, reason: `HTTP ${res.status}` })
    await sleep(delayMs, signal)
  }

  // Unreachable for maxAttempts >= 1 (the loop returns or throws inside), but satisfies the
  // type checker and guards against a future off-by-one.
  if (lastError) throw lastError
  throw new Error('fetchWithRetry: exhausted attempts')
}
