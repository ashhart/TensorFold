// Structured classification of a provider / stream error, plus a jittered
// exponential backoff helper. Both are PURE (no side effects, no I/O) so the
// retry policy that wraps the model stream in loop.ts is unit-testable in
// isolation and deterministic.
//
// Why this lives next to (not inside) model-router.ts: the router classifies an
// error to decide *which account/route* to fail over to (rate-limit / quota
// exhaustion → cooldown a route). THIS module answers a different question for a
// SINGLE stream call — "should I retry the same call, compact first, or fail?".
// A flaky provider that drops the socket mid-stream (ECONNRESET) or 503s for a
// few seconds ("the run stops after 30s") should transparently retry with
// backoff; a 401 or a content-policy refusal must fail immediately. Keeping the
// two concerns separate avoids overloading the router's exhaustion model with
// transient-retry semantics.

export type ApiErrorCategory =
  | 'transient' // retry the same call after a backoff (429 / 5xx / dropped socket / "overloaded")
  | 'auth' // bad/expired key or forbidden — retrying is pointless, surface it
  | 'context-overflow' // request too large for the model — compact, then retry
  | 'content-policy' // provider refused on policy grounds — abort, do not retry
  | 'unknown' // unrecognized — treat conservatively (no retry)

export interface ApiErrorClassification {
  category: ApiErrorCategory
  // Whether the SAME stream call may be retried (after a backoff). Only true for
  // 'transient'. 'context-overflow' is handled by compaction, not a blind retry,
  // so it is NOT retryable on its own.
  retryable: boolean
  // Whether the caller should compact the transcript before trying again. Only
  // true for 'context-overflow'.
  shouldCompress: boolean
  // The HTTP status, when one could be extracted (handy for logging / notices).
  status?: number
  // A short, human-readable reason for the classification (surfaced in notices).
  reason: string
}

// --- error-shape extraction (mirrors the tolerant readers in model-router.ts so
// the same Error & { status; headers } objects that openai.ts throws classify
// correctly here too, without importing the router) -------------------------

function errorRecord(error: unknown): Record<string, unknown> {
  return typeof error === 'object' && error ? (error as Record<string, unknown>) : {}
}

export function extractErrorStatus(error: unknown): number | undefined {
  const record = errorRecord(error)
  const direct = record.status ?? record.statusCode ?? record.code
  if (typeof direct === 'number' && Number.isFinite(direct)) return direct
  if (typeof direct === 'string' && /^\d{3}$/.test(direct)) return Number(direct)
  // Fall back to an `HTTP 503 …` prefix in the message (openai.ts formats errors that way).
  const message = String(record.message ?? error ?? '')
  const http = message.match(/\bHTTP\s+(\d{3})\b/i)
  return http ? Number(http[1]) : undefined
}

export function extractErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  const record = errorRecord(error)
  if (record.message != null || record.error != null || record.type != null) {
    return String(record.message ?? record.error ?? record.type)
  }
  return String(error)
}

// A node-level network error code (ETIMEDOUT, ECONNRESET, …) may sit on `.code`
// or be embedded in the message string; surface whichever is present.
function networkErrorCode(error: unknown): string | null {
  const record = errorRecord(error)
  const code = typeof record.code === 'string' ? record.code : null
  if (code) return code.toUpperCase()
  // `cause` is where fetch() stashes the underlying system error.
  const cause = record.cause && typeof record.cause === 'object' ? (record.cause as Record<string, unknown>) : null
  const causeCode = cause && typeof cause.code === 'string' ? cause.code : null
  return causeCode ? causeCode.toUpperCase() : null
}

// --- pattern sets ------------------------------------------------------------

// Transient HTTP statuses: rate-limit + the standard "try again" server family.
// 408 (request timeout) and 409 (conflict) are included — both are commonly
// retryable on idempotent reads against flaky gateways.
const TRANSIENT_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504, 529])

// Node / undici socket-level failures that mean "the connection died, try again".
const TRANSIENT_NET_CODES = new Set([
  'ETIMEDOUT', 'ECONNRESET', 'ECONNREFUSED', 'ECONNABORTED', 'EPIPE',
  'EAI_AGAIN', 'ENOTFOUND', 'ENETUNREACH', 'EHOSTUNREACH', 'UND_ERR_SOCKET',
  'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT',
])

const AUTH_STATUS = new Set([401, 403])
const CONTEXT_STATUS = new Set([413]) // payload too large

const TRANSIENT_MSG =
  /\b(overloaded|temporarily unavailable|service unavailable|server error|bad gateway|gateway timeout|timed?\s*out|timeout|connection (?:reset|closed|aborted|error)|socket hang ?up|stream (?:closed|interrupted|error)|network error|fetch failed|try again|rate[\s_-]?limit|too many requests|throttl(?:e|ed|ing)|capacity)\b/i

const AUTH_MSG =
  /\b(unauthorized|forbidden|invalid[_\s-]?api[_\s-]?key|invalid[_\s-]?token|authentication|api key|expired (?:key|token|credential)|permission denied|access denied)\b/i

const CONTEXT_MSG =
  /\b(context[_\s-]?length[_\s-]?exceeded|context window|maximum context|context length|too many tokens|token limit|reduce the length|input too large|payload too large|request too large|prompt is too long|maximum (?:input|prompt) length)\b/i

const CONTENT_POLICY_MSG =
  /\b(content[_\s-]?policy|content[_\s-]?filter|safety (?:policy|system|filter)|moderation|flagged|responsible ai|prohibited content|policy violation|cannot assist with that request)\b/i

// --- classifier --------------------------------------------------------------

function classification(
  category: ApiErrorCategory,
  reason: string,
  status?: number,
): ApiErrorClassification {
  return {
    category,
    retryable: category === 'transient',
    shouldCompress: category === 'context-overflow',
    status,
    reason,
  }
}

/**
 * Classify a provider / stream error into one actionable category.
 *
 * Precedence matters: context-overflow and content-policy are checked BEFORE the
 * generic transient/auth matchers, because some providers report a 400 with a
 * context-length body (which should compact, not retry) and a refusal can ride
 * on a 400/403 too (which should abort, never retry). Auth is checked before the
 * transient message net so a 401 whose body happens to contain "try again" is
 * still treated as auth.
 */
export function classifyApiError(error: unknown): ApiErrorClassification {
  const status = extractErrorStatus(error)
  const message = extractErrorMessage(error)
  const netCode = networkErrorCode(error)

  // 1. Context overflow → compact then retry (highest precedence: a 400/413 with
  //    a context-length body must not be mistaken for a hard client error).
  if ((status != null && CONTEXT_STATUS.has(status)) || CONTEXT_MSG.test(message)) {
    return classification('context-overflow', 'Request exceeded the model context window — compacting before retry.', status)
  }

  // 2. Content policy → abort (never retry, never compact).
  if (CONTENT_POLICY_MSG.test(message)) {
    return classification('content-policy', 'Provider refused the request on content-policy grounds.', status)
  }

  // 3. Auth → fail fast (retrying a bad key just burns the run).
  if ((status != null && AUTH_STATUS.has(status)) || AUTH_MSG.test(message)) {
    return classification('auth', 'Authentication or permission failure — check the provider key/credentials.', status)
  }

  // 4. Transient → retry with backoff. Status family OR a dropped-socket code OR
  //    a recognizable "overloaded / unavailable / timeout" message.
  if (status != null && TRANSIENT_STATUS.has(status)) {
    return classification('transient', `Provider returned a transient HTTP ${status} — retrying with backoff.`, status)
  }
  if (netCode && TRANSIENT_NET_CODES.has(netCode)) {
    return classification('transient', `Network connection failed (${netCode}) — retrying with backoff.`, status)
  }
  if (TRANSIENT_MSG.test(message)) {
    return classification('transient', 'Provider reported a transient/overloaded condition — retrying with backoff.', status)
  }

  // 5. Anything else → unknown, treated conservatively (no automatic retry).
  return classification('unknown', 'Unrecognized provider error — not retrying automatically.', status)
}

// --- jittered exponential backoff -------------------------------------------

export interface BackoffOptions {
  // Inject a [0,1) source for deterministic tests; defaults to Math.random.
  random?: () => number
}

/**
 * Jittered exponential backoff for retry attempt `attempt` (1-based).
 *
 * Base delay grows as min(base * 2^(attempt-1), max); a random jitter of up to
 * +50% of that capped delay is then added ("equal jitter"-ish) so a fleet of
 * clients retrying after the same outage don't thunder back in lockstep. The
 * jitter is added AFTER the cap, so the returned value can sit between `max` and
 * `1.5 * max` — that headroom is intentional (it spreads the herd past the cap).
 *
 * Pure: with a fixed `random`, the output is fully determined by the inputs, so
 * the bounds and monotonic growth of the pre-jitter term are unit-tested.
 */
export function jitteredBackoff(attempt: number, base = 500, max = 15_000, opts: BackoffOptions = {}): number {
  const random = opts.random ?? Math.random
  const safeAttempt = Number.isFinite(attempt) && attempt > 0 ? Math.floor(attempt) : 1
  const safeBase = Number.isFinite(base) && base > 0 ? base : 0
  const safeMax = Number.isFinite(max) && max > 0 ? max : safeBase
  // 2^(attempt-1) can overflow for large attempts; cap the exponent so the
  // multiplication stays finite, then clamp to max anyway.
  const exponent = Math.min(safeAttempt - 1, 31)
  const raw = safeBase * 2 ** exponent
  const capped = Math.min(raw, safeMax)
  const jitter = Math.max(0, Math.min(1, random())) * 0.5 * capped
  return Math.round(capped + jitter)
}
