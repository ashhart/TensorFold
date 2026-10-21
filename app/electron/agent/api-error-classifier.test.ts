// Run: jiti electron/agent/api-error-classifier.test.ts
import assert from 'node:assert/strict'
import { classifyApiError, jitteredBackoff, extractErrorStatus, supportedEffort } from './api-error-classifier.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('api-error-classifier tests:')

// --- transient (retryable) ---------------------------------------------------
{
  for (const status of [408, 409, 425, 429, 500, 502, 503, 504, 529]) {
    const c = classifyApiError(Object.assign(new Error(`HTTP ${status} Something`), { status }))
    assert.equal(c.category, 'transient', `HTTP ${status} should be transient`)
    assert.equal(c.retryable, true, `HTTP ${status} should be retryable`)
    assert.equal(c.shouldCompress, false)
    assert.equal(c.status, status)
  }
  ok('rate-limit + 5xx + retryable-4xx statuses classify as transient/retryable')
}

{
  // Status is parsed out of the `HTTP <code>` message even with no .status field.
  const c = classifyApiError(new Error('HTTP 503 Service Unavailable — upstream connect error'))
  assert.equal(c.category, 'transient')
  assert.equal(c.retryable, true)
  assert.equal(c.status, 503)
  ok('status is recovered from an "HTTP 503 …" message when no status field is set')
}

{
  for (const code of ['ETIMEDOUT', 'ECONNRESET', 'EPIPE', 'EAI_AGAIN', 'UND_ERR_SOCKET']) {
    const c = classifyApiError(Object.assign(new Error('socket failure'), { code }))
    assert.equal(c.category, 'transient', `${code} should be transient`)
    assert.equal(c.retryable, true)
  }
  // fetch() stashes the system error on .cause
  const viaCause = classifyApiError(Object.assign(new Error('fetch failed'), { cause: { code: 'ECONNRESET' } }))
  assert.equal(viaCause.category, 'transient')
  ok('dropped-socket network codes (incl. via .cause) classify as transient/retryable')
}

{
  for (const msg of ['Model is overloaded, please try again', 'The server is temporarily unavailable', 'request timed out', 'socket hang up', 'upstream connection reset']) {
    const c = classifyApiError(new Error(msg))
    assert.equal(c.category, 'transient', `"${msg}" should be transient`)
    assert.equal(c.retryable, true)
  }
  ok('overloaded / unavailable / timeout / connection-reset messages classify as transient')
}

// --- auth (fail fast) --------------------------------------------------------
{
  for (const status of [401, 403]) {
    const c = classifyApiError(Object.assign(new Error(`HTTP ${status}`), { status }))
    assert.equal(c.category, 'auth')
    assert.equal(c.retryable, false, `HTTP ${status} must NOT be retryable`)
    assert.equal(c.shouldCompress, false)
  }
  const byMsg = classifyApiError(new Error('Incorrect API key provided: invalid_api_key'))
  assert.equal(byMsg.category, 'auth')
  assert.equal(byMsg.retryable, false)
  ok('401/403 + invalid-api-key classify as auth (never retried)')
}

{
  // A 401 whose body happens to contain "try again" is still auth, not transient.
  const c = classifyApiError(Object.assign(new Error('HTTP 401 Unauthorized — try again later'), { status: 401 }))
  assert.equal(c.category, 'auth')
  assert.equal(c.retryable, false)
  ok('auth wins over a transient-looking message when the status is 401')
}

// --- context overflow (compact then retry) -----------------------------------
{
  const msgs = [
    'This model\'s maximum context length is 128000 tokens, however you requested 140000',
    'context_length_exceeded',
    'prompt is too long for the context window',
    'reduce the length of the messages',
  ]
  for (const m of msgs) {
    const c = classifyApiError(new Error(m))
    assert.equal(c.category, 'context-overflow', `"${m}" should be context-overflow`)
    assert.equal(c.shouldCompress, true, `"${m}" should request compression`)
    assert.equal(c.retryable, false, 'context overflow is handled by compaction, not a blind retry')
  }
  // 413 payload-too-large is a context overflow even if the body is terse.
  const c413 = classifyApiError(Object.assign(new Error('HTTP 413 Payload Too Large'), { status: 413 }))
  assert.equal(c413.category, 'context-overflow')
  assert.equal(c413.shouldCompress, true)
  ok('context-length / payload-too-large classify as context-overflow (compress, not retry)')
}

{
  // Precedence: a 400 carrying a context-length body compacts — it is NOT a hard
  // unknown client error, and NOT a blind transient retry.
  const c = classifyApiError(Object.assign(new Error('HTTP 400 — context_length_exceeded: too many tokens'), { status: 400 }))
  assert.equal(c.category, 'context-overflow')
  assert.equal(c.shouldCompress, true)
  assert.equal(c.retryable, false)
  ok('a 400 with a context-length body classifies as context-overflow, not unknown/transient')
}

// --- content policy (abort) --------------------------------------------------
{
  for (const m of ['Your request was rejected by our content policy', 'flagged by the safety system', 'This violates our moderation guidelines']) {
    const c = classifyApiError(new Error(m))
    assert.equal(c.category, 'content-policy', `"${m}" should be content-policy`)
    assert.equal(c.retryable, false)
    assert.equal(c.shouldCompress, false)
  }
  ok('content-policy / safety / moderation messages classify as content-policy (abort)')
}

// --- unknown (conservative: no retry) ----------------------------------------
{
  const c = classifyApiError(new Error('Some entirely novel failure mode 1234'))
  assert.equal(c.category, 'unknown')
  assert.equal(c.retryable, false)
  assert.equal(c.shouldCompress, false)
  // A 400 with no recognizable body is unknown (not retried).
  const c400 = classifyApiError(Object.assign(new Error('HTTP 400 Bad Request'), { status: 400 }))
  assert.equal(c400.category, 'unknown')
  assert.equal(c400.retryable, false)
  ok('unrecognized errors (incl. a bare 400) classify as unknown and are not retried')
}

{
  // llama.cpp answers a chat template's refusal with HTTP 500; the same body would fail again (#884).
  for (const msg of [
    'HTTP 500: System message must be at the beginning.',
    'HTTP 500: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.',
    'HTTP 500: Conversation roles must alternate user/assistant/user/assistant/...',
  ]) {
    const c = classifyApiError(Object.assign(new Error(msg), { status: 500 }))
    assert.equal(c.category, 'unknown', msg)
    assert.equal(c.retryable, false, msg)
    assert.equal(c.status, 500)
  }
  assert.equal(classifyApiError(Object.assign(new Error('HTTP 500 Internal Server Error'), { status: 500 })).retryable, true)
  ok("a chat template's refusal on HTTP 500 is not retried; a bare 500 still is")
}

{
  // The level a reasoning-effort refusal takes instead: the nearest it names, a tie going to more reasoning (#884).
  const refusal = (refused: string, levels: string) => new Error(`HTTP 500: Unexpected reasoning effort ${refused}. Supported types are ${levels}.`)
  assert.equal(supportedEffort(refusal('high', 'xhigh (default), medium, and low')), 'xhigh')
  assert.equal(supportedEffort(refusal('minimal', 'low, medium, and high')), 'low')
  assert.equal(supportedEffort(refusal('xhigh', 'low and medium')), 'medium')
  assert.equal(supportedEffort(refusal('high', 'high')), null)
  assert.equal(supportedEffort(new Error('HTTP 500: System message must be at the beginning.')), null)
  assert.equal(supportedEffort(new Error('HTTP 500 Internal Server Error')), null)
  ok("a reasoning-effort refusal names the nearest level the server takes; other errors name none")
}

{
  // Tolerant of odd shapes (string, plain object, undefined).
  assert.equal(classifyApiError('overloaded').category, 'transient')
  assert.equal(classifyApiError({ message: 'HTTP 502 Bad Gateway', status: 502 }).category, 'transient')
  assert.equal(classifyApiError(undefined).category, 'unknown')
  assert.equal(extractErrorStatus({ statusCode: 503 }), 503)
  assert.equal(extractErrorStatus('HTTP 429 Too Many Requests'), 429)
  ok('classifier tolerates string / plain-object / undefined error shapes')
}

// --- jittered backoff --------------------------------------------------------
{
  // With random()=0 the result is exactly the capped exponential term (no jitter).
  const zero = () => 0
  assert.equal(jitteredBackoff(1, 500, 15_000, { random: zero }), 500)
  assert.equal(jitteredBackoff(2, 500, 15_000, { random: zero }), 1000)
  assert.equal(jitteredBackoff(3, 500, 15_000, { random: zero }), 2000)
  assert.equal(jitteredBackoff(4, 500, 15_000, { random: zero }), 4000)
  ok('with no jitter the delay is base*2^(attempt-1)')
}

{
  // The pre-jitter term is capped at max and never exceeds it.
  const zero = () => 0
  assert.equal(jitteredBackoff(10, 500, 15_000, { random: zero }), 15_000, 'capped at max')
  assert.equal(jitteredBackoff(20, 500, 15_000, { random: zero }), 15_000, 'still capped, no overflow')
  // Huge attempt must not overflow to Infinity/NaN.
  const huge = jitteredBackoff(1000, 500, 15_000, { random: zero })
  assert.ok(Number.isFinite(huge) && huge === 15_000, 'no overflow for a very large attempt')
  ok('the capped exponential term is bounded by max and does not overflow')
}

{
  // With random()=1 the jitter adds exactly +50% of the capped delay.
  const one = () => 1
  assert.equal(jitteredBackoff(1, 500, 15_000, { random: one }), 750) // 500 + 0.5*500
  assert.equal(jitteredBackoff(3, 500, 15_000, { random: one }), 3000) // 2000 + 0.5*2000
  assert.equal(jitteredBackoff(10, 500, 15_000, { random: one }), 22_500) // 15000 + 0.5*15000
  ok('jitter adds at most +50% of the capped delay (max random)')
}

{
  // Monotonic non-decreasing in attempt for a fixed jitter source, and every value
  // sits within [capped, 1.5*capped].
  const mid = () => 0.5
  let prev = -1
  for (let attempt = 1; attempt <= 8; attempt++) {
    const delay = jitteredBackoff(attempt, 500, 15_000, { random: mid })
    const capped = Math.min(500 * 2 ** (attempt - 1), 15_000)
    assert.ok(delay >= capped, `attempt ${attempt}: delay ${delay} >= capped ${capped}`)
    assert.ok(delay <= capped * 1.5 + 0.5, `attempt ${attempt}: delay ${delay} <= 1.5*capped`)
    assert.ok(delay >= prev, `attempt ${attempt}: monotonic non-decreasing`)
    prev = delay
  }
  ok('delay is monotonic non-decreasing and stays within [capped, 1.5*capped]')
}

{
  // Degenerate inputs are clamped, never NaN/negative.
  assert.equal(jitteredBackoff(0, 500, 15_000, { random: () => 0 }), 500, 'attempt<1 treated as attempt 1')
  assert.equal(jitteredBackoff(-5, 500, 15_000, { random: () => 0 }), 500)
  assert.ok(jitteredBackoff(3, -1, 15_000, { random: () => 0 }) >= 0, 'negative base clamps to >=0')
  const r = jitteredBackoff(3, 500, 15_000, { random: () => 2 }) // out-of-range random clamped to 1
  assert.equal(r, 3000, 'random()>1 is clamped to 1')
  ok('degenerate attempt/base/random inputs are clamped to safe finite values')
}

console.log(`\n${passed} checks passed.`)
