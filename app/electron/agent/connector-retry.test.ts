// Run: jiti electron/agent/connector-retry.test.ts
import assert from 'node:assert/strict'
import { fetchWithRetry, parseRetryAfter, shouldRetryStatus } from './connector-retry.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('connector-retry tests:')

// --- a fake Response good enough for the wrapper (status + headers.get + body.cancel) ---
function fakeResponse(status: number, headers: Record<string, string> = {}) {
  const lower: Record<string, string> = {}
  for (const k of Object.keys(headers)) lower[k.toLowerCase()] = headers[k]
  let cancelled = false
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (n: string) => (n.toLowerCase() in lower ? lower[n.toLowerCase()] : null) },
    body: { cancel: async () => { cancelled = true } },
    get _cancelled() { return cancelled },
  }
}

// A scripted fetch: each call shifts the next outcome. An outcome is either a Response
// (returned) or an Error (thrown). Records how many times it was invoked.
function scriptedFetch(outcomes: Array<any>) {
  const calls: any[] = []
  const fn = async (_input: any, init: any) => {
    calls.push(init)
    const next = outcomes.shift()
    if (next instanceof Error) throw next
    if (typeof next === 'function') return next()
    return next
  }
  return Object.assign(fn, { calls })
}

const noWait = async () => {} // never actually sleep in tests
const fixedRandom = () => 0 // deterministic backoff (no jitter)

// ---- shouldRetryStatus: transient family retries; auth/not-found/413/2xx do not ----
{
  for (const s of [408, 409, 425, 429, 500, 502, 503, 504, 529]) {
    assert.equal(shouldRetryStatus(s), true, `HTTP ${s} should retry`)
  }
  for (const s of [200, 201, 204, 301, 400, 401, 403, 404, 413, 422]) {
    assert.equal(shouldRetryStatus(s), false, `HTTP ${s} should NOT retry`)
  }
  ok('shouldRetryStatus: 429/5xx/retryable-4xx retry; auth/404/413/2xx do not')
}

// ---- parseRetryAfter: integer & fractional seconds, HTTP-date, garbage ----
{
  assert.equal(parseRetryAfter('5', 0), 5000, 'integer seconds -> ms')
  assert.equal(parseRetryAfter('0.5', 0), 500, 'fractional seconds -> ms')
  assert.equal(parseRetryAfter(null, 0), null, 'missing header -> null')
  assert.equal(parseRetryAfter('', 0), null, 'empty header -> null')
  assert.equal(parseRetryAfter('not-a-thing', 0), null, 'garbage -> null')
  // HTTP-date 10s in the future relative to `now`.
  const now = Date.parse('2026-01-01T00:00:00Z')
  const future = new Date(now + 10_000).toUTCString()
  assert.equal(parseRetryAfter(future, now), 10_000, 'HTTP-date -> delta ms')
  // A date in the PAST clamps to 0, never negative.
  const past = new Date(now - 10_000).toUTCString()
  assert.equal(parseRetryAfter(past, now), 0, 'past HTTP-date clamps to 0')
  ok('parseRetryAfter: seconds, fractional, HTTP-date, past-clamp, garbage')
}

// ---- happy path: a 200 returns immediately, fetch called exactly once ----
{
  const f = scriptedFetch([fakeResponse(200)])
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait })
  assert.equal(res.status, 200)
  assert.equal(f.calls.length, 1, 'no retry on success')
  ok('200 returns immediately with no retry')
}

// ---- retries a 503 then succeeds; returns the eventual 200 ----
{
  const f = scriptedFetch([fakeResponse(503), fakeResponse(200)])
  const onRetry: any[] = []
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, random: fixedRandom, onRetry: (i) => onRetry.push(i) })
  assert.equal(res.status, 200)
  assert.equal(f.calls.length, 2, 'one retry')
  assert.equal(onRetry.length, 1, 'onRetry fired once')
  assert.equal(onRetry[0].status, 503)
  assert.ok(onRetry[0].delayMs >= 0)
  ok('retries a transient 503 then returns the 200')
}

// ---- retries a thrown ECONNRESET then succeeds ----
{
  const netErr = Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' })
  const f = scriptedFetch([netErr, fakeResponse(200)])
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, random: fixedRandom })
  assert.equal(res.status, 200)
  assert.equal(f.calls.length, 2, 'retried after the dropped socket')
  ok('retries a dropped-socket network error then succeeds')
}

// ---- exhausts attempts on a persistent 503 and hands back the last 503 ----
{
  const f = scriptedFetch([fakeResponse(503), fakeResponse(503), fakeResponse(503)])
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, random: fixedRandom, maxAttempts: 3 })
  assert.equal(res.status, 503, 'returns the final failing response (caller handles !res.ok)')
  assert.equal(f.calls.length, 3, 'tried exactly maxAttempts times')
  ok('persistent 503 returns the last response after maxAttempts tries')
}

// ---- a persistent thrown error is rethrown after exhausting attempts ----
{
  const netErr = Object.assign(new Error('ETIMEDOUT'), { code: 'ETIMEDOUT' })
  const f = scriptedFetch([netErr, netErr, netErr])
  await assert.rejects(
    () => fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, random: fixedRandom, maxAttempts: 3 }),
    /ETIMEDOUT/,
  )
  assert.equal(f.calls.length, 3, 'tried maxAttempts times before giving up')
  ok('persistent network error is rethrown after maxAttempts')
}

// ---- does NOT retry a 401 (auth): returns immediately ----
{
  const f = scriptedFetch([fakeResponse(401)])
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait })
  assert.equal(res.status, 401)
  assert.equal(f.calls.length, 1, 'auth failure is not retried')
  ok('401 is returned immediately, never retried')
}

// ---- does NOT retry a 404 ----
{
  const f = scriptedFetch([fakeResponse(404)])
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait })
  assert.equal(res.status, 404)
  assert.equal(f.calls.length, 1)
  ok('404 is returned immediately, never retried')
}

// ---- does NOT retry a non-transient thrown error (e.g. a TypeError) ----
{
  const f = scriptedFetch([new TypeError('bad url')])
  await assert.rejects(() => fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait }), /bad url/)
  assert.equal(f.calls.length, 1, 'a non-transient throw is not retried')
  ok('non-transient thrown error is not retried')
}

// ---- honors Retry-After (seconds) for the delay ----
{
  const f = scriptedFetch([fakeResponse(429, { 'Retry-After': '2' }), fakeResponse(200)])
  let sleptMs = -1
  const sleep = async (ms: number) => { sleptMs = ms }
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep, random: fixedRandom })
  assert.equal(res.status, 200)
  assert.equal(sleptMs, 2000, 'waited the Retry-After seconds, not the backoff curve')
  ok('honors Retry-After header for the wait')
}

// ---- gives up (returns the 429) when Retry-After exceeds the cap ----
{
  const f = scriptedFetch([fakeResponse(429, { 'Retry-After': '120' }), fakeResponse(200)])
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, retryAfterCapMs: 30_000 })
  assert.equal(res.status, 429, 'did not wait an absurd Retry-After; handed back the 429')
  assert.equal(f.calls.length, 1, 'did not retry past the cap')
  ok('Retry-After beyond the cap aborts the retry (returns the 429)')
}

// ---- cancels the body of a discarded retryable response (no socket leak) ----
{
  const failing = fakeResponse(503)
  const f = scriptedFetch([failing, fakeResponse(200)])
  await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, random: fixedRandom })
  assert.equal(failing._cancelled, true, 'the discarded 503 response body was cancelled')
  ok('discarded retryable response body is cancelled (no leak)')
}

// ---- already-aborted signal: throws and never calls fetch ----
{
  const ctrl = new AbortController()
  ctrl.abort()
  const f = scriptedFetch([fakeResponse(200)])
  await assert.rejects(() => fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, signal: ctrl.signal }))
  assert.equal(f.calls.length, 0, 'an already-aborted signal short-circuits before any fetch')
  ok('already-aborted signal short-circuits before fetching')
}

// ---- maxAttempts:1 disables retry entirely ----
{
  const f = scriptedFetch([fakeResponse(503), fakeResponse(200)])
  const res = await fetchWithRetry('https://x/api', {}, { fetchImpl: f, sleep: noWait, maxAttempts: 1 })
  assert.equal(res.status, 503, 'with maxAttempts:1 the first (failing) response is returned as-is')
  assert.equal(f.calls.length, 1)
  ok('maxAttempts:1 disables retry')
}

// ---- the injected signal is forwarded into the underlying fetch init ----
{
  const ctrl = new AbortController()
  const f = scriptedFetch([fakeResponse(200)])
  await fetchWithRetry('https://x/api', { headers: { a: '1' } }, { fetchImpl: f, sleep: noWait, signal: ctrl.signal })
  assert.equal(f.calls[0].signal, ctrl.signal, 'signal threaded into fetch init')
  assert.equal(f.calls[0].headers.a, '1', 'original init preserved')
  ok('signal + original init are forwarded to the underlying fetch')
}

console.log(`\n${passed} checks passed.`)
