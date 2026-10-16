// Run: jiti electron/agent/loop-resilience.test.ts
//
// Covers the transient-error resilience policy the run loop applies around the
// model stream: streamRetryDecision (pure branching) + abortableDelay (the
// abort-aware backoff sleep). The full runAgent drive is intentionally NOT
// exercised here — like the other loop tests in this repo, the loop's behavior is
// verified through its extracted pure helpers.
import assert from 'node:assert/strict'
import { streamRetryDecision, abortableDelay, cancelledToolMutationPatch, compactionMadeProgress, isDelegatedTransport } from './loop.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('loop resilience tests:')

const base = { compactRetries: 0, canCompact: true, random: () => 0 }

{
  const patch = cancelledToolMutationPatch({ changedPaths: ['src/partial.ts', 'src/partial.ts'], meta: 'exit 130' })
  assert.deepEqual(patch?.changedPaths, ['src/partial.ts'])
  assert.equal(patch?.state, 'error')
  assert.match(patch?.output ?? '', /changes remain recorded/)
  assert.equal(cancelledToolMutationPatch({ changedPaths: [] }), null)
  ok('cancelled tools retain explicit partial-mutation ownership evidence')
}

// --- delegated engines are never re-run ----------------------------------------
{
  const timeout = new Error('Claude CLI total timeout exceeded.')
  assert.equal(streamRetryDecision(timeout, 1, { ...base, delegated: true }).kind, 'fail')
  assert.equal(streamRetryDecision(Object.assign(new Error('HTTP 503'), { status: 503 }), 1, { ...base, delegated: true }).kind, 'fail')
  for (const t of ['codex-cli', 'claude-cli', 'cursor-cli', 'gemini-cli', 'pi-cli', 'opencode', 'kilo']) assert.ok(isDelegatedTransport(t), t)
  for (const t of ['chat-completions', undefined, null, '']) assert.equal(isDelegatedTransport(t as any), false)
  ok('a delegated CLI turn that failed (maybe after editing files) is never retried')
}

// --- transient → retry with backoff -----------------------------------------
{
  const d = streamRetryDecision(Object.assign(new Error('HTTP 503'), { status: 503 }), 1, base)
  assert.equal(d.kind, 'retry')
  if (d.kind === 'retry') {
    assert.ok(d.delayMs > 0, 'a retry carries a positive backoff delay')
    assert.match(d.reason, /transient/i)
  }
  // ECONNRESET mid-stream is the classic "stops after 30s" — also a retry.
  const reset = streamRetryDecision(Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' }), 2, base)
  assert.equal(reset.kind, 'retry')
  ok('a transient stream error (5xx / dropped socket) yields a backoff retry')
}

{
  // Backoff grows with the attempt (random injected to 0 so it is deterministic).
  const a1 = streamRetryDecision(new Error('overloaded'), 1, base)
  const a2 = streamRetryDecision(new Error('overloaded'), 2, base)
  const a3 = streamRetryDecision(new Error('overloaded'), 3, base)
  assert.ok(a1.kind === 'retry' && a2.kind === 'retry' && a3.kind === 'retry')
  if (a1.kind === 'retry' && a2.kind === 'retry' && a3.kind === 'retry') {
    assert.ok(a2.delayMs > a1.delayMs && a3.delayMs > a2.delayMs, 'backoff is monotonic across attempts')
  }
  ok('retry backoff increases monotonically with the attempt number')
}

// --- retries exhausted → fail ------------------------------------------------
{
  // Past the retry cap (default 3) a transient error must fail cleanly, never loop forever.
  const d = streamRetryDecision(new Error('overloaded'), 4, base)
  assert.equal(d.kind, 'fail', 'transient error past the cap fails cleanly (no infinite retry)')
  // An explicit lower cap is honored.
  const capped = streamRetryDecision(new Error('overloaded'), 2, { ...base, maxRetries: 1 })
  assert.equal(capped.kind, 'fail')
  ok('a transient error beyond the retry cap fails (bounded, never infinite)')
}

// --- context overflow → compact ----------------------------------------------
{
  const d = streamRetryDecision(new Error('context_length_exceeded: too many tokens'), 1, base)
  assert.equal(d.kind, 'compact', 'context overflow compacts rather than blind-retrying')
  // Once the compaction cap is hit, it fails instead of compacting forever.
  const exhausted = streamRetryDecision(new Error('maximum context length is 128000 tokens'), 1, { ...base, compactRetries: 2 })
  assert.equal(exhausted.kind, 'fail', 'context overflow past the compaction cap fails')
  // If compaction is unavailable (autoCompact off / no context limit), overflow fails — it
  // can't be retried blindly.
  const noCompact = streamRetryDecision(new Error('context_length_exceeded'), 1, { ...base, canCompact: false })
  assert.equal(noCompact.kind, 'fail', 'context overflow with compaction disabled fails cleanly')
  ok('context overflow compacts (capped); fails when compaction is exhausted or unavailable')
}

{
  assert.equal(compactionMadeProgress(null, 10_000, 10_000), false, 'null compaction is not success')
  assert.equal(compactionMadeProgress({} as any, 10_000, 10_000), false, 'an unchanged transcript is not progress')
  assert.equal(compactionMadeProgress({} as any, 10_000, 7_500), true)
  ok('overflow retries only after compaction actually reduces the transcript')
}

// --- non-retryable → fail immediately ----------------------------------------
{
  for (const err of [
    Object.assign(new Error('HTTP 401 Unauthorized'), { status: 401 }),
    new Error('rejected by our content policy'),
    new Error('some unrecognized novel failure'),
    Object.assign(new Error('HTTP 400 Bad Request'), { status: 400 }),
  ]) {
    assert.equal(streamRetryDecision(err, 1, base).kind, 'fail', `${err.message} must fail without retry`)
  }
  ok('auth / content-policy / unknown / bare-400 errors fail immediately (no retry, no compact)')
}

// --- abortableDelay ----------------------------------------------------------
{
  // A zero/negative delay resolves immediately.
  const c = new AbortController()
  const t0 = Date.now()
  await abortableDelay(0, c.signal)
  assert.ok(Date.now() - t0 < 50, 'a 0ms delay resolves immediately')
  ok('abortableDelay(0) resolves immediately')
}

{
  // An already-aborted signal resolves immediately (does not reject).
  const c = new AbortController()
  c.abort()
  const t0 = Date.now()
  await abortableDelay(10_000, c.signal)
  assert.ok(Date.now() - t0 < 50, 'an already-aborted signal short-circuits a long delay')
  ok('abortableDelay resolves immediately when the signal is already aborted')
}

{
  // Aborting mid-wait resolves early (well before the full delay), and does not reject.
  const c = new AbortController()
  const t0 = Date.now()
  setTimeout(() => c.abort(), 20)
  await abortableDelay(5_000, c.signal)
  const elapsed = Date.now() - t0
  assert.ok(elapsed < 1_000, `abort during the wait resolves early (elapsed ${elapsed}ms << 5000ms)`)
  ok('abortableDelay resolves early when the signal aborts mid-wait')
}

{
  // A normal short delay actually waits (sanity: it is a real timer, not a no-op).
  const c = new AbortController()
  const t0 = Date.now()
  await abortableDelay(30, c.signal)
  assert.ok(Date.now() - t0 >= 25, 'a normal delay waits roughly its duration')
  ok('abortableDelay waits for the full duration when not aborted')
}

console.log(`\n${passed} checks passed.`)
