// Run: node electron/agent/compaction.test.ts   (Node 22.18+ strips TS types natively)
// Proves the compaction boundary never produces an API-invalid message sequence.
import assert from 'node:assert'
import type { ApiMessage } from './types'
import { deterministicCompactionSummary, estimateTokens, planCompaction, shouldSkipCompactionForThrash, nextIneffectiveStreak, INEFFECTIVE_STREAK_LIMIT } from './compaction.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const sys = (): ApiMessage => ({ role: 'system', content: 'You are TensorFold Studio.' })
const user = (t: string): ApiMessage => ({ role: 'user', content: t })
const asstText = (t: string): ApiMessage => ({ role: 'assistant', content: t })
const asstCall = (id: string): ApiMessage => ({ role: 'assistant', content: null, tool_calls: [{ id, type: 'function', function: { name: 'read_file', arguments: '{"path":"a.ts"}' } }] })
const tool = (id: string): ApiMessage => ({ role: 'tool', tool_call_id: id, name: 'read_file', content: 'file contents' })

// The real API invariant: every `tool` message must trace back, through only
// other tool messages, to an assistant message that carried tool_calls.
function assertApiValid(msgs: ApiMessage[], label: string) {
  for (let i = 0; i < msgs.length; i++) {
    if (msgs[i].role !== 'tool') continue
    let j = i - 1
    while (j >= 0 && msgs[j].role === 'tool') j--
    assert.ok(j >= 0 && msgs[j].role === 'assistant' && (msgs[j].tool_calls?.length ?? 0) > 0,
      `${label}: tool message at ${i} is not anchored to an assistant tool_calls turn`)
  }
}

// Reconstruct exactly as compactThread does: head + one summary user msg + tail.
function applyPlan(msgs: ApiMessage[], keepRecent: number): ApiMessage[] | null {
  const plan = planCompaction(msgs, keepRecent)
  if (!plan) return null
  const { cut, sysCount } = plan
  const summaryMsg: ApiMessage = { role: 'user', content: '[compacted summary]' }
  const out = [...msgs.slice(0, sysCount), summaryMsg, ...msgs.slice(cut)]
  // tail must never start on an orphan tool result
  assert.notStrictEqual(msgs[cut]?.role, 'tool', 'cut landed on a tool message')
  return out
}

console.log('compaction boundary tests:')

// 1) cut would land mid tool-group → must advance past the tool results
{
  const msgs = [sys(), user('go'), asstCall('c1'), tool('c1'), tool('c1b'), asstText('done'), user('more'), asstCall('c2'), tool('c2'), asstText('ok')]
  const out = applyPlan(msgs, 6)!
  assert.ok(out, 'should compact')
  assert.strictEqual(out[0].role, 'system', 'system stays at head')
  assertApiValid(out, 'case1')
  ok('advances cut past tool results so the tail is API-valid')
}

// 2) every keepRecent from 1..8 yields a valid sequence (fuzz the boundary)
{
  const msgs = [sys(), user('a'), asstCall('c1'), tool('c1'), asstText('b'), asstCall('c2'), tool('c2'), tool('c2b'), asstText('c'), user('d'), asstCall('c3'), tool('c3'), asstText('e')]
  for (let k = 1; k <= 8; k++) {
    const out = applyPlan(msgs, k)
    if (out) { assert.strictEqual(out[0].role, 'system'); assertApiValid(out, `case2 k=${k}`) }
  }
  ok('valid for every keepRecent boundary 1..8')
}

// 3) no system message → sysCount 0, still valid
{
  const msgs = [user('a'), asstCall('c1'), tool('c1'), asstText('b'), user('c'), asstText('d'), user('e'), asstText('f')]
  const out = applyPlan(msgs, 3)
  if (out) assertApiValid(out, 'case3')
  ok('handles a conversation with no system prompt')
}

// 4) too short to fold → returns null (no churn)
{
  assert.strictEqual(planCompaction([sys(), user('a'), asstText('b')], 6), null)
  ok('returns null when there is nothing worth folding')
}

// 5) token estimate is monotonic and sane
{
  const small = estimateTokens([user('hi')])
  const big = estimateTokens([user('x'.repeat(4000))])
  assert.ok(big > small && big >= 950 && big <= 1100, `~1k tokens for 4k chars, got ${big}`)
  ok('estimateTokens ≈ chars/4')
}

// 6) anti-thrash streak: a pass that saves well under 10% extends the streak;
// a pass that saves a useful amount resets it to 0.
{
  assert.strictEqual(nextIneffectiveStreak(undefined, 5, 1000), 1, 'first low-savings pass -> streak 1')
  assert.strictEqual(nextIneffectiveStreak(1, 5, 1000), 2, '2nd consecutive low-savings pass -> streak 2')
  assert.strictEqual(nextIneffectiveStreak(2, 5, 1000), 3, 'streak keeps climbing while ineffective')
  assert.strictEqual(nextIneffectiveStreak(3, 500, 1000), 0, 'a useful pass (>=10%) resets the streak to 0')
  assert.strictEqual(nextIneffectiveStreak(2, 0, 0), 2, 'a zero-size fold leaves the streak unchanged (nothing to judge)')
  ok('nextIneffectiveStreak: extends on low savings, resets on a useful pass')
}

// 7) anti-thrash skip: only skips once the streak has actually reached the limit,
// AND the thread hasn't grown enough since the last pass to plausibly be worth it.
{
  assert.strictEqual(INEFFECTIVE_STREAK_LIMIT, 2, 'sanity: this test assumes the documented limit of 2')
  assert.strictEqual(shouldSkipCompactionForThrash(1, 10_000, 9_000), false, 'streak below the limit -> never skip')
  assert.strictEqual(shouldSkipCompactionForThrash(2, 9_500, 9_000), true, '2nd consecutive ineffective pass + thread barely grown -> skip')
  assert.strictEqual(shouldSkipCompactionForThrash(2, 12_000, 9_000), false, 'streak at the limit BUT thread grew substantially -> run again')
  assert.strictEqual(shouldSkipCompactionForThrash(undefined, 10_000, 9_000), false, 'no streak yet -> never skip')
  assert.strictEqual(shouldSkipCompactionForThrash(5, 10_000, undefined), false, 'no prior pass to compare against -> never skip')
  ok('shouldSkipCompactionForThrash: only skips at/above the streak limit when the thread has not grown enough')
}

// 8) a very large final assistant/tool group must not push the latest user task into
// the folded slice. The current task contract always survives verbatim.
{
  const msgs = [sys(), user('old'), asstText('old answer'), user('LATEST USER TASK'), asstCall('c1'), tool('c1'), tool('c1b'), asstText('working')]
  const plan = planCompaction(msgs, 2)!
  assert.ok(plan)
  assert.ok(plan.cut <= 3, `cut ${plan.cut} must retain latest user at index 3`)
  assert.equal(msgs.slice(plan.cut).some((message) => message.role === 'user' && message.content === 'LATEST USER TASK'), true)
  ok('preserves the latest user turn verbatim')
}

// 10) the latest-user rule must not disable folding for long single-user runs. A task
// thread that never receives a second user message is exactly the auto-compaction
// audience; the verified summary is required to preserve the user goal, so folding
// the original request is the fallback — not "never compact".
{
  const msgs = [sys(), user('FIX THE FLAKY TEST'), ...Array.from({ length: 40 }, (_, i) => (i % 2 ? tool(`c${i}`) : asstCall(`c${i}`)))]
  const plan = planCompaction(msgs, 8)
  assert.ok(plan, 'single-user agentic transcript must remain compactable')
  assert.ok(plan!.cut > 1, `cut ${plan?.cut} must fold more than the system prompt alone`)
  ok('single-user agentic transcripts stay compactable')
}

// 9) deterministic fallback is bounded and retains both ends of the source.
{
  const messages = [user(`HEAD-${'a'.repeat(9000)}`), asstText(`TAIL-${'z'.repeat(9000)}`)]
  const summary = deterministicCompactionSummary(messages, 4000)
  assert.ok(summary.length <= 4100)
  assert.match(summary, /HEAD-/)
  assert.match(summary, /TAIL-/)
  assert.match(summary, /message middle omitted/)
  ok('deterministic summary fallback is bounded and keeps source head and tail')
}

console.log(`\n${passed} checks passed.`)
