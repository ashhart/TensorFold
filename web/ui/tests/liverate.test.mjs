import assert from 'node:assert/strict'
import { test } from 'node:test'
import { charsPerToken, DEFAULT_CHARS_PER_TOKEN, liveRate, prefillRate } from '../src/liverate.ts'

test('the ratio comes from the last reply the engine counted', () => {
  const msgs = [
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: 'x'.repeat(1350), stats: { tokens: 300, measured: 'engine' } },
    { role: 'user', content: 'again' },
    { role: 'assistant', content: 'y'.repeat(50), stats: { tokens: null, measured: 'counted' } },   // not the engine's count
  ]
  assert.equal(charsPerToken(msgs), 4.5)
  assert.equal(charsPerToken([]), DEFAULT_CHARS_PER_TOKEN)
  assert.equal(charsPerToken([{ role: 'assistant', content: 'z'.repeat(100), stats: { tokens: 90, measured: 'engine' } }]), 1.5)
})

test('the live speed counts text, not stream events', () => {
  // the GB10 reply above: 1,350 characters in 2.4 s is about 125 tok/s, where its 171 events read 71
  assert.equal(Math.round(liveRate(1350, 2.4, 4.5)), 125)
  assert.equal(liveRate(1350, 0.1, 4.5), null)
  assert.equal(liveRate(0, 2, 4.5), null)
})

test("the prompt's reading speed counts only the tokens not restored from the cache", () => {
  assert.equal(prefillRate({ prefill_seconds: 0.5 }, 1200, 200), 2000)
  assert.equal(prefillRate({ prefill_seconds: 0.5 }, 1200, 1200), undefined)   // all from the cache: nothing read
  assert.equal(prefillRate({}, 1200, 0), undefined)
  assert.equal(prefillRate(undefined, 1200, 0), undefined)
})
