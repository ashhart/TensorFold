// Run: npx jiti electron/agent/turn-stats.test.ts
import assert from 'node:assert/strict'
import { createTurnStats, finalizeTurnStats, readCortheonReport, readEngineTimings, recordModelCall, recordToolCalls } from './turn-stats'

const route = { model: 'TensorFold/Nemotron', providerId: 'tensorfold' }

{
  const acc = createTurnStats()
  recordModelCall(acc, { requestAt: 1_000, firstTokenAt: 1_400, lastTokenAt: 2_400, completionTokens: 201 })
  recordToolCalls(acc, 2)
  recordModelCall(acc, { requestAt: 3_000, firstTokenAt: 3_100, lastTokenAt: 4_100, completionTokens: 101 })
  const stats = finalizeTurnStats(acc, route)
  assert.equal(stats.modelCalls, 2)
  assert.equal(stats.toolCalls, 2)
  assert.equal(stats.completionTokens, 302)
  // (200 + 100) decode tokens over 2 s of decode.
  assert.equal(stats.decodeTokensPerSecond, 150)
  // First token comes from the run's first call only.
  assert.equal(stats.firstTokenMs, 400)
}

{
  // No usage event: timings that were measured stay, token facts stay absent (never estimated).
  const acc = createTurnStats()
  recordModelCall(acc, { requestAt: 1_000, firstTokenAt: 1_250, lastTokenAt: 3_000, completionTokens: null })
  const stats = finalizeTurnStats(acc, route)
  assert.equal(stats.completionTokens, undefined)
  assert.equal(stats.decodeTokensPerSecond, undefined)
  assert.equal(stats.firstTokenMs, 250)
}

{
  // A single-token reply has no decode interval; the first call that streamed a token sets first-token time.
  const acc = createTurnStats()
  recordModelCall(acc, { requestAt: 1_000, firstTokenAt: null, lastTokenAt: null, completionTokens: 0 })
  recordModelCall(acc, { requestAt: 2_000, firstTokenAt: 2_300, lastTokenAt: 2_300, completionTokens: 1 })
  const stats = finalizeTurnStats(acc, route)
  assert.equal(stats.completionTokens, 1)
  assert.equal(stats.decodeTokensPerSecond, undefined)
  assert.equal(stats.firstTokenMs, 300)
}

{
  // Garbage in: negative or non-finite numbers are ignored.
  const acc = createTurnStats()
  recordModelCall(acc, { requestAt: 5_000, firstTokenAt: 4_000, lastTokenAt: Number.NaN, completionTokens: -3 })
  recordToolCalls(acc, -1)
  const stats = finalizeTurnStats(acc, route)
  assert.deepEqual(stats, { model: route.model, providerId: 'tensorfold', modelCalls: 1, toolCalls: 0 })
}

{
  // The engine's numbers win over the client clock: decode and prefill are pooled across the run's calls.
  const acc = createTurnStats()
  const first = readEngineTimings({ tokens_per_second: 200, seconds: 1, prefill_seconds: 0.5, time_to_first_token: 0.12 })
  const second = readEngineTimings({ tokens_per_second: 100, seconds: 2, prefill_seconds: 0.25, time_to_first_token: 0.08 })
  recordModelCall(acc, { requestAt: 1_000, firstTokenAt: 1_900, lastTokenAt: 2_000, completionTokens: 201, promptTokens: 2_000, cachedTokens: 500, engine: first })
  recordToolCalls(acc, 1)
  recordModelCall(acc, { requestAt: 3_000, firstTokenAt: 3_100, lastTokenAt: 3_200, completionTokens: 201, promptTokens: 2_300, cachedTokens: 2_000, engine: second })
  const stats = finalizeTurnStats(acc, route)
  // (200 + 200) decoded tokens over (1 + 2) engine seconds, not the 200 ms the client saw.
  assert.equal(stats.decodeTokensPerSecond, 133.3)
  assert.equal(stats.decodeSeconds, 3)
  // (1,500 + 300) new prompt tokens over (0.5 + 0.25) prefill seconds.
  assert.equal(stats.prefillTokensPerSecond, 2400)
  assert.equal(stats.promptTokens, 4_300)
  assert.equal(stats.cachedTokens, 2_500)
  // First token is the first call's engine figure.
  assert.equal(stats.firstTokenMs, 120)
}

{
  // A reply whose prompt was fully cached reads no new tokens: prefill is omitted, never infinite.
  const acc = createTurnStats()
  recordModelCall(acc, { requestAt: 1_000, firstTokenAt: 1_050, lastTokenAt: 1_500, completionTokens: 41, promptTokens: 900, cachedTokens: 900, engine: readEngineTimings({ tokens_per_second: 80, seconds: 0.5, prefill_seconds: 0, time_to_first_token: 0.02 }) })
  const stats = finalizeTurnStats(acc, route)
  assert.equal(stats.prefillTokensPerSecond, undefined)
  assert.equal(stats.decodeTokensPerSecond, 80)
  assert.equal(stats.cachedTokens, 900)
  // A null prefill (the engine did not time it) is unknown, not zero.
  const unknown = createTurnStats()
  recordModelCall(unknown, { requestAt: 0, firstTokenAt: null, lastTokenAt: null, completionTokens: 1, promptTokens: 10, cachedTokens: 0, engine: readEngineTimings({ tokens_per_second: 0, seconds: 0, prefill_seconds: null, time_to_first_token: null }) })
  const unknownStats = finalizeTurnStats(unknown, route)
  assert.equal(unknownStats.prefillTokensPerSecond, undefined)
  assert.equal(unknownStats.decodeTokensPerSecond, undefined)
  assert.equal(unknownStats.firstTokenMs, undefined)
}

{
  // No `tensorfold` object: the stream's own clock is the fallback, and missing cache counts stay absent.
  assert.equal(readEngineTimings(undefined), null)
  assert.equal(readEngineTimings({ tokens_per_second: 'fast', seconds: 1 }), null)
  assert.equal(readEngineTimings([1, 2]), null)
  const acc = createTurnStats()
  recordModelCall(acc, { requestAt: 1_000, firstTokenAt: 1_400, lastTokenAt: 2_400, completionTokens: 201, promptTokens: 80, cachedTokens: null, engine: null })
  const stats = finalizeTurnStats(acc, route)
  assert.equal(stats.decodeTokensPerSecond, 200)
  assert.equal(stats.decodeSeconds, 1)
  assert.equal(stats.prefillTokensPerSecond, undefined)
  assert.equal(stats.promptTokens, 80)
  assert.equal(stats.cachedTokens, undefined)
  assert.equal(stats.firstTokenMs, 400)
}

{
  // The engine's `cortheon` object: counts and web sources only; anything else is dropped.
  const report = readCortheonReport({ task: 'research_answer', research_first: true, rounds: 1, searches: 2, pages_read: 2,
    sources: ['https://ziglang.org/download/0.17.0/release-notes.html', 'file:///etc/passwd', 'javascript:alert(1)', 42, 'https://ziglang.org/news/0.17.0-released/'], checks: [] })
  assert.deepEqual(report, { task: 'research_answer', searches: 2, pagesRead: 2, sources: ['https://ziglang.org/download/0.17.0/release-notes.html', 'https://ziglang.org/news/0.17.0-released/'] })
  assert.equal(readCortheonReport(null), null)
  assert.equal(readCortheonReport([1]), null)
  assert.deepEqual(readCortheonReport({ searches: -1, pages_read: 'x' }), { task: null, searches: 0, pagesRead: 0, sources: [] })

  const acc = createTurnStats()
  recordModelCall(acc, { requestAt: 1, firstTokenAt: 2, lastTokenAt: 3, completionTokens: 2, cortheon: report })
  recordModelCall(acc, { requestAt: 4, firstTokenAt: 5, lastTokenAt: 6, completionTokens: 2, cortheon: { task: null, searches: 1, pagesRead: 0, sources: ['https://ziglang.org/news/0.17.0-released/'] } })
  const stats = finalizeTurnStats(acc, route)
  assert.deepEqual(stats.cortheon, { task: 'research_answer', searches: 3, pagesRead: 2, sources: ['https://ziglang.org/download/0.17.0/release-notes.html', 'https://ziglang.org/news/0.17.0-released/'] })
  assert.equal(finalizeTurnStats(createTurnStats(), route).cortheon, undefined, 'no Cortheon object, no field')
}

console.log('turn-stats: ok')
