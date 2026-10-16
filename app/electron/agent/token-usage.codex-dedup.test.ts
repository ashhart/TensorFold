// Run: jiti electron/agent/token-usage.codex-dedup.test.ts
// LOW (C-provider-transport): codexEvents() deduped token_count events by raw
// timestamp string GLOBALLY across every rollout file, so two CONCURRENT codex
// sessions (two rollout-*.jsonl files) that each emit a token_count at the same
// millisecond would have the second silently dropped -- undercounting total token
// usage. Dedup must be scoped PER FILE (one rollout = one session): a repeated
// timestamp within a single file is a re-read duplicate and should still collapse.
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { collectTokenUsage } from './token-usage'

let ok = 0

function rollout(model: string, ts: string, total: number): string {
  const turnContext = JSON.stringify({ type: 'turn_context', payload: { model } })
  const tokenCount = JSON.stringify({
    type: 'event_msg',
    timestamp: ts,
    payload: { type: 'token_count', info: { last_token_usage: { total_tokens: total } } },
  })
  return [turnContext, tokenCount, ''].join('\n')
}

async function main() {
  const cacheDir = await fs.mkdtemp(path.join(os.tmpdir(), 'token-usage-cache-'))
  const codexRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'token-usage-codex-'))
  const prevHome = process.env.HOME
  process.env.HOME = codexRoot

  try {
    const sessionDir = path.join(codexRoot, '.codex', 'sessions', '2026', '06', '30')
    await fs.mkdir(sessionDir, { recursive: true })

    // Two distinct concurrent sessions, each a token_count at the SAME millisecond.
    const sharedTs = '2026-06-30T10:00:00.000Z'
    await fs.writeFile(path.join(sessionDir, 'rollout-sessA.jsonl'), rollout('gpt-5-codex', sharedTs, 100), 'utf8')
    await fs.writeFile(path.join(sessionDir, 'rollout-sessB.jsonl'), rollout('gpt-5-codex', sharedTs, 250), 'utf8')

    const snapshot = await collectTokenUsage(cacheDir)
    const model = snapshot.windows.all.models.find((m) => m.model === 'gpt-5-codex')
    assert.ok(model, 'expected a gpt-5-codex model bucket')
    // Both events must count: 100 + 250 = 350. A global timestamp dedup would drop the
    // second and yield only 100 (or 250).
    assert.equal(model!.total, 350, 'both concurrent-session token_count events at the same ms are counted')
    ok++

    // Within a SINGLE file, a repeated timestamp is a duplicate re-read and must still
    // collapse to one event.
    const dupFile = path.join(sessionDir, 'rollout-dup.jsonl')
    const tc = JSON.stringify({ type: 'turn_context', payload: { model: 'gpt-5-codex' } })
    const dupTs = '2026-06-30T11:00:00.000Z'
    const ev = JSON.stringify({
      type: 'event_msg',
      timestamp: dupTs,
      payload: { type: 'token_count', info: { last_token_usage: { total_tokens: 70 } } },
    })
    await fs.rm(path.join(sessionDir, 'rollout-sessA.jsonl'))
    await fs.rm(path.join(sessionDir, 'rollout-sessB.jsonl'))
    await fs.writeFile(dupFile, [tc, ev, ev, ''].join('\n'), 'utf8')
    const snap2 = await collectTokenUsage(cacheDir)
    const model2 = snap2.windows.all.models.find((m) => m.model === 'gpt-5-codex')
    assert.ok(model2, 'expected a gpt-5-codex model bucket (dup file)')
    assert.equal(model2!.total, 70, 'a repeated timestamp within one file collapses to one event')
    ok++

    console.log(`token-usage.codex-dedup.test.ts: all ${ok} assertions passed`)
  } finally {
    if (prevHome === undefined) delete process.env.HOME
    else process.env.HOME = prevHome
    await fs.rm(cacheDir, { recursive: true, force: true })
    await fs.rm(codexRoot, { recursive: true, force: true })
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
