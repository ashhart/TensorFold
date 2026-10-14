// Run: jiti electron/agent/token-usage.test.ts
// LOW #27: codexEvents() must not permanently attribute a token_count event to
// model 'unknown' just because it appears in the rollout file before the first
// successfully-parsed turn_context line. A corrupted/truncated turn_context line
// (plausible mid-write) is silently swallowed today, and any token_count read
// before a LATER, successfully-parsed turn_context in the same file stays stamped
// 'unknown' forever -- skewing per-model token-usage breakdowns.
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { collectTokenUsage } from './token-usage'

async function main() {
  const cacheDir = await fs.mkdtemp(path.join(os.tmpdir(), 'token-usage-cache-'))
  const codexRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'token-usage-codex-'))
  const prevHome = process.env.HOME
  process.env.HOME = codexRoot

  try {
    const sessionDir = path.join(codexRoot, '.codex', 'sessions', '2026', '06', '30')
    await fs.mkdir(sessionDir, { recursive: true })
    const rolloutFile = path.join(sessionDir, 'rollout-test-aaaa.jsonl')

    // Line 1: a truncated/corrupt turn_context line (mid-write race) -- contains
    // "model" so it passes the substring pre-filter, but fails JSON.parse and is
    // silently swallowed by the existing catch block.
    const corruptTurnContext = '{"type":"turn_context","payload":{"model":"gpt-5-cod'
    // Line 2: a token_count event that arrives before any turn_context has been
    // successfully parsed yet.
    const earlyTokenCount = JSON.stringify({
      type: 'event_msg',
      timestamp: '2026-06-30T10:00:00.000Z',
      payload: {
        type: 'token_count',
        info: { last_token_usage: { total_tokens: 111, input_tokens: 100, output_tokens: 11 } },
      },
    })
    // Line 3: a valid, complete turn_context line -- the model becomes known here.
    const validTurnContext = JSON.stringify({ type: 'turn_context', payload: { model: 'gpt-5-codex' } })
    // Line 4: a later token_count event, after the model is known (already worked
    // correctly before this fix).
    const laterTokenCount = JSON.stringify({
      type: 'event_msg',
      timestamp: '2026-06-30T10:05:00.000Z',
      payload: {
        type: 'token_count',
        info: { last_token_usage: { total_tokens: 222, input_tokens: 200, output_tokens: 22 } },
      },
    })

    await fs.writeFile(
      rolloutFile,
      [corruptTurnContext, earlyTokenCount, validTurnContext, laterTokenCount, ''].join('\n'),
      'utf8',
    )

    const snapshot = await collectTokenUsage(cacheDir)
    const allModels = snapshot.windows.all.models

    const unknownModel = allModels.find((m) => m.model === 'unknown')
    assert.equal(
      unknownModel,
      undefined,
      `expected no 'unknown' model bucket once a later turn_context resolves the model, got: ${JSON.stringify(allModels)}`,
    )

    const knownModel = allModels.find((m) => m.model === 'gpt-5-codex')
    assert.ok(knownModel, 'expected a gpt-5-codex model bucket to exist')
    // Both the early (pre-model) and later (post-model) token_count events should
    // be attributed to the resolved model: 111 + 222 = 333 total tokens.
    assert.equal(knownModel!.total, 333, 'both token_count events attributed to the resolved model')

    console.log('token-usage: codexEvents backfills pre-model token_count events once model is known')
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
