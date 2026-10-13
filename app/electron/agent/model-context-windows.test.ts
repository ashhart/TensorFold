// Run: ./node_modules/.bin/jiti electron/agent/model-context-windows.test.ts
// Task #41: model-aware context window scaling. Proves the registry returns real
// windows for known model families, falls back to name inference, and that the
// recommended compaction threshold clamps to a safe [4k, 512k] band at 85%.
import assert from 'node:assert'
import {
  getModelContextWindow,
  recommendedCompactionThreshold,
  recommendedKeepRecent,
} from './model-context-windows.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('model-context-windows tests:')

// 1) Known cloud families return their real windows
{
  assert.strictEqual(getModelContextWindow('claude-3-5-sonnet-20241022'), 200_000, 'sonnet 200k')
  assert.strictEqual(getModelContextWindow('claude-3-opus-20240229'), 200_000, 'opus 200k')
  assert.strictEqual(getModelContextWindow('claude-3-5-haiku-20241022'), 200_000, 'haiku 200k')
  assert.strictEqual(getModelContextWindow('claude-sonnet-4-20250514'), 200_000, 'sonnet 4 200k')
  assert.strictEqual(getModelContextWindow('gpt-5.5'), 205_000, 'gpt-5.5 205k')
  assert.strictEqual(getModelContextWindow('gpt-5.5-extra-high'), 205_000, 'gpt-5.5 alias 205k')
  assert.strictEqual(getModelContextWindow('gpt-4o'), 128_000, 'gpt-4o 128k')
  assert.strictEqual(getModelContextWindow('gpt-4o-mini'), 128_000, 'gpt-4o-mini 128k')
  assert.strictEqual(getModelContextWindow('gpt-4-turbo'), 128_000, 'gpt-4-turbo 128k')
  assert.strictEqual(getModelContextWindow('o1-preview'), 128_000, 'o1 128k')
  assert.strictEqual(getModelContextWindow('gemini-1.5-pro'), 1_000_000, 'gemini 1.5 pro 1M')
  ok('known cloud families return their published windows')
}

// 2) Known local families return their windows
{
  assert.strictEqual(getModelContextWindow('llama-2-7b'), 4_096, 'llama2 4k')
  assert.strictEqual(getModelContextWindow('llama3.1:8b'), 128_000, 'llama 3.1 128k')
  assert.strictEqual(getModelContextWindow('qwen2.5-coder:7b'), 32_768, 'qwen2.5 32k')
  assert.strictEqual(getModelContextWindow('gemma2:9b'), 8_192, 'gemma 8k')
  ok('known local families return their windows')
}

// 3) Case-insensitive matching
{
  assert.strictEqual(getModelContextWindow('CLAUDE-3-5-SONNET'), 200_000, 'uppercase sonnet')
  assert.strictEqual(getModelContextWindow('GPT-4O'), 128_000, 'uppercase gpt-4o')
  ok('matching is case-insensitive')
}

// 4) Name inference fallback for "Nk" pattern when no family matches
{
  assert.strictEqual(getModelContextWindow('my-custom-7b-4k'), 4_000, 'parse 4k from name')
  assert.strictEqual(getModelContextWindow('frankenmodel-32k-instruct'), 32_000, 'parse 32k from name')
  assert.strictEqual(getModelContextWindow('weirdo-8k'), 8_000, 'parse 8k from name')
  ok('falls back to parsing the Nk hint in the model name')
}

// 5) Truly unknown models return null (caller keeps its existing behaviour)
{
  assert.strictEqual(getModelContextWindow('totally-unknown-model'), null, 'unknown -> null')
  assert.strictEqual(getModelContextWindow(''), null, 'empty -> null')
  assert.strictEqual(getModelContextWindow(null), null, 'null -> null')
  assert.strictEqual(getModelContextWindow(undefined), null, 'undefined -> null')
  ok('unknown / empty / nullish models return null')
}

// 6) recommendedCompactionThreshold = 85% of window, clamped to [4k, 512k]
{
  assert.strictEqual(recommendedCompactionThreshold(200_000), 170_000, '200k -> 170k')
  assert.strictEqual(recommendedCompactionThreshold(128_000), 108_800, '128k -> 108.8k')
  assert.strictEqual(recommendedCompactionThreshold(100_000), 85_000, '100k -> 85k')
  // Floor: 85% of 4k = 3.4k, clamps up to 4k
  assert.strictEqual(recommendedCompactionThreshold(4_096), 4_000, '4k window -> 4k floor')
  // Ceiling: 85% of a 1M window = 850k, clamps down to 512k
  assert.strictEqual(recommendedCompactionThreshold(1_000_000), 512_000, '1M -> 512k ceiling')
  assert.strictEqual(recommendedCompactionThreshold(0), 4_000, '0 -> 4k floor')
  assert.strictEqual(recommendedCompactionThreshold(-5), 4_000, 'negative -> 4k floor')
  ok('recommendedCompactionThreshold scales to 85% and clamps to [4k, 512k]')
}

// 7) recommendedKeepRecent scales with window, clamped to [4, 24]
{
  // Unknown/null window -> baseline 6 (preserves the prior flat KEEP_RECENT)
  assert.strictEqual(recommendedKeepRecent(null), 6, 'null window -> 6 baseline')
  assert.strictEqual(recommendedKeepRecent(0), 6, '0 window -> 6 baseline')
  assert.strictEqual(recommendedKeepRecent(-1), 6, 'negative -> 6 baseline')
  // Tiny window (4k llama2) clamps to the floor of 4
  assert.strictEqual(recommendedKeepRecent(4_096), 4, '4k -> floor 4')
  // 32k (qwen2.5, the baseline reference point) -> 6
  assert.strictEqual(recommendedKeepRecent(32_768), 6, '32k -> 6 baseline')
  // 128k (gpt-4o, llama3, mistral) -> 6 + round((128000-32000)/16000) = 6 + 6 = 12
  assert.strictEqual(recommendedKeepRecent(128_000), 12, '128k -> 12')
  // 200k (claude) -> 6 + round(168000/16000) = 6 + 11 = 17
  assert.strictEqual(recommendedKeepRecent(200_000), 17, '200k -> 17')
  // 1M (gemini) -> clamps to ceiling 24
  assert.strictEqual(recommendedKeepRecent(1_000_000), 24, '1M -> ceiling 24')
  // Monotonic: bigger window never keeps fewer
  const six = recommendedKeepRecent(32_000)
  const twelve = recommendedKeepRecent(128_000)
  const seventeen = recommendedKeepRecent(200_000)
  assert.ok(six <= twelve && twelve <= seventeen, `monotonic: ${six} <= ${twelve} <= ${seventeen}`)
  ok('recommendedKeepRecent scales with window, clamped to [4, 24]')
}

// 8) keepRecentForModel via getModelContextWindow round-trips for real models
{
  // A 200k Claude keeps more recent context than a 32k Qwen.
  const claudeKeep = recommendedKeepRecent(getModelContextWindow('claude-sonnet-4-20250514'))
  const qwenKeep = recommendedKeepRecent(getModelContextWindow('qwen2.5-coder:7b'))
  assert.ok(claudeKeep > qwenKeep, `claude (${claudeKeep}) keeps more than qwen (${qwenKeep})`)
  // A 4k Llama 2 stays at the floor.
  const llama2Keep = recommendedKeepRecent(getModelContextWindow('llama-2-7b'))
  assert.strictEqual(llama2Keep, 4, 'llama2 stays at the floor')
  ok('keepRecent scales correctly for real model ids')
}

console.log(`\n${passed} checks passed.`)
