// Run: ./node_modules/.bin/jiti electron/main/host/context-limit.test.ts
// Task #41: model-aware effective context limit + multi-model room floor.
// Tests the pure helpers so we never have to stand up the Electron store.
import assert from 'node:assert'
import type { Provider } from '../../agent/types'
import { computeContextLimit, roomContextLimit } from './context-limit.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const cloud = (models: string[] = []): Provider => ({
  id: 'p-cloud', name: 'Cloud', baseUrl: 'https://api.example.com', apiKey: 'k', models, kind: 'cloud',
})
const local = (models: string[] = []): Provider => ({
  id: 'p-local', name: 'Local', baseUrl: 'http://localhost:11434', apiKey: '', models, kind: 'local',
})

// Defaults that mirror store.settings fallbacks.
const D = { cloudLimit: 128_000, localLimit: 24_000 }

console.log('context-limit tests:')

// 1) Cloud Sonnet, default 128k cloud setting: min(128k, 85% of 200k=170k) = 128k
{
  const limit = computeContextLimit(cloud(), 'claude-3-5-sonnet-20241022', D)
  assert.strictEqual(limit, 128_000, 'default cloud setting still caps Sonnet at 128k')
  ok('cloud Sonnet with default 128k setting -> 128k (backward compatible)')
}

// 2) Cloud Sonnet, user raises the cloud threshold to 256k: scales up to the
//    model's recommended 170k (85% of 200k), not the unreachable 256k.
{
  const limit = computeContextLimit(cloud(), 'claude-3-5-sonnet-20241022', { cloudLimit: 256_000, localLimit: 24_000 })
  assert.strictEqual(limit, 170_000, '256k setting scales Sonnet up to its 170k recommended trigger')
  ok('cloud Sonnet with 256k setting -> 170k (model-aware scaling)')
}

// 3) Cloud GPT-4o, 256k setting: 85% of 128k = 108.8k
{
  const limit = computeContextLimit(cloud(), 'gpt-4o', { cloudLimit: 256_000, localLimit: 24_000 })
  assert.strictEqual(limit, 108_800, 'GPT-4o scales to 108.8k')
  ok('cloud GPT-4o with 256k setting -> 108.8k')
}

// 4) Explicit override wins, but is still capped by the model window.
{
  const within = computeContextLimit(cloud(), 'claude-3-5-sonnet-20241022', { ...D, override: 150_000 })
  assert.strictEqual(within, 150_000, 'override below the model trigger is honoured verbatim')
  const above = computeContextLimit(cloud(), 'claude-3-5-sonnet-20241022', { ...D, override: 999_000 })
  assert.strictEqual(above, 170_000, 'override above the model trigger clamps to 170k')
  ok('contextLimitOverride is honoured and clamped to the model window')
}

// 5) Cloud unknown model: no real window -> keep the configured cloud setting.
{
  const limit = computeContextLimit(cloud(), 'totally-unknown-model', D)
  assert.strictEqual(limit, 128_000, 'unknown cloud model keeps the configured 128k')
  ok('cloud unknown model -> configured cloud limit (no crash)')
}

// 6) Local Haiku-style: min(local 24k default, name/family window) = 24k
{
  const limit = computeContextLimit(local(), 'llama3.1:8b', D)
  assert.strictEqual(limit, 24_000, 'local llama3.1 capped by 24k local default')
  ok('local model with default 24k local setting -> 24k')
}

// 7) Local small model: 4k window pulls the limit below the local default.
{
  const limit = computeContextLimit(local(), 'llama-2-7b', D)
  // min(cloud 128k, local 24k, recommended 4k) = 4k
  assert.strictEqual(limit, 4_000, 'local llama2 4k window wins')
  ok('local Llama-2 4k window pulls the trigger down to 4k')
}

// 8) Local unknown model: falls back to inference / local default, no crash.
{
  const limit = computeContextLimit(local(), 'my-weird-local-model', D)
  assert.strictEqual(limit, 24_000, 'unknown local model keeps prior 24k inference fallback')
  ok('local unknown model -> 24k fallback (backward compatible)')
}

// 8b) Local unknown model + a large override: must still clamp to the local safety net, or the
//     transcript overflows a tiny local context window.
{
  const limit = computeContextLimit(local(), 'my-weird-local-model', { ...D, override: 128_000 })
  assert.strictEqual(limit, 24_000, 'override on an unknown local model clamps to the 24k local net')
  ok('local override clamps to the local safety net (no overflow)')
}

// --- roomContextLimit: multi-model floor ---

// 9) Single Sonnet seat -> its trigger (with a raised cloud setting)
{
  const limit = roomContextLimit([{ provider: cloud(), model: 'claude-3-5-sonnet-20241022' }], { cloudLimit: 256_000, localLimit: 24_000 })
  assert.strictEqual(limit, 170_000, 'single Sonnet seat -> 170k')
  ok('roomContextLimit single Sonnet seat -> 170k')
}

// 10) Sonnet + GPT-4o -> the lower (108.8k) floor
{
  const limit = roomContextLimit([
    { provider: cloud(), model: 'claude-3-5-sonnet-20241022' },
    { provider: cloud(), model: 'gpt-4o' },
  ], { cloudLimit: 256_000, localLimit: 24_000 })
  assert.strictEqual(limit, 108_800, 'mixed room floors to the smaller window')
  ok('roomContextLimit Sonnet + GPT-4o -> 108.8k (lowest wins)')
}

// 11) Order independence
{
  const a = roomContextLimit([
    { provider: cloud(), model: 'gpt-4o' },
    { provider: cloud(), model: 'claude-3-5-sonnet-20241022' },
  ], { cloudLimit: 256_000, localLimit: 24_000 })
  assert.strictEqual(a, 108_800, 'order does not matter')
  ok('roomContextLimit is order-independent')
}

// 12) Empty room -> 0 (no compaction floor)
{
  assert.strictEqual(roomContextLimit([], D), 0, 'empty room -> 0')
  ok('roomContextLimit empty -> 0')
}

console.log(`\n${passed} checks passed.`)
