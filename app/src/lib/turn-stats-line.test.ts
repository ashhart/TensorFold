// Run: npx jiti src/lib/turn-stats-line.test.ts
import assert from 'node:assert/strict'
import { repoDisplayName, statsModelLabel, turnStatsSegments, turnStatsTitle } from './turn-stats-line'
import type { Provider } from './types'

const engine = { id: 'tensorfold', source: 'tensorfold', transport: 'chat-completions' } as Provider
const codex = { id: 'codex', transport: 'codex-cli' } as Provider

assert.equal(repoDisplayName('TensorFold/Qwen3.8-27B-MLX-4bit'), 'Qwen3.8 27B')
assert.equal(repoDisplayName('mlx-community/Qwen3.5-2B-MLX-4bit'), 'Qwen3.5 2B')
assert.equal(repoDisplayName('plain-model'), 'plain model')
assert.equal(repoDisplayName('local/' + 'a'.repeat(64)), 'Local checkpoint', 'a removed library model retains a readable fallback instead of its opaque identity')

const titles = [{ repo: 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit', title: 'Nemotron 3.5 Lightning' }]
assert.equal(statsModelLabel('TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit', engine, titles), 'Nemotron 3.5 Lightning')
assert.equal(statsModelLabel('TensorFold/Qwen3.8-27B-MLX-4bit', engine, titles), 'Qwen3.8 27B')
assert.equal(statsModelLabel('gpt-5.5', codex), 'GPT-5.5')

assert.deepEqual(
  turnStatsSegments({ model: 'm', providerId: 'tensorfold', modelCalls: 3, toolCalls: 2, completionTokens: 900, decodeTokensPerSecond: 211.6, firstTokenMs: 420 }, 'Nemotron 3.5 Lightning'),
  ['Nemotron 3.5 Lightning', '2 tool calls', 'decode 212 tok/s', 'first token 0.4 s'],
)
// Engine timings: decode and prefill both shown, large rates grouped.
const engineStats = { model: 'm', providerId: 'tensorfold', modelCalls: 4, toolCalls: 4, completionTokens: 1184, decodeTokensPerSecond: 1212.4, decodeSeconds: 0.976, prefillTokensPerSecond: 3140.2, promptTokens: 18406, cachedTokens: 16210, firstTokenMs: 380 }
assert.deepEqual(
  turnStatsSegments(engineStats, 'Qwen3.5 2B'),
  ['Qwen3.5 2B', '4 tool calls', 'decode 1,212 tok/s', 'prefill 3,140 tok/s', 'first token 0.4 s'],
)
assert.equal(turnStatsTitle(engineStats), 'Prompt 18,406 tokens\nCached 16,210 tokens\nNew 2,196 tokens\nOutput 1,184 tokens\nDecode 0.98 s')
// A prompt with no cache report shows only its size; nothing known means no title.
assert.equal(turnStatsTitle({ model: 'm', providerId: 'local', modelCalls: 1, toolCalls: 0, promptTokens: 42, decodeSeconds: 2.25 }), 'Prompt 42 tokens\nDecode 2.3 s')
assert.equal(turnStatsTitle({ model: 'm', providerId: 'cloud', modelCalls: 1, toolCalls: 0 }), undefined)
// Nothing measured: only the facts that are known.
assert.deepEqual(turnStatsSegments({ model: 'm', providerId: 'cloud', modelCalls: 1, toolCalls: 0 }, 'GPT-5.5'), ['GPT-5.5'])
assert.deepEqual(turnStatsSegments({ model: 'm', providerId: 'cloud', modelCalls: 1, toolCalls: 1, firstTokenMs: 1250 }, 'GPT-5.5'), ['GPT-5.5', '1 tool call', 'first token 1.3 s'])

console.log('turn-stats-line: ok')
