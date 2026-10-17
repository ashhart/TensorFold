import assert from 'node:assert/strict'
import { clearHostedContexts, hostedContextWindow, recordHostedContext, recordHostedContexts } from './hosted-context'
import { computeContextLimit } from '../main/host/context-limit'
import { buildContextPressureSnapshot } from './context-pressure'
import { recommendedCompactionThreshold } from './model-context-windows'
import { defaultSettings } from '../store'
import type { Provider, Thread } from './types'

// A recorded window reads back; the key trims, and a missing or invalid report never erases one.
clearHostedContexts()
recordHostedContext('box', 'model-a', 32768)
assert.equal(hostedContextWindow('box', 'model-a'), 32768)
assert.equal(hostedContextWindow('box', ' model-a '), 32768)
recordHostedContext('box', 'model-a', 0)
recordHostedContext('box', 'model-a', 1.5)
recordHostedContext('box', 'model-a', 'x' as unknown as number)
recordHostedContext('box', '', 4096)
recordHostedContext('', 'model-a', 4096)
assert.equal(hostedContextWindow('box', 'model-a'), 32768)
assert.equal(hostedContextWindow('box', 'model-b'), null)
assert.equal(hostedContextWindow('other', 'model-a'), null)
assert.equal(hostedContextWindow(null, 'model-a'), null)
recordHostedContexts(null, ['x'], { x: 1 })
recordHostedContexts('box', ['b', 'c'], { b: 4096, c: 8192, d: 4096 })
assert.equal(hostedContextWindow('box', 'b'), 4096)
assert.equal(hostedContextWindow('box', 'c'), 8192)
assert.equal(hostedContextWindow('box', 'd'), null, 'a window for a model the list omitted is not taken')

// The compaction trigger follows the window the provider's endpoint reported, in vLLM's
// max_model_len form, ahead of the name-based guesses; only an explicit override caps it.
clearHostedContexts()
const box: Provider = { id: 'box', name: 'vLLM box', kind: 'local', baseUrl: 'http://127.0.0.1:8000/v1', apiKey: '', models: ['qwen3.8-27b'] }
const inputs = { cloudLimit: 128_000, localLimit: 24_000 }
const guessed = computeContextLimit(box, 'qwen3.8-27b', inputs)
recordHostedContext('box', 'qwen3.8-27b', 131_072)
const reported = computeContextLimit(box, 'qwen3.8-27b', inputs)
assert.equal(reported, recommendedCompactionThreshold(131_072))
assert.notEqual(reported, guessed, 'the reported window replaces the guess')
assert.equal(computeContextLimit(box, 'qwen3.8-27b', { ...inputs, override: 40_000 }), Math.min(40_000, recommendedCompactionThreshold(131_072)))
assert.equal(computeContextLimit(box, 'unlisted-model', inputs), guessed, 'a model the endpoint said nothing about keeps the guess')

// A hosted box on the LAN counts as a cloud provider and is honored the same way.
const lab: Provider = { ...box, id: 'lab', kind: 'cloud', baseUrl: 'http://192.168.7.41:8000/v1' }
recordHostedContext('lab', 'qwen3.8-27b', 131_072)
assert.equal(computeContextLimit(lab, 'qwen3.8-27b', inputs), recommendedCompactionThreshold(131_072))

// The context pressure registry names the endpoint as the window's source, locally and for a LAN host.
clearHostedContexts()
const settings = { ...defaultSettings(), providers: [box, lab], activeProviderId: 'box', activeModel: 'qwen3.8-27b' }
const before = buildContextPressureSnapshot(settings as Parameters<typeof buildContextPressureSnapshot>[0], []).registry.find((row) => row.id === 'box:qwen3.8-27b')
assert.ok(before, 'the provider lands in the registry without a report')
assert.notEqual(before?.source, 'endpoint')
recordHostedContext('box', 'qwen3.8-27b', 131_072)
recordHostedContext('lab', 'qwen3.8-27b', 131_072)
const local = buildContextPressureSnapshot(settings as Parameters<typeof buildContextPressureSnapshot>[0], []).registry.find((row) => row.id === 'box:qwen3.8-27b')
assert.equal(local?.source, 'endpoint')
assert.equal(local?.nominalContextTokens, 131_072)
assert.equal(local?.recommendedWorkingTokens, recommendedCompactionThreshold(131_072))
const lanSettings = { ...settings, activeProviderId: 'lab' }
const lanThread = {
  id: 't1', title: 'lab thread', providerId: 'lab', model: 'qwen3.8-27b',
  apiMessages: [{ role: 'user', content: 'hello there' }], updatedAt: 1,
} as unknown as Thread
const lanRow = buildContextPressureSnapshot(lanSettings as Parameters<typeof buildContextPressureSnapshot>[0], [lanThread]).threads[0]
assert.ok(lanRow, 'the LAN thread is pressure-scored')
assert.equal(lanRow?.kind, 'cloud')
assert.equal(lanRow?.contextWindowTokens, 131_072)

clearHostedContexts()
console.log('hosted-context: provider-reported windows (vLLM max_model_len) drive the compaction limits and the pressure registry')
