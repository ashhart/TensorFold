import assert from 'node:assert/strict'
import { effortDisplayLabel, estimateComposerTokens, formatContextTokens, modelContextWindowTokens, providerEffortOptions, providerFamily, providerIconName, providerModelCapabilities, providerSupportsFastMode } from './provider-model-metadata.ts'

const codex: any = { id: 'codex-cli', name: 'Codex CLI OAuth', transport: 'codex-cli', kind: 'cloud', models: ['gpt-5.5'] }
const openai: any = { id: 'openai', name: 'OpenAI', source: 'openai', kind: 'cloud', models: ['gpt-4o'] }
const claude: any = { id: 'claude-cli', name: 'Claude CLI OAuth', transport: 'claude-cli', kind: 'cloud', models: ['sonnet'] }
const gemini: any = { id: 'gemini', name: 'Gemini', source: 'gemini', kind: 'cloud', models: ['gemini-2.5-pro'] }
const grok: any = { id: 'grok', name: 'Grok', kind: 'cloud', models: ['grok-build'] }
const cursor: any = { id: 'cursor', name: 'Cursor Agent', kind: 'cloud', models: ['gpt-5.3-codex'] }
const opencode: any = { id: 'opencode', name: 'OpenCode', kind: 'cloud', models: ['openai/gpt-5'] }
const kilo: any = { id: 'kilo', name: 'Kilo', kind: 'cloud', models: ['kilo/kilo-auto/free'] }
const pi: any = { id: 'pi', name: 'Pi', kind: 'cloud', models: [] }

assert.equal(providerFamily(codex, 'gpt-5.5'), 'openai')
assert.equal(providerIconName(codex, 'gpt-5.5'), 'openai')
assert.equal(providerFamily(claude, 'sonnet'), 'claude')
assert.equal(providerIconName(gemini, 'gemini-2.5-pro'), 'spark')
assert.equal(providerIconName(grok, 'grok-build'), 'bolt')
assert.equal(providerFamily(cursor, 'gpt-5.3-codex'), 'cursor')
assert.equal(providerFamily(opencode, 'openai/gpt-5'), 'opencode')
assert.equal(providerFamily(kilo, 'openai/gpt-5'), 'kilo')
assert.equal(providerFamily(pi, null), 'pi')
assert.equal(providerIconName(opencode, 'openai/gpt-5'), 'terminal')
assert.equal(providerIconName(kilo, 'kilo/kilo-auto/free'), 'layers')
assert.equal(providerIconName(pi, null), 'cpu')

assert.deepEqual(providerEffortOptions(codex, 'gpt-5.5').map((option) => option.value), ['low', 'medium', 'high', 'xhigh'])
assert.equal(providerEffortOptions(codex, 'gpt-5.5').find((option) => option.value === 'medium')?.isDefault, true)
assert.equal(providerEffortOptions(codex, 'gpt-5.4').find((option) => option.value === 'high')?.isDefault, true)
assert.equal(providerSupportsFastMode(codex, 'gpt-5.5'), true)
assert.equal(providerSupportsFastMode(openai, 'gpt-4o'), false)
assert.equal(effortDisplayLabel(codex, 'gpt-5.5', 'xhigh'), 'Extra High')
assert.equal(effortDisplayLabel(openai, 'gpt-4o', 'high'), 'High')
assert.equal(providerModelCapabilities(claude, 'claude-opus-4-8').effortOptions.some((option) => option.value === 'ultrathink'), true)
assert.equal(providerEffortOptions(claude, 'claude-opus-4-8').some((option) => option.value === 'ultrathink'), true)
assert.equal(providerEffortOptions(claude, 'claude-opus-4-8').find((option) => option.value === 'xhigh')?.runtime, 'native')
assert.equal(providerEffortOptions(claude, 'claude-opus-4-8').find((option) => option.value === 'max')?.runtime, 'native')
assert.deepEqual(providerModelCapabilities(claude, 'claude-opus-4-8').contextWindowOptions.map((option) => option.value), ['200k', '1m'])
assert.equal(effortDisplayLabel(gemini, 'gemini-3-pro', 'high'), 'High')
assert.equal(effortDisplayLabel(gemini, 'gemini-2.5-pro', 'dynamic'), 'Dynamic')
assert.equal(effortDisplayLabel(gemini, 'gemini-2.5-pro', '512'), '512 Tokens')
assert.equal(effortDisplayLabel(grok, 'grok-build', 'none'), 'None')
assert.deepEqual(providerEffortOptions(cursor, 'gpt-5.3-codex').map((option) => option.value), ['low', 'medium', 'high', 'xhigh'])
assert.deepEqual(providerEffortOptions(cursor, 'claude-opus-4-6').map((option) => option.value), ['low', 'medium', 'high', 'max'])
assert.equal(providerSupportsFastMode(cursor, 'gpt-5.3-codex'), true)
assert.deepEqual(providerModelCapabilities(cursor, 'gpt-5.5-high').contextWindowOptions.map((option) => option.value), ['1m'])
assert.deepEqual(providerEffortOptions(opencode, 'openai/gpt-5'), [])
assert.deepEqual(providerEffortOptions(kilo, 'kilo/kilo-auto/free'), [])
assert.deepEqual(providerEffortOptions(pi, null), [])

assert.equal(modelContextWindowTokens(codex, 'gpt-5.5'), 205_000)
assert.equal(modelContextWindowTokens(claude, 'claude-opus-4-8'), 200_000)
assert.equal(modelContextWindowTokens(gemini, 'gemini-2.5-pro'), 1_000_000)
assert.equal(modelContextWindowTokens(grok, 'grok-build'), 256_000)
assert.equal(modelContextWindowTokens({ id: 'local', name: 'Local', kind: 'local', models: [] } as any, 'Qwen3.6 27B'), 205_000)

assert.equal(formatContextTokens(2_100), '2.1k')
assert.equal(formatContextTokens(205_000), '205k')
assert.equal(formatContextTokens(1_000_000), '1M')
assert.equal(estimateComposerTokens('', []), 2_100)

// Live steering (VST-033): a Guide cue lands mid-turn for the native loop and the
// steer-capable delegated engines; the remaining delegated engines apply it only after
// the current engine turn — the composer labels the difference honestly. The flag keys on
// TRANSPORT: an API-routed gemini rides the native loop (steers between steps), while the
// gemini-cli transport cannot take mid-turn input.
assert.equal(providerModelCapabilities(codex, 'gpt-5.5').liveSteering, true, 'codex app-server steers via turn/steer')
assert.equal(providerModelCapabilities(claude, 'claude-opus-4-8').liveSteering, true, 'claude steers via stream-json stdin')
assert.equal(providerModelCapabilities({ ...pi, transport: 'pi-cli' }, null).liveSteering, true, 'pi steers via rpc steer')
assert.equal(providerModelCapabilities({ id: 'api', name: 'OpenAI', kind: 'cloud', models: [] } as any, 'gpt-5').liveSteering, true, 'native loop applies cues between steps')
assert.equal(providerModelCapabilities(gemini, 'gemini-2.5-pro').liveSteering, true, 'API gemini rides the native loop')
assert.equal(providerModelCapabilities({ ...cursor, transport: 'cursor-cli' }, 'gpt-5.3-codex').liveSteering, false)
assert.equal(providerModelCapabilities({ ...gemini, transport: 'gemini-cli' }, 'gemini-3-pro').liveSteering, false)
assert.equal(providerModelCapabilities({ ...opencode, transport: 'opencode' }, 'openai/gpt-5').liveSteering, false)
assert.equal(providerModelCapabilities({ ...kilo, transport: 'kilo' }, 'kilo/kilo-auto/free').liveSteering, false)

console.log('provider-model-metadata tests passed')
