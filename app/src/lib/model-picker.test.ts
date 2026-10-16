import assert from 'node:assert/strict'
import type { EngineSnapshot, Provider } from './types'
import { filterModelChoices, modelChoiceKey, modelPickerChoices, recentModelChoices } from './model-picker'

const local = { id: 'tensorfold', name: 'TensorFold', source: 'tensorfold', kind: 'local', baseUrl: '', models: ['small', 'large', 'draft'] } as Provider
const cloud = { id: 'cloud', name: 'Connected service', kind: 'cloud', baseUrl: 'https://example.test/v1', models: ['default'], accounts: [
  { id: 'personal', name: 'Personal', models: ['account-only'] },
  { id: 'disabled', name: 'Disabled', enabled: false, models: ['hidden'] },
] } as Provider
const engine = { status: { available: true, state: 'ready', model: 'small', contextTokens: 8192, memoryLimitBytes: 8 * 1024 ** 3 }, installed: [
  { repo: 'small', contextTokens: 262144, memoryFloorBytes: 3 * 1024 ** 3, fitsMemory: true },
  { repo: 'large', memoryFloorBytes: 12 * 1024 ** 3, fitsMemory: true },
  { repo: 'draft', fitsMemory: true, drafter: 'companion', drafterInstalled: false },
], suggestions: [{ repo: 'small', title: 'Small local model' }], performance: null, pulls: [], hubCache: '' } as unknown as EngineSnapshot

const choices = modelPickerChoices([cloud, local], engine)
const libraryModel = 'local/' + 'a'.repeat(64)
const libraryChoices = modelPickerChoices([{ ...local, models: [libraryModel] }], { ...engine, installed: [{ ...engine.installed[0], repo: libraryModel, title: 'My local checkpoint', source: 'library' }] })
assert.equal(libraryChoices[0].label, 'My local checkpoint', 'local library identifiers display their human-readable checkpoint title')
assert.equal(filterModelChoices(libraryChoices, 'my local checkpoint', 'local').length, 1, 'discovered models remain searchable by their displayed title')
const codexChoices = modelPickerChoices([{ ...cloud, accounts: [], transport: 'codex-cli', models: ['codex-cli-default', 'gpt-5.5'] }], null)
assert.equal(codexChoices.find((c) => c.model === 'codex-cli-default')?.label, 'Codex default', 'runtime default must not impersonate a fixed model')
assert.equal(choices[0].provider.id, 'tensorfold', 'built-in choices lead without losing other routes')
const ready = choices.find((c) => c.model === 'small')!
assert.equal(ready.label, 'Small local model')
assert.equal(ready.state, 'Ready')
assert.match(ready.detail, /8\.2k context/i, 'running model uses actual served context, not catalog maximum')
assert.doesNotMatch(ready.detail, /262/)
assert.ok(choices.find((c) => c.model === 'large')?.disabledReason, 'numeric memory evidence beats stale fitsMemory')
assert.match(choices.find((c) => c.model === 'draft')?.disabledReason ?? '', /drafter/)
assert.ok(!choices.some((c) => c.accountId === 'disabled'), 'disabled accounts remain unselectable')

const recents = [
  { providerId: 'cloud', accountId: 'personal', model: 'account-only' },
  { providerId: 'cloud', accountId: 'disabled', model: 'hidden' },
  { providerId: 'cloud', accountId: 'removed', model: 'default' },
  { providerId: 'tensorfold', accountId: null, model: 'large' },
]
assert.deepEqual(recentModelChoices(modelPickerChoices([local], engine), recents, ''), [], 'local-only picker never restores external routes through recents')
assert.deepEqual(recentModelChoices(choices, recents, '').map((c) => c.model), ['account-only'], 'account-specific models are usable without default-catalog membership')
const key = modelChoiceKey('cloud', 'account-only', 'personal')
assert.deepEqual(recentModelChoices(choices, recents, key), [], 'current route is excluded')
assert.deepEqual(filterModelChoices(choices, 'PERSONAL account-only', 'external').map((c) => c.key), [key])
assert.equal(filterModelChoices(choices, 'no matching model', 'all').length, 0)
assert.ok(filterModelChoices(choices, '', 'local').every((c) => c.local), 'local scope uses trust classification')
const remote = { ...local, id: 'remote', source: undefined, baseUrl: 'http://192.0.2.1:1234/v1', models: ['remote-model'] } as Provider
assert.equal(modelPickerChoices([remote], null)[0].local, false, 'LAN endpoints cannot claim on-device execution')
assert.equal(modelPickerChoices([local], null)[0].state, 'Checking engine', 'no snapshot never claims readiness')
assert.doesNotMatch(modelPickerChoices([local], { ...engine, status: { ...engine.status, state: 'stopped' } })[0].detail, /context/, 'unloaded catalog maximum must not impersonate the served window')

console.log('Model picker policy, account and hardware checks passed.')
