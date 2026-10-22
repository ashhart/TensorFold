import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFileSync } from 'node:fs'
import { providerWithAccount } from './provider-accounts'
import { normalizeContextLimitChoices } from '../../shared/context-limit-choices'
import { clearHostedContexts, explicitContextCaps, forgetHostedContexts, invalidateHostedContexts, hostedContextIdentity, hostedContextWindow, hostedTrigger, hostedWorkingTokens, recordHostedContext, recordHostedContexts } from './hosted-context'
import { store } from '../main/state'
import { computeContextLimit, effectiveContextLimit, roomContextLimit } from '../main/host/context-limit'
import { buildContextPressureSnapshot } from './context-pressure'
import { recommendedCompactionThreshold } from './model-context-windows'
import { defaultSettings } from '../store'
import type { Provider, Thread } from './types'

// A recorded window reads back; the key trims, and a missing or invalid report never erases one.
const identityOf = (provider: Provider) => hostedContextIdentity(provider)
const boxProvider: Provider = { id: 'box', name: 'vLLM box', kind: 'local', baseUrl: 'http://127.0.0.1:8000/v1', apiKey: '', models: ['qwen3.8-27b'] }
const labProvider: Provider = { ...boxProvider, id: 'lab', kind: 'cloud', baseUrl: 'http://192.168.7.41:8000/v1' }

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
const box = boxProvider
const inputs = { cloudLimit: 128_000, localLimit: 24_000 }
const guessed = computeContextLimit(box, 'qwen3.8-27b', inputs)
recordHostedContext(identityOf(box), 'qwen3.8-27b', 131_072)
const reported = computeContextLimit(box, 'qwen3.8-27b', inputs)
assert.equal(reported, recommendedCompactionThreshold(131_072))
assert.notEqual(reported, guessed, 'the reported window replaces the guess')
assert.equal(computeContextLimit(box, 'qwen3.8-27b', { ...inputs, override: 40_000 }), Math.min(40_000, recommendedCompactionThreshold(131_072)))
assert.equal(computeContextLimit(box, 'unlisted-model', inputs), guessed, 'a model the endpoint said nothing about keeps the guess')

// A hosted box on the LAN counts as a cloud provider and is honored the same way.
const lab = labProvider
recordHostedContext(identityOf(lab), 'qwen3.8-27b', 131_072)
assert.equal(computeContextLimit(lab, 'qwen3.8-27b', inputs), recommendedCompactionThreshold(131_072))

// The context pressure registry names the endpoint as the window's source, locally and for a LAN host.
clearHostedContexts()
const settings = { ...defaultSettings(), providers: [box, lab], activeProviderId: 'box', activeModel: 'qwen3.8-27b' }
const before = buildContextPressureSnapshot(settings as Parameters<typeof buildContextPressureSnapshot>[0], []).registry.find((row) => row.id === 'box:qwen3.8-27b')
assert.ok(before, 'the provider lands in the registry without a report')
assert.notEqual(before?.source, 'endpoint')
recordHostedContext(identityOf(box), 'qwen3.8-27b', 131_072)
recordHostedContext(identityOf(lab), 'qwen3.8-27b', 131_072)
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

// The reported window replaces the guesses; the user's own setting still caps it.
clearHostedContexts()
recordHostedContext(identityOf(box), 'qwen3.8-27b', 131_072)
recordHostedContext(identityOf(lab), 'qwen3.8-27b', 131_072)
assert.equal(
  computeContextLimit(box, 'qwen3.8-27b', { cloudLimit: 128_000, localLimit: 8_000 }),
  8_000,
  'an explicit local setting caps the window the endpoint reported',
)
assert.equal(
  computeContextLimit(box, 'qwen3.8-27b', { cloudLimit: 64_000, localLimit: 24_000 }),
  64_000,
  'an explicit cloud setting caps it too',
)
assert.equal(
  computeContextLimit(box, 'qwen3.8-27b', { cloudLimit: 128_000, localLimit: 24_000 }),
  recommendedCompactionThreshold(131_072),
  'the store defaults are not a choice, so they do not cap a reported window',
)
assert.deepEqual(explicitContextCaps(128_000, 24_000, true), [], 'the defaults contribute no cap')
assert.deepEqual(explicitContextCaps(128_000, 8_000, true), [8_000])
assert.deepEqual(explicitContextCaps(64_000, 8_000, false), [64_000], 'a cloud path does not read the local setting')

// The pressure panel honours the same cap.
const cappedSettings = { ...settings, localContextLimit: 8_000 } as Parameters<typeof buildContextPressureSnapshot>[0]
const cappedRow = buildContextPressureSnapshot(cappedSettings, []).registry.find((row) => row.id === 'box:qwen3.8-27b')
assert.equal(cappedRow?.nominalContextTokens, 131_072, 'the reported window is still what the endpoint serves')
assert.equal(cappedRow?.recommendedWorkingTokens, 8_000, 'an explicit setting caps the working tokens')
const cappedLanSettings = { ...settings, activeProviderId: 'lab', contextLimit: 64_000 } as Parameters<typeof buildContextPressureSnapshot>[0]
const cappedLanThread = buildContextPressureSnapshot(cappedLanSettings, [lanThread]).threads[0]
assert.equal(cappedLanThread?.contextWindowTokens, 131_072, 'the reported window is still what the endpoint serves')
assert.equal(cappedLanThread?.workingLimitTokens, 64_000, 'a LAN host is capped by the explicit cloud setting')

// A small reported capacity is served as it is; the 4k floor is for guesses, not for an endpoint.
clearHostedContexts()
recordHostedContext(identityOf(box), 'tiny', 2_048)
assert.equal(hostedWorkingTokens(2_048), 2_048)
assert.equal(hostedWorkingTokens(0), 0, 'an unusable window has no working tokens')
assert.equal(hostedTrigger(2_048, []), 2_048, 'a 2048-token endpoint cannot serve a 4000-token trigger')
assert.equal(computeContextLimit(box, 'tiny', inputs), 2_048)
assert.equal(hostedTrigger(131_072, []), recommendedCompactionThreshold(131_072), 'a large reported window is untouched')
const tinySettings = { ...settings, providers: [box], activeModel: 'tiny' } as Parameters<typeof buildContextPressureSnapshot>[0]
const tinyRow = buildContextPressureSnapshot(tinySettings, []).registry.find((row) => row.id === 'box:tiny')
assert.equal(tinyRow?.nominalContextTokens, 2_048)
assert.equal(tinyRow?.recommendedWorkingTokens, 2_048, 'the pressure panel does not raise it above the endpoint either')

// A window belongs to the endpoint that reported it.
clearHostedContexts()
recordHostedContext(identityOf(box), 'qwen3.8-27b', 131_072)
const moved = { ...box, baseUrl: 'http://127.0.0.1:9000/v1' }
assert.equal(computeContextLimit(moved, 'qwen3.8-27b', inputs), guessed, 'an edited endpoint does not read back the old endpoint window')
assert.equal(computeContextLimit(box, 'qwen3.8-27b', inputs), recommendedCompactionThreshold(131_072), 'the endpoint that answered still holds its window')
const movedRow = buildContextPressureSnapshot({ ...tinySettings, providers: [moved] } as Parameters<typeof buildContextPressureSnapshot>[0], []).registry.find((row) => row.id === 'box:qwen3.8-27b')
assert.notEqual(movedRow?.source, 'endpoint', 'the pressure panel forgets it too')
// A refresh that fails after the edit leaves the old endpoint's number on the old endpoint, not on the new one.
assert.equal(hostedContextWindow(identityOf(moved), 'qwen3.8-27b'), null, 'a failed refresh never answers for the new endpoint')

forgetHostedContexts(identityOf(box))
assert.equal(hostedContextWindow(identityOf(box), 'qwen3.8-27b'), null, 'a dropped configuration leaves nothing behind')
recordHostedContext(identityOf(box), 'qwen3.8-27b', 131_072)

// Two configurations of one provider do not share a window.
clearHostedContexts()
const firstAccount = box
const secondAccount = { ...box, baseUrl: 'http://127.0.0.1:8001/v1' }
recordHostedContext(identityOf(firstAccount), 'qwen3.8-27b', 131_072)
recordHostedContext(identityOf(secondAccount), 'qwen3.8-27b', 8_192)
assert.equal(computeContextLimit(firstAccount, 'qwen3.8-27b', inputs), recommendedCompactionThreshold(131_072))
assert.equal(computeContextLimit(secondAccount, 'qwen3.8-27b', inputs), recommendedCompactionThreshold(8_192), 'the second configuration keeps its own window')

clearHostedContexts()
console.log('hosted-context: provider-reported windows (vLLM max_model_len) drive the compaction limits and the pressure registry')

test('same-endpoint accounts keep independent limits in runtime and pressure reads', () => {
  clearHostedContexts()
  const provider: Provider = { ...box, accounts: [
    { id: 'first', name: 'First', kind: 'api-key', enabled: true, apiKey: 'fixture-first', createdAt: 1, updatedAt: 1 },
    { id: 'second', name: 'Second', kind: 'api-key', enabled: true, apiKey: 'fixture-second', createdAt: 1, updatedAt: 1 },
    { id: 'shared', name: 'Shared key', kind: 'api-key', enabled: true, apiKey: 'fixture-first', createdAt: 1, updatedAt: 1 },
  ] }
  const first = providerWithAccount(provider, 'first')
  const second = providerWithAccount(provider, 'second')
  const shared = providerWithAccount(provider, 'shared')
  recordHostedContext(identityOf(first), 'qwen3.8-27b', 131_072)
  assert.notEqual(identityOf(first), identityOf(second))
  assert.notEqual(identityOf(first), identityOf(shared), 'different account IDs are isolated even with a shared key')
  assert.match(identityOf(first), /^[a-f0-9]{64}$/)
  assert.ok(!identityOf(first).includes(first.apiKey), 'cache keys contain no raw credential')
  assert.equal(computeContextLimit(second, 'qwen3.8-27b', inputs), guessed, 'failed account refresh keeps its fallback')
  recordHostedContext(identityOf(second), 'qwen3.8-27b', 8_192)
  const accountSettings = { ...settings, providers: [provider], activeProviderAccountId: 'second' }
  const snapshot = buildContextPressureSnapshot(accountSettings, [{
    ...lanThread, providerId: provider.id, providerAccountId: 'first',
  }])
  assert.equal(snapshot.activeProfile?.nominalContextTokens, 8_192)
  assert.equal(snapshot.registry[0]?.nominalContextTokens, 8_192)
  assert.equal(snapshot.threads[0]?.contextWindowTokens, 131_072, 'thread account overrides the active account')
  const defaultAccount = buildContextPressureSnapshot({
    ...accountSettings, providers: [{ ...provider, defaultAccountId: 'first' }], activeProviderAccountId: null,
  }, [{ ...lanThread, providerId: provider.id, providerAccountId: null }])
  assert.equal(defaultAccount.activeProfile?.nominalContextTokens, 131_072, 'the active route follows the provider default')
  assert.notEqual(defaultAccount.threads[0]?.contextWindowTokens, 131_072, 'an explicit base-account thread stays separate')
  assert.equal(computeContextLimit(first, 'qwen3.8-27b', inputs), recommendedCompactionThreshold(131_072))
  assert.equal(computeContextLimit(second, 'qwen3.8-27b', inputs), recommendedCompactionThreshold(8_192))
  const edited = { ...second, apiKey: 'fixture-rotated' }
  assert.equal(hostedContextWindow(identityOf(edited), 'qwen3.8-27b'), null, 'credential edits cannot reuse a report')
  const pendingIdentity = identityOf(first)
  invalidateHostedContexts([provider], [{ ...provider, apiKey: 'fixture-base-edit' }])
  recordHostedContext(pendingIdentity, 'qwen3.8-27b', 262_144)
  assert.equal(hostedContextWindow(identityOf(first), 'qwen3.8-27b'), null, 'late replies and reverted edits stay invalidated')
  recordHostedContext(identityOf(first), 'qwen3.8-27b', 131_072)
  invalidateHostedContexts([provider], [structuredClone(provider)])
  assert.equal(hostedContextWindow(identityOf(first), 'qwen3.8-27b'), 131_072, 'an unchanged save preserves the report')
  clearHostedContexts()
})

test('explicit default-valued caps survive settings save/load and both pressure paths', () => {
  clearHostedContexts()
  recordHostedContext(identityOf(box), 'qwen3.8-27b', 262_144)
  recordHostedContext(identityOf(lab), 'qwen3.8-27b', 262_144)
  const fresh = defaultSettings()
  assert.deepEqual(normalizeContextLimitChoices(fresh), { contextLimitExplicit: false, localContextLimitExplicit: false })
  const small = { ...fresh, localContextLimit: 8_000 }
  Object.assign(small, normalizeContextLimitChoices(small, fresh))
  const chosen = { ...small, localContextLimit: 24_000 }
  Object.assign(chosen, normalizeContextLimitChoices(chosen, small))
  const saved = JSON.parse(JSON.stringify(chosen))
  const loaded = { ...defaultSettings(), ...saved, ...normalizeContextLimitChoices(saved) }
  assert.equal(loaded.localContextLimitExplicit, true)
  assert.equal(normalizeContextLimitChoices(fresh, loaded).localContextLimitExplicit, true, 'stale unrelated saves preserve intent')
  assert.equal(computeContextLimit(box, 'qwen3.8-27b', { ...inputs, ...loaded }), 24_000)
  const localSnapshot = buildContextPressureSnapshot({ ...loaded, providers: [box], activeProviderId: box.id }, [])
  assert.equal(localSnapshot.registry[0]?.recommendedWorkingTokens, 24_000)
  const cloud = { ...loaded, contextLimit: 128_000, contextLimitExplicit: true, providers: [lab], activeProviderId: lab.id }
  assert.equal(computeContextLimit(lab, 'qwen3.8-27b', { ...inputs, ...cloud }), 128_000)
  assert.equal(buildContextPressureSnapshot(cloud, [lanThread]).threads[0]?.workingLimitTokens, 128_000)
  assert.deepEqual(normalizeContextLimitChoices({ contextLimit: 128_000, localContextLimit: 24_000 }), {
    contextLimitExplicit: true, localContextLimitExplicit: true,
  }, 'legacy saved values conservatively remain caps when their intent is unknown')
  assert.equal(computeContextLimit(box, 'qwen3.8-27b', { ...inputs, ...fresh }), recommendedCompactionThreshold(262_144))
  clearHostedContexts()
})

test('actual effective limit resolves requested and active accounts, explicit flags and room floors', () => {
  const previous = store.settings
  clearHostedContexts()
  try {
    const provider: Provider = { ...box, accounts: [
      { id: 'first', name: 'First', kind: 'api-key', apiKey: 'fixture-first', enabled: true, createdAt: 1, updatedAt: 1 },
      { id: 'second', name: 'Second', kind: 'api-key', apiKey: 'fixture-second', enabled: true, createdAt: 1, updatedAt: 1 },
    ] }
    const first = providerWithAccount(provider, 'first'), second = providerWithAccount(provider, 'second')
    recordHostedContext(identityOf(first), 'qwen3.8-27b', 131_072)
    recordHostedContext(identityOf(second), 'qwen3.8-27b', 8_192)
    store.settings = { ...defaultSettings(), activeProviderId: provider.id, activeProviderAccountId: 'first', providers: [provider] }
    assert.equal(effectiveContextLimit(provider, 'qwen3.8-27b'), recommendedCompactionThreshold(131_072))
    assert.equal(effectiveContextLimit(provider, 'qwen3.8-27b', 'second'), recommendedCompactionThreshold(8_192))
    assert.equal(effectiveContextLimit(second, 'qwen3.8-27b'), recommendedCompactionThreshold(8_192))
    assert.equal(effectiveContextLimit(provider, 'qwen3.8-27b', null), guessed, 'explicit base-account routes stay separate')
    store.settings.localContextLimitExplicit = true
    assert.equal(effectiveContextLimit(first, 'qwen3.8-27b'), 24_000)
    assert.equal(roomContextLimit([{ provider: first, model: 'qwen3.8-27b' }], {
      ...inputs, localContextLimitExplicit: true,
    }), 24_000)
  } finally {
    store.settings = previous
    clearHostedContexts()
  }
})

test('threshold controls mark actual default-valued choices as explicit', () => {
  const source = readFileSync(new URL('../../src/components/settings/RuntimeSettings.tsx', import.meta.url), 'utf8')
  const callbacks = [...source.matchAll(/onChange=\{\(e\) => patch\((\{ (?:contextLimit|localContextLimit):[^}]+\})\)\}/g)]
  assert.equal(callbacks.length, 2)
  for (const callback of callbacks) {
    const local = callback[1].includes('localContextLimit:')
    const key = local ? 'localContextLimit' : 'contextLimit'
    let choice: Record<string, unknown> = {}
    new Function('e', 'patch', `patch(${callback[1]})`)({ target: { value: local ? '24000' : '128000' } }, (next: Record<string, unknown>) => { choice = next })
    assert.equal(choice[key], local ? 24_000 : 128_000)
    assert.equal(choice[`${key}Explicit`], true, 'a choice equal to the default still records intent')
  }
})
