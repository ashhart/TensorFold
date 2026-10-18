import assert from 'node:assert/strict'
import { effectiveContextLimit, roomContextLimit } from './context-limit'
import { buildContextPressureSnapshot } from '../../agent/context-pressure'
import { clearEngineContexts, recordEngineContext } from '../../agent/engine/context'
import { builtinEngineProvider } from '../../agent/engine/provider'
import { defaultSettings } from '../../store'
import { store } from '../state'
const original = store.settings
const model = 'example/profiled'
const provider = { ...builtinEngineProvider(), models: [model] }
try {
  for (const [contextTokens, maxOutputTokens, served] of [[8192, 2048, null], [1024, 256, null], [8192, 2048, 65536], [8192, 2048, 4096]] as const) {
    clearEngineContexts()
    if (served) recordEngineContext(model, served)
    const profile = { id: 'small', name: 'Small', contextTokens, maxOutputTokens, temperature: 0, topP: 1, topK: 0, minP: 0, speculativeDecoding: 'off' as const }
    store.settings = { ...defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: model, engineModelProfiles: [{ model, revision: 1, activeProfileId: profile.id, profiles: [profile] }] }
    const allowed = Math.min(contextTokens, served ?? contextTokens)
    const host = effectiveContextLimit(provider, model)
    const pressure = buildContextPressureSnapshot(store.settings, []).activeProfile!
    assert.ok(host + maxOutputTokens <= allowed)
    assert.equal(pressure.nominalContextTokens, allowed)
    assert.ok(pressure.recommendedWorkingTokens + maxOutputTokens <= allowed)
    assert.equal(roomContextLimit([{ provider, model }], { cloudLimit: 128000, localLimit: 24000, engineModelProfiles: store.settings.engineModelProfiles }), host)
    const cloud = { ...provider, id: 'external', source: 'custom' as const, kind: 'cloud' as const, baseUrl: 'https://example.test/v1' }
    assert.notEqual(effectiveContextLimit(cloud, model), host, 'a matching external model ID does not acquire built-in profiles')
  }
  console.log('Cold and restarted per-model context, reply reservation and mixed-room floor passed')
} finally { store.settings = original; clearEngineContexts() }
