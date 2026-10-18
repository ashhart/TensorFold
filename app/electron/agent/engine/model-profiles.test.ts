import assert from 'node:assert/strict'
import { activeEngineModelSettings, assertEngineModelSamplingSupported, defaultEngineModelProfile, engineProfileWorkingLimit, isGreedyOnlyEngineModel, loadEngineModelProfiles, updateEngineModelProfiles, validateEngineModelGeneration, validateEngineModelProfileSet } from '../../../shared/model-profiles'
import { defaultSettings } from '../../store/defaults'
import { modelLaunchOptions } from './launch-options'
import { builtinEngineProvider, setEngineServeHook } from './provider'
import { streamChat } from '../openai'

const generation = { ...defaultEngineModelProfile(8192), contextTokens: 8192, maxOutputTokens: 1024, temperature: 0.7, topP: 0.85, topK: 30, minP: 0.05 }
const row = { model: 'example/model-a', revision: 0, activeProfileId: 'small', profiles: [{ id: 'small', name: 'Small', ...generation }] }
const base = defaultSettings()
const first = updateEngineModelProfiles(base, { ...row, expectedRevision: 0 })
assert.equal(activeEngineModelSettings(first, row.model)?.temperature, 0.7)
assert.equal(activeEngineModelSettings(first, 'example/model-b'), null, 'model identities never inherit another profile')
assert.equal(first.engineModelProfiles?.[0].revision, 1)
assert.equal(base.engineModelProfiles, undefined, 'edits do not mutate the live settings snapshot')
assert.throws(() => updateEngineModelProfiles(first, { ...row, expectedRevision: 0 }), /changed elsewhere/)
const second = updateEngineModelProfiles(first, { ...row, model: 'example/model-b', expectedRevision: 0 })
assert.equal(second.engineModelProfiles?.length, 2)
const defaults = updateEngineModelProfiles(second, { ...row, activeProfileId: null, expectedRevision: 1 })
assert.equal(activeEngineModelSettings(defaults, row.model), null)
assert.equal(defaults.engineModelProfiles?.find((entry) => entry.model === row.model)?.profiles.length, 1, 'restoring defaults retains saved profiles')
assert.equal(activeEngineModelSettings(defaults, 'example/model-b')?.topK, 30)

for (const [field, values] of Object.entries({ contextTokens: [NaN, Infinity, 0, 1023, 8192.5, 1048577], maxOutputTokens: [-1, 0, 8192, Infinity, 2.5], temperature: [-0.1, 2.1, NaN, '0.5'], topP: [0, -1, 1.1, NaN], topK: [-1, 1001, 2.5], minP: [-1, 1.1], speculativeDecoding: ['mtp', '--custom-flag', false] })) {
  for (const value of values) assert.throws(() => validateEngineModelGeneration({ ...generation, [field]: value }), `${field}=${value} is rejected`)
}
assert.throws(() => validateEngineModelProfileSet({ ...row, profiles: [row.profiles[0], { ...row.profiles[0], id: 'different', name: 'small' }] }), /unique/)
assert.throws(() => validateEngineModelProfileSet({ ...row, activeProfileId: 'removed' }), /no longer exists/)
assert.throws(() => validateEngineModelProfileSet({ ...row, profiles: [{ ...row.profiles[0], name: 'bad\nname' }] }), /characters/)
const clean = validateEngineModelProfileSet({ ...row, serverPassword: 'fixture-unaccepted', profiles: [{ ...row.profiles[0], flags: ['--unsafe'], permission: 'auto' }] })
assert.equal((clean as any).serverPassword, undefined)
assert.equal((clean.profiles[0] as any).flags, undefined)
assert.equal((clean.profiles[0] as any).permission, undefined, 'profiles never grant tool permissions')
const malformed = { model: 'example/model-a', profiles: 'damaged' }
const recovery = loadEngineModelProfiles([malformed])
assert.deepEqual(recovery.engineModelProfiles, [])
assert.deepEqual(recovery.engineModelProfilesRecovery, [malformed], 'malformed source data is retained without applying it')
assert.match(recovery.engineModelProfilesProblem ?? '', /retained for recovery/)
for (const modelType of ['glm_moe_dsa']) {
  assert.equal(isGreedyOnlyEngineModel({ modelType }), true)
  assert.throws(() => assertEngineModelSamplingSupported({ modelType }, generation), /greedy/)
  assert.doesNotThrow(() => assertEngineModelSamplingSupported({ modelType }, { ...generation, temperature: 0, topP: 1, topK: 0, minP: 0 }))
}
for (const modelType of ['qwen4_exp', 'glm5_next', 'qwen3_5', 'nemotron_h', 'lfm2_moe']) {
  assert.equal(isGreedyOnlyEngineModel({ modelType }), false)
  assert.doesNotThrow(() => assertEngineModelSamplingSupported({ modelType }, generation))
}
assert.equal(engineProfileWorkingLimit({ ...generation, contextTokens: 1024, maxOutputTokens: 256 }), 768)
assert.equal(engineProfileWorkingLimit({ ...generation, maxOutputTokens: 2048 }, 4096), 2048)

// Exercise the production request boundary without starting an engine or touching a model.
// The GUI's whitelisted launch options never opt into the engines' greedy-only two-Mac mode.
const args = modelLaunchOptions(generation, 32768)
assert.deepEqual(args, ['--context', '8192', '--max-tokens', '1024', '--temperature', '0.7', '--top-p', '0.85', '--top-k', '30', '--min-p', '0.05'])
assert.equal(args.includes('--speed-up'), false)
const originalFetch = globalThis.fetch
const requests: any[] = []
try {
  globalThis.fetch = async (_input, init) => {
    requests.push(JSON.parse(String(init?.body)))
    return new Response('data: {"choices":[{"delta":{"content":"sampled fixture"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', { headers: { 'Content-Type': 'text/event-stream' } })
  }
  for (const modelType of ['qwen4_exp', 'glm5_next']) {
    for (const profile of [generation, null]) {
      setEngineServeHook(async (model) => ({ model, baseUrl: 'http://127.0.0.1:64212/v1', apiKey: 'synthetic-engine-key', generation: profile, greedyOnly: isGreedyOnlyEngineModel({ modelType }) }))
      const output: string[] = []
      for await (const chunk of streamChat({ provider: builtinEngineProvider(), model: `example/${modelType}`, messages: [{ role: 'user', content: 'synthetic sample' }], temperature: 0.45 })) if (chunk.type === 'content') output.push(chunk.text)
      assert.equal(output.join(''), 'sampled fixture')
      const body = requests.at(-1)
      assert.equal(body.temperature, profile ? generation.temperature : 0.45, `${modelType} must preserve both profile and default-chat temperature instead of forcing greedy decoding`)
      if (profile) assert.deepEqual([body.top_p, body.top_k, body.min_p, body.max_tokens], [0.85, 30, 0.05, 1024], `${modelType} profile sampling controls reach the actual chat request`)
    }
  }
  setEngineServeHook(async (model) => ({ model, baseUrl: 'http://127.0.0.1:64212/v1', apiKey: 'synthetic-engine-key', greedyOnly: isGreedyOnlyEngineModel({ modelType: 'glm_moe_dsa' }) }))
  for await (const _chunk of streamChat({ provider: builtinEngineProvider(), model: 'example/glm_moe_dsa', messages: [{ role: 'user', content: 'synthetic greedy sample' }], temperature: 0.45 })) { /* exhaust */ }
  assert.equal(requests.at(-1).temperature, 0, 'GLM-5.3 EXLR8 retains its actual greedy-only request boundary')
} finally {
  globalThis.fetch = originalFetch
  setEngineServeHook(null)
}
console.log('Model profile validation, isolation, revision conflicts, recovery, single-Mac sampling request support, retained EXLR8 restriction and reply reservation passed')
