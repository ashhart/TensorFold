import assert from 'node:assert/strict'
import { buildContextPressureSnapshot } from '../context-pressure'
import { defaultSettings } from '../../store/defaults'
import { computeContextLimit } from '../../main/host/context-limit'
import { clearEngineContexts, engineContextWindow, recordEngineContext } from './context'
import { builtinEngineProvider } from './provider'
import { EngineService } from './service'
import { makeHarness, seedModel } from './test-fixtures/harness'

const MODEL = 'mlx-community/Qwen3.5-2B-MLX-4bit'
const inputs = { cloudLimit: 128_000, localLimit: 24_000 }

async function main() {
  const engine = builtinEngineProvider()
  clearEngineContexts()

  // Before the engine has served the model: the configured local limit, as for any local provider.
  assert.equal(computeContextLimit(engine, MODEL, inputs), 24_000)

  // The service records max_model_len when the engine turns ready (fake engine reports 65536).
  const h = await makeHarness({ FAKE_CONTEXT: '65536' })
  const service = new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 64 * 1024 ** 3, pollMs: 20, emit: () => {} })
  try {
    await seedModel(h.hub, MODEL)
    const status = await service.load(MODEL)
    assert.equal(status.contextTokens, 65536)
    assert.equal(engineContextWindow(MODEL), 65536)
  } finally {
    service.shutdown()
    await h.cleanup()
  }

  // Compaction triggers at 85% of the served window, not at 24k; an override is still clamped to it.
  assert.equal(computeContextLimit(engine, MODEL, inputs), Math.floor(65536 * 0.85))
  assert.equal(computeContextLimit(engine, MODEL, { ...inputs, override: 20_000 }), 20_000)
  assert.equal(computeContextLimit(engine, MODEL, { ...inputs, override: 900_000 }), Math.floor(65536 * 0.85))
  // Other providers and other engine models are unchanged.
  assert.equal(computeContextLimit(engine, 'Org/Other', inputs), 24_000)
  assert.equal(computeContextLimit({ ...engine, id: 'omlx', source: 'omlx' }, MODEL, inputs), 24_000)

  // The context meter's profile for the active engine model uses the same window.
  const settings = { ...defaultSettings(), activeProviderId: engine.id, activeModel: MODEL }
  settings.providers = settings.providers.map((p) => (p.id === engine.id ? { ...p, models: [MODEL] } : p))
  const snapshot = buildContextPressureSnapshot(settings, [])
  assert.equal(snapshot.activeProfile?.nominalContextTokens, 65536)
  assert.equal(snapshot.activeProfile?.recommendedWorkingTokens, Math.floor(65536 * 0.85))
  assert.equal(snapshot.activeProfile?.source, 'engine')

  recordEngineContext(MODEL, null)
  recordEngineContext(MODEL, -1)
  assert.equal(engineContextWindow(MODEL), 65536, 'missing or invalid reports never erase a known window')
  clearEngineContexts()
  assert.equal(buildContextPressureSnapshot(settings, []).activeProfile?.nominalContextTokens, 24_000, 'falls back to the local limit')

  console.log('engine context window tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
