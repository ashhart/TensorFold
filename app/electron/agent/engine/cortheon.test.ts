import assert from 'node:assert/strict'
import { EngineService } from './service'
import { CORTHEON_RELEASE_POLICY } from '../../../shared/release-feature-policy'
import { makeHarness, seedModel } from './test-fixtures/harness'

const MODEL = 'mlx-community/Qwen3.5-2B-MLX-4bit'
for (const supported of [false, true]) {
  const h = await makeHarness(supported ? { FAKE_ENGINE_CORTHEON: '1' } : {})
  const svc = new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 20, emit: () => {} })
  try {
    assert.deepEqual((await svc.status()).cortheon, { supported, enabled: false, active: false, disabledReason: CORTHEON_RELEASE_POLICY.reason }, 'the engine capability remains factual while the desktop release is paused')
    const before = await h.records()
    await assert.rejects(svc.setCortheon(true), /temporarily disabled for beta\.6/)
    assert.deepEqual(await h.records(), before, 'the refused enable does not probe or restart the engine')
    await seedModel(h.hub, MODEL)
    await svc.load(MODEL)
    const status = await svc.setCortheon(false)
    assert.equal(status.state, 'ready')
    assert.equal(status.model, MODEL)
    assert.equal(status.cortheon?.active, false)
    assert.ok((await h.records()).filter((r) => r.kind === 'serve').every((r) => !r.args.includes('--cortheon')))
  } finally { await svc.unload().catch(() => {}); await h.cleanup() }
}
console.log('Cortheon release pause reports raw engine support truthfully and preserves ordinary model serving')
