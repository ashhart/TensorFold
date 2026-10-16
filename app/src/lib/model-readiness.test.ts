import assert from 'node:assert/strict'
import { modelForReadiness } from '../../shared/model-readiness'
import { createMockApi } from './mock/api'
import { mockState } from './mock/state'
import type { Provider } from './types'

async function main() {
  const api = createMockApi()
  const before = mockState.settings
  const repo = 'mlx-community/Qwen3.5-2B-MLX-4bit'
  try {
    mockState.settings = { ...before, activeProviderId: 'tensorfold', activeModel: 'missing-global-model', providers: [{ id: 'tensorfold', name: 'TensorFold', source: 'tensorfold', kind: 'local', models: [repo], baseUrl: '', apiKey: '' } as Provider] }
    assert.equal((await api.getModelReadiness({ providerId: 'tensorfold' })).ready, false)
    assert.equal((await api.getModelReadiness({ providerId: 'tensorfold', model: repo })).ready, true, 'thread selection readiness must not inspect an unrelated global model')
    assert.equal(modelForReadiness({ providerId: 'tensorfold', model: null }, 'tensorfold', 'old'), null, 'explicit null is not an inherited model')
    assert.equal(modelForReadiness({ providerId: 'other' }, 'tensorfold', 'old'), null)
    assert.equal(modelForReadiness(undefined, 'tensorfold', 'old'), 'old', 'older clients inherit the active route')
    assert.throws(() => modelForReadiness({ model: 42 as any }, null, null), /Invalid model/)
  } finally { mockState.settings = before }
  console.log('Selected-model readiness checks passed.')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
