import assert from 'node:assert/strict'
import type { Settings, Thread } from '../types'

const globals = globalThis as Record<string, unknown>
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = () => {}
globals.removeEventListener = () => {}
globals.window = globals
const { api } = await import('../ipc')
const { useStore } = await import('../store')

async function main() {
  const before = useStore.getState()
  const originalConfigure = api.configureThread
  const originalSave = before.saveSettings
  const provider = { id: 'local', name: 'Local', kind: 'local', baseUrl: 'http://127.0.0.1:1234/v1', apiKey: '', models: ['old', 'new'] }
  const thread = { id: 'selection-fixture', providerId: 'local', providerAccountId: null, model: 'old' } as Thread
  const settings = { providers: [provider], activeProviderId: 'local', activeProviderAccountId: null, activeModel: 'old' } as Settings
  try {
    useStore.setState({ settings, threads: [thread], saveSettings: async () => {} })
    api.configureThread = async () => { throw new Error('Fixture save failed') }
    await assert.rejects(() => useStore.getState().setThreadProvider(thread.id, 'local', 'new', null), /Fixture save failed/)
    assert.equal(useStore.getState().threads[0].model, 'old', 'rejected selection must not present an unpersisted model as active')
    api.configureThread = async () => {}
    await useStore.getState().setThreadProvider(thread.id, 'local', 'new', null)
    assert.equal(useStore.getState().threads[0].model, 'new')
  } finally { api.configureThread = originalConfigure; useStore.setState({ settings: before.settings, threads: before.threads, saveSettings: originalSave }) }
  console.log('Thread model selection persistence checks passed.')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
