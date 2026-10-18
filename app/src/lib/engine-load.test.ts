import assert from 'node:assert/strict'
const globals = globalThis as Record<string, unknown>
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = () => {}; globals.removeEventListener = () => {}; globals.window = globals
await (await import('./browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('./ipc')
const { useStore } = await import('./store')
const { loadEngineModel } = await import('./engine')
const { defaultSettings } = await import('../../electron/store/defaults')
const before = useStore.getState(), original = { load: api.engineLoad, unload: api.engineUnload, snapshot: api.engineSnapshot }
const settings = { ...defaultSettings(), activeProviderId: 'tensorfold', activeModel: 'Fixture/Working' }
let current = 'Fixture/Working', writes = 0, refusal = false
const loads: string[] = []
const status = (model: string) => ({ model, state: 'ready' }) as any
try {
  api.engineSnapshot = async () => ({ status: status(current) }) as any
  api.engineLoad = async (model) => { loads.push(model); if (model === 'Fixture/Broken') throw new Error('Synthetic load refused'); current = model; return status(model) }
  api.engineUnload = async () => { current = ''; return { state: 'stopped', model: null } as any }
  useStore.setState({ settings, saveSettings: async (value) => { writes++; if (refusal) throw new Error('Synthetic save refused'); useStore.setState({ settings: value }) } })
  await assert.rejects(loadEngineModel('Fixture/Broken'), /Synthetic load refused/)
  assert.equal(useStore.getState().settings?.activeModel, 'Fixture/Working')
  assert.equal(writes, 0, 'a refused load never changes the chosen model')
  refusal = true
  await assert.rejects(loadEngineModel('Fixture/Other'), /Synthetic save refused/)
  assert.equal(current, 'Fixture/Working', 'a durable selection failure restores the previous engine')
  assert.equal(useStore.getState().settings?.activeModel, 'Fixture/Working')
  refusal = false
  await Promise.all([loadEngineModel('Fixture/One'), loadEngineModel('Fixture/Two')])
  assert.equal(current, 'Fixture/Two')
  assert.equal(useStore.getState().settings?.activeModel, 'Fixture/Two', 'rapid selections complete in order')
  assert.deepEqual(loads, ['Fixture/Broken', 'Fixture/Other', 'Fixture/Working', 'Fixture/One', 'Fixture/Two'])
} finally {
  api.engineLoad = original.load; api.engineUnload = original.unload; api.engineSnapshot = original.snapshot
  useStore.setState({ settings: before.settings, saveSettings: before.saveSettings })
}
console.log('Rejected loads, durable selection failure recovery and serialized model selections passed')
