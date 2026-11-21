import assert from 'node:assert/strict'
import { executeTool } from './tools'
import type { AppAction, Settings, Thread } from './types'

const globals = globalThis as any
globals.localStorage = { getItem: () => null, setItem() {}, removeItem() {} }
globals.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} })
globals.addEventListener = () => {}; globals.removeEventListener = () => {}; globals.window = globals
await (await import('../../src/lib/browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('../../src/lib/ipc')
const { useStore } = await import('../../src/lib/store')
const baseline: Settings = { ...await api.getSettings(), reasoningEffort: 'medium' }
const originalSave = api.saveSettings
let writes: Settings[] = []
function thread(id: string): Thread {
  return { id, title: id, kind: 'chat', cwd: '', providerId: 'synthetic-local', model: 'fixture', status: 'idle',
    createdAt: 1, updatedAt: 1, timeline: [], apiMessages: [] }
}
function reset(action: AppAction) {
  writes = []
  api.saveSettings = async (settings) => { writes.push(settings); return settings }
  useStore.setState({ settings: { ...baseline }, threads: [thread('source'), thread('visible')],
    status: {}, approvals: {}, view: { name: 'thread', id: 'visible' }, notice: null })
  useStore.getState().applyEvent({ type: 'item_add', runId: 'setting-fixture-run', threadId: 'source', item: {
    id: 'setting-proposal', kind: 'tool', tool: 'change_setting', title: 'Change setting', appAction: action,
  } })
  assert.equal(writes.length, 0, 'receiving a proposed setting must not bypass user confirmation')
}
const currentAction = () => useStore.getState().threads.find(t => t.id === 'source')!.timeline[0].appAction!

try {
  const proposal = await executeTool('change_setting', { key: 'reasoningEffort', value: 'ultrathink' }, { cwd: '/workspace' })
  assert.equal(proposal.ok, true)
  assert.ok(proposal.appAction)
  assert.equal(proposal.appAction.state, 'pending')
  reset(proposal.appAction)
  let acknowledge!: () => void
  api.saveSettings = async (settings) => {
    writes.push(settings)
    await new Promise<void>(resolve => { acknowledge = resolve })
    return settings
  }
  const pending = useStore.getState().resolveAppAction('source', 'setting-proposal', 'apply')
  assert.equal(writes.length, 1, 'Apply must persist the canonical effort accepted by the actual tool')
  assert.equal(writes[0].reasoningEffort, 'ultrathink')
  assert.equal(currentAction().state, 'pending', 'the confirmation waits for persistence acknowledgement')
  assert.equal(useStore.getState().settings?.reasoningEffort, 'medium')
  acknowledge(); await pending
  assert.equal(useStore.getState().settings?.reasoningEffort, 'ultrathink')
  assert.equal(currentAction().state, 'applied')
  assert.deepEqual(useStore.getState().view, { name: 'thread', id: 'visible' })
  assert.equal(useStore.getState().threads.find(t => t.id === 'visible')!.timeline.length, 0)

  reset({ ...proposal.appAction, payload: { key: 'reasoningEffort', value: 'ultrathinks' } })
  await useStore.getState().resolveAppAction('source', 'setting-proposal', 'apply')
  assert.equal(writes.length, 0, 'an unsupported plural spelling cannot be saved from a captured action')
  assert.equal(currentAction().state, 'pending')
  assert.equal(useStore.getState().settings?.reasoningEffort, 'medium')

  reset(proposal.appAction)
  api.saveSettings = async (settings) => { writes.push(settings); throw new Error('Synthetic persistence failure') }
  await useStore.getState().resolveAppAction('source', 'setting-proposal', 'apply')
  assert.equal(writes.length, 1)
  assert.equal(currentAction().state, 'pending', 'a failed save must leave the proposal available to retry')
  assert.equal(useStore.getState().settings?.reasoningEffort, 'medium')
  assert.equal(useStore.getState().notice?.kind, 'error')

  reset(proposal.appAction)
  await useStore.getState().resolveAppAction('source', 'setting-proposal', 'dismiss')
  assert.equal(writes.length, 0)
  assert.equal(currentAction().state, 'dismissed')
  assert.equal(useStore.getState().settings?.reasoningEffort, 'medium')
  console.log('Setting confirmation passed: real tool/store, persistence acknowledgement, invalid effort, failed save and dismissal.')
} finally {
  api.saveSettings = originalSave
}
