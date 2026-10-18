import assert from 'node:assert/strict'

const globals = globalThis as any
globals.localStorage = { getItem: () => null, setItem() {}, removeItem() {} }
globals.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} })
globals.addEventListener = () => {}; globals.removeEventListener = () => {}; globals.window = globals
await (await import('../../browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('../../ipc')
const { useStore } = await import('../index')
const settings = await api.getSettings()
const original = { cancelRun: api.cancelRun, cancelRemoteOrchestratorThread: api.cancelRemoteOrchestratorThread }
function reset(hostId?: string) {
  useStore.setState({ settings: { ...settings, remoteOrchestrators: [{ id: 'synthetic-host', name: 'Synthetic', enabled: true, baseUrl: 'https://synthetic.example.invalid' } as any] }, projects: [], tickets: [], threads: [{ id: 'active', title: 'Synthetic', kind: 'chat', cwd: '', providerId: null, model: null, status: 'running', createdAt: 1, updatedAt: 1, timeline: [], apiMessages: [], ...(hostId ? { orchestratorHostId: hostId } : {}) } as any], status: { active: 'tool' }, approvals: { active: [{ callId: 'approval', itemId: 'item' }] }, view: { name: 'thread', id: 'active' } })
}
try {
  let acknowledge!: () => void
  let localCalls = 0
  api.cancelRun = async () => { localCalls++; await new Promise<void>(resolve => { acknowledge = resolve }) }
  reset()
  const pending = useStore.getState().cancel('active')
  assert.equal(useStore.getState().status.active, 'tool')
  acknowledge(); await pending
  assert.equal(localCalls, 1)
  assert.equal(useStore.getState().status.active, 'tool', 'Acknowledging Stop must not report idle while backend work can still run')
  assert.equal(useStore.getState().approvals.active.length, 1, 'Pending approval ownership must remain until the terminal event settles it')
  useStore.getState().applyEvent({ runId: 'synthetic-run', threadId: 'active', type: 'done' })
  assert.equal(useStore.getState().status.active, 'idle')
  assert.equal(useStore.getState().approvals.active, undefined)
  let remoteCalls = 0
  api.cancelRemoteOrchestratorThread = async (host, threadId) => { remoteCalls++; assert.equal(host, 'synthetic-host'); assert.equal(threadId, 'active') }
  reset('synthetic-host')
  await useStore.getState().cancel('active')
  assert.equal(remoteCalls, 1)
  assert.equal(useStore.getState().status.active, 'tool', 'Remote cancellation acknowledgement also does not prove work drained')
  assert.equal(useStore.getState().approvals.active.length, 1)
  useStore.getState().applyEvent({ runId: 'synthetic-remote-run', threadId: 'active', type: 'done', orchestratorHostId: 'synthetic-host' })
  assert.equal(useStore.getState().status.active, 'idle')
  reset()
  api.cancelRun = async () => { useStore.getState().applyEvent({ runId: 'old-run', threadId: 'active', type: 'done' }); useStore.setState({ status: { active: 'thinking' }, approvals: { active: [{ callId: 'new-approval', itemId: 'new-item' }] } }) }
  await useStore.getState().cancel('active')
  assert.equal(useStore.getState().status.active, 'thinking', 'A late cancellation acknowledgement must not retire a later run')
  assert.equal(useStore.getState().approvals.active[0].callId, 'new-approval')
  reset()
  api.cancelRun = async () => { throw new Error('Synthetic Stop request could not be delivered') }
  await assert.rejects(useStore.getState().cancel('active'), /could not be delivered/)
  assert.equal(useStore.getState().status.active, 'tool')
  assert.equal(useStore.getState().approvals.active.length, 1)
  console.log('PASS actual renderer Stop preserves busy status and approvals through local/remote acknowledgement or failed delivery; only terminal events settle idle')
} finally { Object.assign(api, original) }
