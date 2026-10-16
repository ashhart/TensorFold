import assert from 'node:assert/strict'
import { PhoneService, PHONE_MAX_ACTIONS, PHONE_GRANT_LIFETIME_MS, type PhoneBackend, type PhoneRawObservation } from './service'

const device = { id: 'android:example', name: 'QA phone', platform: 'android' as const, transport: 'usb' as const, state: 'ready' as const, detail: null }
function harness() {
  let now = 100_000
  let reads = 0
  let inputs = 0
  let text = 'Notes'
  let local = true
  const cancelled: string[] = []
  const backend: PhoneBackend = {
    observe: async () => { reads++; return { width: 300, height: 600, image: null, app: 'example.notes', warnings: [], nodes: [{ text, description: '', role: 'Button', bounds: { x: 0, y: 0, width: 100, height: 100 }, editable: false, secure: false, enabled: true, focused: false, interactive: true }] } },
    act: async () => { inputs++; text = `View ${inputs}` },
    disconnect: async () => {},
  }
  const service = new PhoneService({
    discover: async () => ({ adb: { available: true, path: '/example/sdk/adb', version: '1' }, devices: [device], mirroring: { available: false, detail: null } }),
    connect: async () => backend,
    connectDeveloper: async () => ({ device, backend }),
    authorizeThread: (id) => { if (id !== 'local-chat' || !local) throw new Error('Local chat required.') },
    cancelThread: (id) => cancelled.push(id), emit: () => {}, now: () => now,
  })
  return { service, backend, cancelled, get reads() { return reads }, get inputs() { return inputs }, jump(ms: number) { now += ms }, external() { local = false } }
}
async function connected(h: ReturnType<typeof harness>) {
  await h.service.refresh()
  await h.service.connect({ deviceId: device.id })
}
function tap(service: PhoneService) {
  const snapshot = service.snapshot()
  return { action: 'tap' as const, sessionId: snapshot.session!.id, observationId: snapshot.observation!.id, x: 20, y: 20 }
}

{
  const h = harness()
  await connected(h)
  assert.equal(h.reads, 0, 'discovery and connection never capture phone content')
  assert.equal(h.service.snapshot().session?.threadId, null)
  assert.equal((await h.service.tool({ action: 'observe' }, 'local-chat')).ok, false)
  assert.equal(h.reads, 0, 'a model has no phone inspection before its explicit grant')
  assert.throws(() => h.service.grant('other-chat'), /Local chat/)
  h.service.grant('local-chat')
  assert.equal((await h.service.tool({ action: 'observe' }, 'local-chat')).ok, true)
  const old = tap(h.service)
  await h.service.action(old, 'local-chat')
  assert.equal(h.inputs, 1)
  assert.equal(h.reads, 3, 'input uses a fresh pre-action check and a post-action observation')
  assert.equal(h.service.snapshot().session?.actions, 1)
  await assert.rejects(() => h.service.action(old, 'local-chat'), /stale/)
  const result = await h.service.tool({ action: 'observe' }, 'local-chat')
  assert.doesNotMatch(result.output, /data:image|base64/, 'screenshots never enter tool output or stored model messages')
  assert.match(result.output, /untrusted data/)
  await h.service.shutdown()
}
{
  const h = harness(); await connected(h); h.service.grant('local-chat'); await h.service.observe('local-chat')
  h.jump(30_001)
  await assert.rejects(() => h.service.action(tap(h.service)), /30 seconds/)
  assert.equal(h.inputs, 0)
  await h.service.action(tap(h.service), 'local-chat')
  assert.equal(h.inputs, 1, 'a delayed approved action still verifies the complete live screen before input')
  await h.service.observe('local-chat')
  await assert.rejects(() => h.service.action({ ...tap(h.service), x: 300 }, 'local-chat'), /inside/)
  h.external()
  assert.equal((await h.service.tool({ action: 'observe' }, 'local-chat')).ok, false)
  assert.equal(h.service.snapshot().session?.threadId, null)
  assert.equal(h.service.snapshot().observation, null, 'changing to an external route revokes cached screen data')
  await h.service.shutdown()
}
{
  const h = harness(); await connected(h); h.service.grant('local-chat')
  h.jump(PHONE_GRANT_LIFETIME_MS + 1)
  assert.equal((await h.service.tool({ action: 'observe' }, 'local-chat')).ok, false)
  assert.equal(h.reads, 0)
  await h.service.shutdown()
}
{
  const h = harness(); await connected(h); h.service.grant('local-chat')
  for (let i = 0; i < PHONE_MAX_ACTIONS; i++) { await h.service.observe('local-chat'); await h.service.action(tap(h.service), 'local-chat') }
  await h.service.observe('local-chat')
  await assert.rejects(() => h.service.action(tap(h.service), 'local-chat'), /40-action/)
  assert.equal(h.inputs, PHONE_MAX_ACTIONS)
  assert.equal(h.service.snapshot().session?.threadId, null)
  assert.deepEqual(h.cancelled, ['local-chat'])
  await h.service.shutdown()
}
{
  const h = harness(); await connected(h); h.service.grant('local-chat')
  h.backend.act = async () => {}
  await h.service.observe('local-chat')
  for (let i = 0; i < 2; i++) await h.service.action(tap(h.service), 'local-chat')
  await assert.rejects(() => h.service.action(tap(h.service), 'local-chat'), /no visible change/)
  assert.equal(h.service.snapshot().session?.threadId, null)
  await h.service.shutdown()
}
{
  const h = harness(); await connected(h); h.service.grant('local-chat')
  let begun!: () => void
  const started = new Promise<void>((resolve) => { begun = resolve })
  h.backend.observe = async (signal) => {
    begun()
    return new Promise<PhoneRawObservation>((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('Stopped.')), { once: true }))
  }
  const pending = h.service.observe('local-chat').then(() => false, () => true)
  await started
  await h.service.stop()
  assert.equal(await pending, true)
  assert.equal(h.service.snapshot().busy, false)
  assert.equal(h.service.snapshot().observation, null)
  assert.equal(h.service.snapshot().session?.threadId, null)
  assert.deepEqual(h.cancelled, ['local-chat'])
  await h.service.shutdown()
}
{
  const h = harness(); await connected(h); h.service.grant('local-chat')
  await h.service.shutdown()
  await assert.rejects(() => h.service.refresh(), /disconnecting/, 'a shut-down phone service admits no new operation')
  h.service.resume()
  await connected(h)
  assert.equal(h.service.snapshot().session?.threadId, null, 'failed-Quit recovery requires an explicit new phone grant')
  assert.equal((await h.service.tool({ action: 'observe' }, 'local-chat')).ok, false, 'resuming never restores the old chat authority')
  await h.service.shutdown()
}
console.log('Phone session, explicit grants, freshness, budgets, route isolation and Stop passed.')
