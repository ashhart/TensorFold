import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { createServer } from 'node:http'
import { createRequire, Module } from 'node:module'
import os from 'node:os'
import path from 'node:path'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-phone-event-projection-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
// Only Electron's version API is replaced; the host emitter, webhook serializer,
// DNS-pinned fetch and loopback HTTP receiver below are the production path.
const require = createRequire(path.join(process.cwd(), 'package.json'))
const electronPath = require.resolve('electron')
const previousElectron = require.cache[electronPath]
const electron = new Module(electronPath)
electron.exports = { app: { getVersion: () => '0.0.0-synthetic' } }
electron.loaded = true
require.cache[electronPath] = electron
const storage = await import('../../store')
const { store } = await import('../state')
const { createEmitHelpers } = await import('./emit-helpers')
const { runAgent } = await import('../../agent/loop')
const { PhoneService } = await import('../../agent/phone/service')
const saved = { ...store }
const originalFetch = globalThis.fetch
const marker = 'PRIVATE_SYNTHETIC_PHONE_CAPTURE_AMBER'
const ordinaryMarker = 'PUBLIC_SYNTHETIC_ORDINARY_AMBER'
let wire: string[] = [], rendered: unknown[] = [], remote: unknown[] = [], observed: unknown[] = [], spined: unknown[] = [], attention: unknown[] = []
let completedDeliveries = 0, attemptedDeliveries = 0
const receiver = createServer((request, response) => {
  let body = ''
  request.setEncoding('utf8')
  request.on('data', (part) => { body += part })
  request.on('end', () => { wire.push(body); response.writeHead(200); response.end('ok') })
})
await new Promise<void>((resolve, reject) => { receiver.once('error', reject); receiver.listen(0, '127.0.0.1', resolve) })
const address = receiver.address()
assert.ok(address && typeof address === 'object')
const receiverUrl = `http://127.0.0.1:${address.port}/synthetic-webhook`
const endpoint: any = { id: 'synthetic-phone-event-endpoint', direction: 'outbound', enabled: true, url: receiverUrl, secretRef: 'public-synthetic-fixture-secret', approvedAt: 1, approvalRequired: true, eventKinds: ['agent'], deliveries: [], createdAt: 1, updatedAt: 1 }
// Count actual delivery completion even when the bounded UI history keeps only 30.
Object.defineProperty(endpoint, 'eventKinds', { enumerable: false, get: () => { attemptedDeliveries++; return ['agent'] } })
Object.defineProperty(endpoint, 'deliveries', { get: () => [], set: () => { completedDeliveries++ }, configurable: true })
const provider = { id: 'synthetic-phone-projection', name: 'Synthetic local model', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['synthetic'] }
const device = { id: 'android:synthetic-projection', name: 'Synthetic phone', platform: 'android' as const, transport: 'usb' as const, state: 'ready' as const, detail: null }
let gestures = 0
const backend: any = { observe: async () => ({ width: 300, height: 600, image: null, app: 'example.synthetic', warnings: [], nodes: [{ text: marker, description: '', role: 'Button', bounds: { x: 0, y: 0, width: 100, height: 100 }, editable: false, secure: false, enabled: true, focused: false, interactive: true }] }), act: async () => { gestures++; throw new Error('No device actions in this fixture') }, disconnect: async () => {} }
const service = new PhoneService({ discover: async () => ({ adb: { available: false, path: null, version: null }, devices: [device], mirroring: { available: false, detail: null } }), connect: async () => backend, connectDeveloper: async () => ({ device, backend }), authorizeThread() {}, cancelThread() {}, emit() {} })
const emit = createEmitHelpers({ remoteServer: () => ({ publish: (event) => remote.push(structuredClone(event)) }), spineHost: () => ({ record: (event: unknown) => spined.push(structuredClone(event)) }) as any, observerHub: () => ({ emit: (...args: unknown[]) => observed.push(structuredClone(args)) }) as any, persistMissions() {}, onAgentEvent: (event) => attention.push(structuredClone(event)) })
let requests = 0, approvals = 0, phone = true
// Only the synthetic model transport is stubbed. Webhooks use an actual owned
// loopback listener through production DNS pinning; other destinations fail.
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  const url = String(input?.url ?? input)
  if (url === receiverUrl) return originalFetch(input, init)
  assert.ok(url.startsWith(provider.baseUrl), 'no external model or webhook destination')
  const step = requests++
  const snapshot = service.snapshot()
  const delta = step === 0 && phone ? { tool_calls: [{ index: 0, id: 'synthetic-observe', type: 'function', function: { name: 'phone_observe', arguments: '{}' } }] }
    : step === 1 && phone ? { tool_calls: [{ index: 0, id: 'synthetic-control', type: 'function', function: { name: 'phone_control', arguments: JSON.stringify({ action: 'home', sessionId: snapshot.session!.id, observationId: snapshot.observation!.id, note: marker }) } }] }
      : { content: phone ? 'Synthetic phone inspection complete.' : ordinaryMarker }
  return new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: 'tool_calls' in delta ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch
async function settleWebhooks() {
  for (let i = 0; i < 400; i++) {
    if (completedDeliveries === attemptedDeliveries && !storage.writeChains.size) { await new Promise((resolve) => setTimeout(resolve, 20)); if (completedDeliveries === attemptedDeliveries && !storage.writeChains.size) return }
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  throw new Error('Owned webhook deliveries did not settle')
}
try {
  store.settings = storage.defaultSettings()
  store.win = { isDestroyed: () => false, webContents: { isDestroyed: () => false, send: (_channel: string, event: unknown) => rendered.push(structuredClone(event)) } } as any
  await service.refresh(); await service.connect({ deviceId: device.id })
  for (const mode of ['phone', 'dynamic-phone', 'incognito', 'ordinary'] as const) {
    await settleWebhooks()
    wire = []; rendered = []; remote = []; observed = []; spined = []; attention = []; requests = 0; approvals = 0; completedDeliveries = 0; attemptedDeliveries = 0
    phone = mode !== 'ordinary'
    store.webhooks = [endpoint]
    await service.stop()
    const thread: any = { id: `synthetic-${mode}`, title: 'Synthetic phone projection', kind: 'chat', cwd: home, phoneLocalOnly: mode === 'phone' || mode === 'incognito', incognito: mode === 'incognito', providerId: provider.id, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Inspect only this synthetic phone.' }], timeline: [] }
    store.threads = [thread]
    service.grant(thread.id)
    await runAgent({ thread, provider, model: 'synthetic', approvalMode: 'suggest', temperature: 0, reasoningEffort: 'low', autoCompact: false, maxSteps: 4, signal: new AbortController().signal, persist() {}, emit: emit.emitToRenderer, requestApproval: async () => { approvals++; return 'deny' }, toolContext: { cwd: home, backend: { kind: 'ssh' } as any, currentThread: thread, phoneLocalOnly: thread.phoneLocalOnly, phoneControlAvailable: true, phonePrivacyGuard: () => null, runPhoneControl: (input) => { if (mode === 'dynamic-phone') thread.phoneLocalOnly = true; return service.tool(input, thread.id) } } })
    await settleWebhooks()
    const contains = (value: unknown, text: string) => JSON.stringify(value).includes(text)
    if (phone) {
      assert.equal(approvals, 1, 'Home reaches review and is denied before device dispatch')
      assert.equal(contains(rendered, marker), true, 'the owning native UI retains the private phone result and approval note')
      assert.equal(contains(wire, marker), false, `${mode}: captured phone text must not enter actual automatic webhook HTTP bytes before or after action review`)
      assert.equal(contains(observed, marker) || contains(spined, marker) || contains(attention, marker), false)
      if (mode === 'phone' || mode === 'incognito') assert.deepEqual([wire.length, observed.length, spined.length, attention.length], [0, 0, 0, 0])
      if (mode === 'incognito') assert.equal(remote.length, 0)
      else assert.equal(contains(remote, marker), true, 'separately authorized owner UI projection is retained; paired-client filtering belongs to the remote server')
    } else {
      assert.equal(contains(wire, ordinaryMarker), false, 'webhooks compact away assistant transcript text')
      assert.ok(wire.length > 0 && observed.length > 0 && spined.length > 0 && attention.length > 0, 'ordinary automatic projections remain operational')
      emit.emitToRenderer({ threadId: thread.id, runId: 'synthetic-public', type: 'item_add', item: { id: 'public-item', kind: 'tool', title: ordinaryMarker, path: ordinaryMarker, state: 'ok' } })
      await settleWebhooks()
      assert.equal(contains(wire, ordinaryMarker), true, 'actual compacted ordinary metadata still reaches the configured loopback webhook')
    }
    assert.equal(gestures, 0)
  }
  await settleWebhooks()
  wire = []; observed = []; spined = []; attention = []; completedDeliveries = 0; attemptedDeliveries = 0; store.threads = []
  emit.emitToRenderer({ threadId: 'synthetic-missing-owner', runId: 'synthetic-missing', type: 'item_add', item: { id: 'missing-item', kind: 'tool', path: marker, title: marker } })
  await settleWebhooks()
  assert.deepEqual([wire.length, observed.length, spined.length, attention.length], [0, 0, 0, 0], 'unresolved thread ownership cannot authorize automatic sinks')
} finally {
  await service.shutdown()
  await settleWebhooks()
  globalThis.fetch = originalFetch
  while (storage.writeChains.size) await Promise.allSettled([...storage.writeChains.values()])
  await new Promise<void>((resolve, reject) => receiver.close((error) => error ? reject(error) : resolve()))
  Object.assign(store, saved)
  if (previousElectron) require.cache[electronPath] = previousElectron
  else delete require.cache[electronPath]
  await storage.closeSessionSearchIndexes()
  await rm(home, { recursive: true, force: true, maxRetries: 5, retryDelay: 20 })
}
console.log('Actual host phone-event webhook wire isolation, dynamic grants, owner UI, ordinary hooks and unresolved-owner controls passed')
