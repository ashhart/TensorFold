import assert from 'node:assert/strict'
import { deflateSync } from 'node:zlib'
import { PhoneService, PHONE_MAX_ACTIONS } from './service'
import type { PhoneBackend, PhoneRawObservation } from './service'
import type { PhoneAction, PhoneActionInput, PhoneDevice, PhoneNode } from '../types'

const device: PhoneDevice = { id: 'android:synthetic-safety-phone', name: 'Synthetic phone', platform: 'android', transport: 'emulator', state: 'ready', detail: null }
type Node = Omit<PhoneNode, 'id'>
const baseNode: Node = { text: 'Browse', description: '', role: 'android.widget.Button', bounds: { x: 10, y: 10, width: 30, height: 30 }, editable: false, secure: false, enabled: true, focused: false, interactive: true }
function png(color: number): string {
  const chunk = (type: string, body: Buffer) => {
    const result = Buffer.alloc(body.length + 12)
    result.writeUInt32BE(body.length, 0); result.write(type, 4); body.copy(result, 8)
    let crc = 0xffffffff
    for (const byte of result.subarray(4, body.length + 8)) { crc ^= byte; for (let bit = 0; bit < 8; bit++) crc = crc >>> 1 ^ (crc & 1 ? 0xedb88320 : 0) }
    result.writeUInt32BE((crc ^ 0xffffffff) >>> 0, body.length + 8)
    return result
  }
  const header = Buffer.alloc(13); header.writeUInt32BE(1, 0); header.writeUInt32BE(1, 4); header[8] = 8; header[9] = 6
  return `data:image/png;base64,${Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', deflateSync(Buffer.from([0, color, 0, 0, 255]))), chunk('IEND', Buffer.alloc(0))]).toString('base64')}`
}
const imageA = png(10), imageB = png(20)
function raw(overrides: Partial<PhoneRawObservation> = {}): PhoneRawObservation {
  return { width: 100, height: 100, image: imageA, nodes: [structuredClone(baseNode)], app: 'dev.example.phone', warnings: [], ...overrides }
}
function fixture(options: { observe?: PhoneBackend['observe']; act?: PhoneBackend['act']; disconnect?: PhoneBackend['disconnect'] } = {}) {
  const actions: PhoneAction[] = []
  const backend: PhoneBackend = { observe: options.observe ?? (async () => raw()), act: async (action, dimensions, signal) => { actions.push(action); await options.act?.(action, dimensions, signal) }, disconnect: options.disconnect ?? (async () => {}) }
  const service = new PhoneService({ discover: async () => ({ adb: { available: true, path: null, version: 'synthetic' }, devices: [device], mirroring: { available: false, detail: null } }), connect: async () => backend, connectDeveloper: async () => ({ device, backend }), authorizeThread: () => {}, cancelThread: () => {}, emit: () => {} })
  return { service, actions }
}
async function prepare(service: PhoneService): Promise<void> {
  await service.refresh(); await service.connect({ deviceId: device.id }); service.grant('synthetic-chat'); await service.observe('synthetic-chat')
}
function input(service: PhoneService, action: PhoneAction): PhoneActionInput {
  const snapshot = service.snapshot()
  assert.ok(snapshot.session && snapshot.observation)
  return { ...action, sessionId: snapshot.session.id, observationId: snapshot.observation.id }
}

// Targeted actions keep full pixel and focus binding even when accessibility text matches.
for (const change of [
  () => raw({ image: imageB }),
  () => raw({ nodes: [{ ...baseNode, enabled: false }] }),
  () => raw({ nodes: [{ ...baseNode, focused: true }] }),
  () => raw({ nodes: [{ ...baseNode, interactive: false }] }),
]) {
  let reads = 0
  const { service, actions } = fixture({ observe: async () => ++reads === 1 ? raw() : change() })
  try {
    await prepare(service)
    await assert.rejects(() => service.action(input(service, { action: 'tap', x: 20, y: 20 }), 'synthetic-chat'), /screen changed/i)
    assert.equal(actions.length, 0)
    assert.equal(service.snapshot().observation, null)
  } finally { await service.shutdown() }
}

// Unknown, disabled, secure or committing controls require a human, not a guess.
for (const node of [
  { ...baseNode, text: '', description: '' },
  { ...baseNode, enabled: false },
  { ...baseNode, text: 'Send message' },
  { ...baseNode, text: 'Approve permission' },
  { ...baseNode, text: 'Delete item' },
  { ...baseNode, text: 'Checkout' },
  { ...baseNode, text: 'Arbitrary text', role: 'android.widget.TextView', interactive: false },
] satisfies Node[]) {
  const { service, actions } = fixture({ observe: async () => raw({ nodes: [node] }) })
  try {
    await prepare(service)
    await assert.rejects(() => service.action(input(service, { action: 'tap', x: 20, y: 20 }), 'synthetic-chat'), /manual|verified labelled|complete the final/i)
    assert.equal(actions.length, 0)
  } finally { await service.shutdown() }
}
{
  const { service, actions } = fixture()
  try {
    await prepare(service)
    await assert.rejects(() => service.action(input(service, { action: 'tap', x: 80, y: 80 }), 'synthetic-chat'), /verified labelled/i)
    assert.equal(actions.length, 0)
  } finally { await service.shutdown() }
}
{
  const { service, actions } = fixture({ observe: async () => raw({ nodes: [{ ...baseNode, secure: true }] }) })
  try {
    await service.refresh(); await service.connect({ deviceId: device.id }); service.grant('synthetic-chat')
    await assert.rejects(() => service.observe('synthetic-chat'), /protected field/i)
    assert.equal(service.snapshot().observation, null)
    assert.equal(actions.length, 0)
  } finally { await service.shutdown() }
}
for (const node of [baseNode, { ...baseNode, text: 'Settings', role: 'ocr', interactive: false }] satisfies Node[]) {
  const { service, actions } = fixture({ observe: async () => raw({ nodes: [node] }) })
  try {
    await prepare(service)
    await service.action(input(service, { action: 'tap', x: 20, y: 20 }), 'synthetic-chat')
    assert.equal(actions.length, 1, 'verified labelled controls and conservative OCR navigation remain usable')
  } finally { await service.shutdown() }
}

for (const nodes of [
  [baseNode],
  [{ ...baseNode, editable: true, focused: false }],
  [{ ...baseNode, editable: true, focused: true, enabled: false }],
] satisfies Node[][]) {
  const { service, actions } = fixture({ observe: async () => raw({ nodes }) })
  try {
    await prepare(service)
    await assert.rejects(() => service.action(input(service, { action: 'type', text: 'Synthetic text' }), 'synthetic-chat'), /focused text field/i)
    assert.equal(actions.length, 0)
  } finally { await service.shutdown() }
}
{
  const node: Node = { ...baseNode, text: 'Search', role: 'android.widget.EditText', editable: true, focused: true }
  const { service, actions } = fixture({ observe: async () => raw({ nodes: [node] }) })
  try {
    await prepare(service)
    for (const action of [{ action: 'enter' }, { action: 'type', text: 'line\nbreak' }, { action: 'type', text: 'line\rbreak' }, { action: 'type', text: 'line\tbreak' }] satisfies PhoneAction[]) {
      await assert.rejects(() => service.action(input(service, action), 'synthetic-chat'), /manual/i)
    }
    assert.equal(actions.length, 0)
    await service.action(input(service, { action: 'type', text: 'Synthetic text' }), 'synthetic-chat')
    assert.equal(actions.length, 1)
  } finally { await service.shutdown() }
}

// An attempted input counts even when its completion or after-screen is lost.
for (const failsDuringInput of [false, true]) {
  let afterInput = false
  const { service, actions } = fixture({
    observe: async () => { if (afterInput) { afterInput = false; throw new Error('Synthetic post-input capture failure') } return raw() },
    act: async () => { if (failsDuringInput) throw new Error('Synthetic uncertain input result'); afterInput = true },
  })
  try {
    await prepare(service)
    for (let index = 0; index < PHONE_MAX_ACTIONS; index++) {
      if (index) await service.observe('synthetic-chat')
      await assert.rejects(() => service.action(input(service, { action: 'home' }), 'synthetic-chat'), /Synthetic/i)
      assert.equal(service.snapshot().session?.actions, index + 1)
      assert.equal(service.snapshot().observation, null, 'a failed dispatched attempt consumes the screen id')
    }
    await service.observe('synthetic-chat')
    await assert.rejects(() => service.action(input(service, { action: 'home' }), 'synthetic-chat'), /40-action limit/i)
    assert.equal(actions.length, PHONE_MAX_ACTIONS)
    assert.equal(service.isGranted('synthetic-chat'), false)
  } finally { await service.shutdown() }
}

// Provider teardown remains exclusive until it finishes, even after Stop settles.
{
  let finish!: () => void
  let disconnectCalls = 0
  const { service } = fixture({ disconnect: async () => { disconnectCalls++; await new Promise<void>((resolve) => { finish = resolve }) } })
  await prepare(service)
  const pending = service.disconnect()
  for (let tick = 0; tick < 20 && !finish; tick++) await Promise.resolve()
  assert.ok(finish)
  assert.equal(service.snapshot().busy, true)
  await assert.rejects(() => service.connect({ deviceId: device.id }), /disconnect|closing/i)
  await assert.rejects(() => service.refresh(), /disconnect|closing/i)
  assert.throws(() => service.grant('synthetic-chat'), /closing/i)
  const duplicate = service.disconnect()
  assert.equal(disconnectCalls, 1)
  finish(); await Promise.all([pending, duplicate])
  assert.equal(service.snapshot().busy, false)
  assert.equal(service.snapshot().session, null)
  await service.connect({ deviceId: device.id })
  // Avoid asking the deliberately blocked fixture to disconnect twice at cleanup.
  const closing = service.shutdown()
  for (let tick = 0; tick < 20 && disconnectCalls < 2; tick++) await Promise.resolve()
  finish(); await closing
}

console.log('Phone service freshness, target safety, dispatch budgets and exclusive teardown checks passed.')
