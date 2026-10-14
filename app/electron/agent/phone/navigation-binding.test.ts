import assert from 'node:assert/strict'
import { deflateSync } from 'node:zlib'
import { PhoneService, PHONE_GRANT_LIFETIME_MS, type PhoneBackend, type PhoneRawObservation } from './service'
import type { PhoneAction, PhoneActionInput } from '../types'

// Valid synthetic PNGs; no real screen, phone, account or provider is opened.
function png(color: number): string {
  const chunk = (name: string, data: Buffer) => {
    const buffer = Buffer.alloc(data.length + 12)
    buffer.writeUInt32BE(data.length); buffer.write(name, 4); data.copy(buffer, 8)
    let crc = 0xffffffff
    for (const byte of buffer.subarray(4, data.length + 8)) { crc ^= byte; for (let i = 0; i < 8; i++) crc = crc >>> 1 ^ (crc & 1 ? 0xedb88320 : 0) }
    buffer.writeUInt32BE((crc ^ 0xffffffff) >>> 0, data.length + 8)
    return buffer
  }
  const header = Buffer.alloc(13); header.writeUInt32BE(100); header.writeUInt32BE(100, 4); header[8] = 8; header[9] = 6
  const pixels = Buffer.alloc(100 * 401)
  for (let y = 0; y < 100; y++) for (let x = 0; x < 100; x++) { pixels[y * 401 + 1 + x * 4] = color; pixels[y * 401 + 4 + x * 4] = 255 }
  return `data:image/png;base64,${Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', deflateSync(pixels)), chunk('IEND', Buffer.alloc(0))]).toString('base64')}`
}
const images = [png(10), png(20), png(30), png(40)]
const node = { text: 'Notes', description: '', role: 'android.widget.Button', bounds: { x: 10, y: 10, width: 30, height: 30 }, editable: false, secure: false, enabled: true, focused: false, interactive: true }
const raw = (image: string, patch: Partial<PhoneRawObservation> = {}): PhoneRawObservation => ({ width: 100, height: 100, image, app: 'dev.example.synthetic', nodes: [structuredClone(node)], warnings: [], ...patch })
function fixture(options: { observe?: (read: number) => PhoneRawObservation | Promise<PhoneRawObservation>; act?: () => void } = {}) {
  let reads = 0, now = 1000, authorized = true
  const actions: PhoneAction[] = []
  const emitted: Array<ReturnType<PhoneService['snapshot']>> = []
  const device = { id: 'android:owned-navigation-fixture', name: 'Synthetic phone', platform: 'android' as const, transport: 'emulator' as const, state: 'ready' as const, detail: null }
  const backend: PhoneBackend = { observe: async () => options.observe ? options.observe(++reads) : raw(images[reads++ % images.length]), act: async (action) => { actions.push(action); options.act?.() }, disconnect: async () => {} }
  const service = new PhoneService({ discover: async () => ({ adb: { available: true, path: null, version: 'synthetic' }, devices: [device], mirroring: { available: false, detail: null } }), connect: async () => backend, connectDeveloper: async () => ({ device, backend }), authorizeThread: () => { if (!authorized) throw new Error('Synthetic privacy authority changed.') }, cancelThread() {}, emit: (snapshot) => { emitted.push(snapshot) }, now: () => now })
  return { service, actions, emitted, async prepare(grant = true) { await service.refresh(); await service.connect({ deviceId: device.id }); if (grant) service.grant('owned-chat'); await service.observe(grant ? 'owned-chat' : undefined) }, advance(ms: number) { now += ms }, revoke() { authorized = false } }
}
const input = (service: PhoneService, action: PhoneAction): PhoneActionInput => ({ ...action, sessionId: service.snapshot().session!.id, observationId: service.snapshot().observation!.id })

for (const action of ['home', 'back'] as const) {
  const h = fixture()
  try {
    await h.prepare(); const accepted = input(h.service, { action }); h.advance(32_000)
    await h.service.action(accepted, 'owned-chat')
    assert.equal(h.actions.length, 1, 'approved non-target navigation accepts pixel-only drift with identical app, dimensions and full accessible semantics')
    assert.notEqual(h.service.snapshot().observation!.id, accepted.observationId)
    await assert.rejects(h.service.action(accepted, 'owned-chat'), /stale/)
    assert.equal(h.actions.length, 1, 'a consumed approval cannot navigate twice')
  } finally { await h.service.shutdown() }
}

for (const action of [
  { action: 'tap', x: 20, y: 20 }, { action: 'swipe', x: 20, y: 20, toX: 70, toY: 70 },
  { action: 'type', text: 'synthetic draft' }, { action: 'launch', app: 'dev.example.target' }, { action: 'enter' },
] as PhoneAction[]) {
  const h = fixture({ observe: (read) => raw(images[(read - 1) % images.length], action.action === 'type' ? { nodes: [{ ...node, editable: true, focused: true }] } : {}) })
  try {
    await h.prepare(action.action !== 'enter')
    await assert.rejects(h.service.action(input(h.service, action), action.action === 'enter' ? undefined : 'owned-chat'), /screen changed/)
    assert.equal(h.actions.length, 0, `${action.action} retains full screenshot binding`)
  } finally { await h.service.shutdown() }
}

for (const fresh of [
  raw(images[1], { app: 'dev.example.different' }), raw(images[1], { width: 101 }), raw(images[1], { height: 101 }),
  raw(images[1], { nodes: [{ ...node, text: 'Changed screen' }] }), raw(images[1], { nodes: [{ ...node, description: 'Changed description' }] }),
  raw(images[1], { nodes: [{ ...node, bounds: { ...node.bounds, x: 11 } }] }), raw(images[1], { nodes: [{ ...node, role: 'android.widget.TextView' }] }),
  raw(images[1], { nodes: [{ ...node, editable: true }] }), raw(images[1], { nodes: [node, { ...node, text: 'New dialog' }] }),
  raw(images[1], { nodes: [{ ...node, focused: true }] }), raw(images[1], { nodes: [{ ...node, enabled: false }] }),
  raw(images[1], { nodes: [{ ...node, interactive: false }] }), raw(images[1], { nodes: [{ ...node, secure: true }] }),
]) {
  const h = fixture({ observe: (read) => read === 1 ? raw(images[0]) : fresh })
  try { await h.prepare(); await assert.rejects(h.service.action(input(h.service, { action: 'home' }), 'owned-chat'), /screen changed|protected field/); assert.equal(h.actions.length, 0) }
  finally { await h.service.shutdown() }
}
for (const app of [null, '', '   ']) {
  const h = fixture({ observe: (read) => raw(images[(read - 1) % images.length], { app }) })
  try { await h.prepare(); await assert.rejects(h.service.action(input(h.service, { action: 'back' }), 'owned-chat'), /screen changed/); assert.equal(h.actions.length, 0, 'an unknown foreground app cannot receive the pixel-drift exception') }
  finally { await h.service.shutdown() }
}
{
  const h = fixture()
  try { await h.prepare(false); h.advance(32_000); await assert.rejects(h.service.action(input(h.service, { action: 'home' })), /older than 30 seconds/); assert.equal(h.actions.length, 0, 'manual preview expiry is unchanged') }
  finally { await h.service.shutdown() }
}
{
  const h = fixture()
  try { await h.prepare(); h.advance(PHONE_GRANT_LIFETIME_MS + 1); await assert.rejects(h.service.action(input(h.service, { action: 'home' }), 'owned-chat'), /allow|expired|grant/i); assert.equal(h.actions.length, 0) }
  finally { await h.service.shutdown() }
}
{
  const h = fixture()
  try { await h.prepare(); const wrong = input(h.service, { action: 'home' }); wrong.sessionId = 'different-session'; await assert.rejects(h.service.action(wrong, 'owned-chat'), /stale/); assert.equal(h.actions.length, 0) }
  finally { await h.service.shutdown() }
}
{
  const h = fixture()
  try { await h.prepare(); await assert.rejects(h.service.action(input(h.service, { action: 'home' }), 'another-chat'), /Allow this chat/); assert.equal(h.actions.length, 0) }
  finally { await h.service.shutdown() }
}
{
  const h = fixture({ observe: (read) => { if (read > 1) h.revoke(); return raw(images[(read - 1) % images.length]) } })
  try { await h.prepare(); await assert.rejects(h.service.action(input(h.service, { action: 'home' }), 'owned-chat'), /Allow this chat/); assert.equal(h.actions.length, 0, 'authorization is checked again after fresh observation and before navigation') }
  finally { await h.service.shutdown() }
}
{
  const control = new AbortController()
  const h = fixture({ observe: (read) => { if (read > 1) control.abort(); return raw(images[(read - 1) % images.length]) } })
  try { await h.prepare(); await assert.rejects(h.service.action(input(h.service, { action: 'home' }), 'owned-chat', control.signal), /cancel|stop/i); assert.equal(h.actions.length, 0) }
  finally { await h.service.shutdown() }
}
{
  const h = fixture()
  try {
    await h.prepare()
    for (let i = 0; i < 2; i++) { if (i) await h.service.observe('owned-chat'); await h.service.action(input(h.service, { action: 'home' }), 'owned-chat') }
    await h.service.observe('owned-chat')
    await assert.rejects(h.service.action(input(h.service, { action: 'home' }), 'owned-chat'), /Three phone actions produced no visible change/)
    assert.equal(h.actions.length, 3)
    assert.equal(h.service.snapshot().session?.threadId, null, 'pixel drift cannot hide repeated ineffective navigation from the existing three-action pause')
  } finally { await h.service.shutdown() }
}
// Retain the adjacent alias/privacy fix: reauthorization precedes projection,
// including the post-action observation, not just the final returned grant check.
{
  const h = fixture({ observe: () => { h.revoke(); return raw(images[0]) } })
  try { await assert.rejects(h.prepare(), /Allow this chat/); assert.ok(h.emitted.every((snapshot) => snapshot.observation === null)); assert.equal(h.actions.length, 0) }
  finally { await h.service.shutdown() }
}
{
  const h = fixture({ observe: (read) => { if (read >= 3) h.revoke(); return raw(images[(read - 1) % images.length]) } })
  try {
    await h.prepare(); await assert.rejects(h.service.action(input(h.service, { action: 'home' }), 'owned-chat'), /Allow this chat/)
    assert.equal(h.actions.length, 1, 'the explicitly approved action precedes a later route change')
    assert.ok(!h.emitted.some((snapshot) => snapshot.observation?.image === images[2]), 'the new post-action screen is never projected after privacy authority changes')
  } finally { await h.service.shutdown() }
}
console.log('Actual PhoneService navigation binding passed: approved Home/Back pixel drift only, full semantic/app/dimension/secure/ID/grant/cancel checks, strict target actions and no-progress pause')
