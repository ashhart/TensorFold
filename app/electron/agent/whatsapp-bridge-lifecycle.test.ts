import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import {
  makeWhatsAppBridgeSource, startWhatsAppBridge, stopAllWhatsAppBridges,
  isWhatsAppBridgeReady, getWhatsAppBridgeRuntime, sendWhatsAppMessage,
  normalizeWhatsAppJid, validateWhatsAppSourcePatch,
  normalizeWhatsAppConversationJid,
} from './whatsapp-bridge.ts'
import type { ConnectorEvent, WhatsAppBridgeSource } from './types'

const own = ['15551230000', 's.whatsapp.net'].join('@')
const contact = ['15551230001', 's.whatsapp.net'].join('@')
const stranger = ['15551230002', 's.whatsapp.net'].join('@')
const ownLid = ['100000000001', 'lid'].join('@')
const tick = () => new Promise<void>((resolve) => setImmediate(resolve))
function fixture() {
  let current: WhatsAppBridgeSource | undefined = { ...makeWhatsAppBridgeSource('Isolated fixture'), allowedChatJids: [contact] }
  const source = current
  const sockets: Array<any> = []
  const events: ConnectorEvent[] = []
  const states: any[] = []
  const timers: Array<{ callback: () => void; delay: number; cancelled: boolean }> = []
  let authCalls = 0
  let mediaCalls = 0
  let failLoad = false
  const B = {
    makeWASocket: () => {
      const socket: any = {
        ev: new EventEmitter(), ended: false,
        end() { socket.ended = true; socket.ev.emit('connection.update', { connection: 'close' }) },
        user: { id: ['15551230000:1', 's.whatsapp.net'].join('@'), lid: ownLid },
        sendMessage: async (jid: string, body: { text: string }, options: { messageId: string }) => {
          socket.ev.emit('messages.upsert', { type: 'notify', messages: [message(jid, options.messageId, body.text, true)] })
          return { key: { id: options.messageId } }
        },
      }
      sockets.push(socket)
      return socket
    },
    fetchLatestBaileysVersion: async () => ({ version: [2, 0, 0] }),
    DisconnectReason: { loggedOut: 401 },
    downloadMediaMessage: async () => { mediaCalls++; throw new Error('Fixture does not download bytes') },
  }
  const deps = {
    loadBaileys: async () => { if (failLoad) throw new Error('Offline retry fixture'); return B },
    getSource: () => current,
    onRuntime: (snapshot: unknown) => { states.push(snapshot) },
    schedule: (callback: () => void, delay: number) => {
      const timer = { callback, delay, cancelled: false }; timers.push(timer)
      return timer as unknown as ReturnType<typeof setTimeout>
    },
    cancelSchedule: (timer: ReturnType<typeof setTimeout>) => { (timer as any).cancelled = true },
  }
  const auth = async () => {
    authCalls++
    return { state: { creds: { registered: true } }, saveCreds: async () => 123, credentialUpdatedAt: () => null }
  }
  return {
    source, sockets, events, states, timers, deps, auth,
    setCurrent: (value: WhatsAppBridgeSource | undefined) => { current = value },
    failLoad: () => { failLoad = true },
    mediaCalls: () => mediaCalls, authCalls: () => authCalls,
    start: (extra = {}) => startWhatsAppBridge(source, '/tmp/unused-offline-fixture', () => {}, (event) => events.push(event), (patch) => { if (current) current = { ...current, ...patch } }, auth, { ...deps, ...extra }),
  }
}
function message(jid: string, id: string, text = 'Synthetic fixture', fromMe = false): any {
  return { key: { remoteJid: jid, id, fromMe }, message: { conversation: text }, messageTimestamp: 1000 }
}
function close(socket: any, code: number) { socket.ev.emit('connection.update', { connection: 'close', lastDisconnect: { error: { output: { statusCode: code } } } }) }

try {
  const f = fixture()
  await f.start()
  assert.equal(isWhatsAppBridgeReady(f.source.id), false, 'socket allocation is not an authenticated open connection')
  f.sockets[0].ev.emit('connection.update', { connection: 'open' })
  assert.equal(isWhatsAppBridgeReady(f.source.id), true)
  assert.equal(getWhatsAppBridgeRuntime(f.source.id)[0].phoneNumber, '+15551230000', 'phone is the socket-authenticated identity')
  assert.ok(f.states.every((state, index) => index === 0 || state.updatedAt > f.states[index - 1].updatedAt), 'same-tick lifecycle pushes have strictly increasing revisions')
  close(f.sockets[0], 500)
  assert.equal(isWhatsAppBridgeReady(f.source.id), false, 'a closed socket cannot be advertised as ready')
  await assert.rejects(() => sendWhatsAppMessage(f.source.id, contact, 'Fixture'), /not connected/)
  stopAllWhatsAppBridges()

  const scope = fixture()
  await scope.start()
  const sock = scope.sockets[0]
  sock.ev.emit('connection.update', { connection: 'open' })
  const emit = (messages: any[]) => sock.ev.emit('messages.upsert', { type: 'notify', messages })
  emit([message(stranger, 'unselected'), message(contact, 'selected'), message(own, 'self-not-selected', 'Self fixture', true)])
  emit([{ ...message(stranger, 'unselected-media'), message: { imageMessage: { mediaKey: 'fixture', caption: 'Blocked media' } } }])
  await tick()
  assert.equal(scope.events.length, 1, 'only exact selected contacts ingest; self-chat is opt-in')
  assert.equal(scope.mediaCalls(), 0, 'unselected attachment never enters download or cache work')
  await assert.rejects(() => sendWhatsAppMessage(scope.source.id, stranger, 'Fixture'), /outside.*scope/)
  scope.setCurrent({ ...scope.source, allowedChatJids: [], allowSelfChat: true })
  emit([message(contact, 'revoked-contact'), message(own, 'self-allowed', 'My synthetic note', true), message(ownLid, 'self-lid', 'My linked identity', true)])
  assert.equal(scope.events.length, 3, 'current settings, not the captured start source, gate live ingress')
  assert.equal(scope.events[1].direction, 'inbound')
  assert.equal(scope.events[1].payload?.selfChat, true)
  assert.equal(scope.events[1].payload?.authenticatedAccountJid, own)
  assert.equal(scope.events[1].payload?.remoteCommandEligible, false)
  assert.equal(scope.events[2].payload?.jid, own, 'the actual socket-authenticated own LID is canonicalized to the account PN')
  assert.equal(scope.events[2].threadKey, scope.events[1].threadKey, 'authenticated PN/LID self-chat aliases share one conversation')
  const beforeSend = scope.events.length
  const sent = await sendWhatsAppMessage(scope.source.id, own, 'Approved synthetic outbound')
  assert.ok(sent.sentId)
  assert.equal(scope.events.length, beforeSend, 'synchronous bridge echo of an outbound self-chat send is suppressed')
  sock.ev.emit('messaging-history.set', { messages: [message(own, 'old-self-note', '/run synthetic', true), message(stranger, 'old-stranger')] })
  assert.equal(scope.events.length, beforeSend + 1)
  assert.equal(scope.events.at(-1)?.payload?.historical, true, 'backfill is explicitly historical content')
  assert.equal(scope.events.at(-1)?.payload?.remoteCommandEligible, false, 'history never becomes a command')
  scope.setCurrent({ ...scope.source, enabled: false, allowSelfChat: true })
  emit([message(own, 'disabled-self', 'Fixture', true)])
  await assert.rejects(() => sendWhatsAppMessage(scope.source.id, own, 'Fixture'), /outside.*scope/)
  assert.equal(scope.events.length, beforeSend + 1)
  stopAllWhatsAppBridges()

  const groups = fixture()
  const group = ['120363000001', 'g.us'].join('@')
  const legacyGroup = ['15551230000-123456', 'g.us'].join('@')
  groups.setCurrent({ ...groups.source, allowedChatJids: [group, legacyGroup], allowSelfChat: true })
  await groups.start()
  const groupSocket = groups.sockets[0]
  groupSocket.ev.emit('connection.update', { connection: 'open' })
  const groupMessage = { ...message(group, 'selected-group'), pushName: 'Synthetic participant', key: { remoteJid: group, id: 'selected-group', fromMe: false, participant: contact } }
  groupSocket.ev.emit('messages.upsert', { type: 'notify', messages: [groupMessage, message(stranger, 'not-selected'), message(group, 'own-group-echo', 'Outbound echo', true)] })
  assert.equal(groups.events.length, 1, 'only explicitly selected groups ingest; our outbound group messages stay suppressed')
  assert.equal(groups.events[0].actor, 'Synthetic participant', 'group author is the participant display name, never the group identity')
  assert.equal(groups.events[0].payload?.selfChat, false)
  assert.equal(groups.events[0].payload?.remoteCommandEligible, false, 'group members never acquire control or authorization')
  await sendWhatsAppMessage(groups.source.id, group, 'Approved synthetic group reply')
  assert.equal(groups.events.length, 1, 'approved outbound group echo is ignored')
  groups.setCurrent({ ...groups.source, allowedChatJids: [] })
  groupSocket.ev.emit('messaging-history.set', { messages: [message(group, 'revoked-group-history')] })
  await assert.rejects(() => sendWhatsAppMessage(groups.source.id, group, 'Fixture'), /outside.*scope/)
  assert.equal(groups.events.length, 1, 'revoked group grant applies to backfill too')
  assert.equal(normalizeWhatsAppJid(group), null, 'group identity can never authenticate the linked phone')
  assert.equal(normalizeWhatsAppConversationJid(group), group)
  assert.equal(normalizeWhatsAppConversationJid(legacyGroup), legacyGroup)
  assert.equal(normalizeWhatsAppConversationJid(['*', 'g.us'].join('@')), null)
  assert.deepEqual(validateWhatsAppSourcePatch({ allowedChatJids: [group, legacyGroup] }), { allowedChatJids: [group, legacyGroup] })
  stopAllWhatsAppBridges()

  const reconnect = fixture()
  reconnect.setCurrent({ ...reconnect.source, allowSelfChat: true })
  await reconnect.start()
  reconnect.sockets[0].ev.emit('connection.update', { connection: 'open' })
  reconnect.sockets[0].ev.emit('messages.upsert', { type: 'notify', messages: [message(contact, 'replay-id')] })
  const priorOutbound = await sendWhatsAppMessage(reconnect.source.id, own, 'Approved synthetic reconnect fixture')
  close(reconnect.sockets[0], 515)
  assert.equal(getWhatsAppBridgeRuntime(reconnect.source.id)[0].phase, 'reconnecting')
  assert.equal(isWhatsAppBridgeReady(reconnect.source.id), false)
  assert.equal(reconnect.timers[0].delay, 1000)
  reconnect.timers[0].callback(); await tick()
  reconnect.sockets[1].ev.emit('connection.update', { connection: 'open' })
  reconnect.sockets[1].ev.emit('messages.upsert', { type: 'notify', messages: [message(contact, 'replay-id'), message(contact, 'new-id'), message(own, priorOutbound.sentId!, 'Approved synthetic reconnect fixture', true)] })
  assert.equal(reconnect.events.length, 2, 'seen state survives the actual transient reconnect callback')
  assert.equal(reconnect.events.some((event) => event.payload?.messageId === priorOutbound.sentId), false, 'outbound self-chat suppression also survives transient reconnect')
  for (let attempt = 2; attempt <= 5; attempt++) {
    close(reconnect.sockets.at(-1), 515)
    reconnect.timers.at(-1)!.callback(); await tick()
    reconnect.sockets.at(-1).ev.emit('connection.update', { connection: 'open' })
  }
  close(reconnect.sockets.at(-1), 515)
  assert.equal(getWhatsAppBridgeRuntime(reconnect.source.id)[0].phase, 'error')
  assert.match(getWhatsAppBridgeRuntime(reconnect.source.id)[0].lastError ?? '', /limit reached/)
  assert.equal(reconnect.sockets.length, 6, 'automatic retry is bounded')
  stopAllWhatsAppBridges()

  const failure = fixture()
  await failure.start(); failure.sockets[0].ev.emit('connection.update', { connection: 'open' })
  close(failure.sockets[0], 410); failure.failLoad(); failure.timers[0].callback(); await tick()
  assert.equal(getWhatsAppBridgeRuntime(failure.source.id)[0].phase, 'error')
  assert.match(getWhatsAppBridgeRuntime(failure.source.id)[0].lastError ?? '', /Offline retry fixture/, 'retry failure is visible')
  stopAllWhatsAppBridges()

  const qr = fixture()
  await qr.start()
  qr.sockets[0].ev.emit('connection.update', { qr: 'synthetic-qr-not-a-credential' })
  const snapshot = getWhatsAppBridgeRuntime(qr.source.id)[0]
  assert.equal(snapshot.qr, 'synthetic-qr-not-a-credential', 'a remount can retrieve the active QR snapshot')
  assert.ok(snapshot.qrExpiresAt! > snapshot.updatedAt)
  qr.timers[0].callback()
  assert.equal(getWhatsAppBridgeRuntime(qr.source.id)[0].qr, null, 'QR expires without persisting QR material')
  qr.sockets[0].ev.emit('connection.update', { connection: 'open' })
  close(qr.sockets[0], 515)
  const pending = qr.timers.at(-1)!
  stopAllWhatsAppBridges()
  assert.equal(pending.cancelled, true, 'manual/global stop cancels reconnect scheduling')
  pending.callback(); await tick()
  assert.equal(qr.sockets.length, 1, 'even a queued stale retry cannot recreate a stopped socket')

  const loggedOut = fixture()
  await loggedOut.start(); loggedOut.sockets[0].ev.emit('connection.update', { connection: 'open' }); close(loggedOut.sockets[0], 401)
  assert.equal(getWhatsAppBridgeRuntime(loggedOut.source.id)[0].phase, 'disconnected')
  assert.equal(loggedOut.timers.length, 0)
  stopAllWhatsAppBridges()

  const resume = fixture()
  await assert.rejects(() => startWhatsAppBridge(resume.source, '/tmp/unused-offline-fixture', () => {}, () => {}, () => {},
    async () => ({ state: { creds: { registered: false } }, saveCreds: async () => null, credentialUpdatedAt: () => null }),
    { ...resume.deps, requireRegistered: true }), /not registered/)
  assert.equal(resume.sockets.length, 0, 'auto-resume refuses unregistered credentials before opening a socket or QR session')
  stopAllWhatsAppBridges()

  assert.equal(normalizeWhatsAppJid('+15551230001'), contact)
  assert.equal(normalizeWhatsAppJid(['15551230000:12', 's.whatsapp.net'].join('@')), own)
  assert.equal(normalizeWhatsAppJid(['123456', 'g.us'].join('@')), null)
  assert.throws(() => validateWhatsAppSourcePatch({ allowedChatJids: ['Untrusted display name'] }), /exact/)
  assert.throws(() => validateWhatsAppSourcePatch({ allowSelfChat: 'yes' }), /Invalid/)
  assert.deepEqual(validateWhatsAppSourcePatch({ allowedChatJids: ['+15551230001', contact] }), { allowedChatJids: [contact] })
} finally { stopAllWhatsAppBridges() }
console.log('Offline WhatsApp lifecycle, identity, scope, self-chat, backfill, reconnect, cancellation and QR regression passed')
