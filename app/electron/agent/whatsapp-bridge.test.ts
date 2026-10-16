// Run: jiti electron/agent/whatsapp-bridge.test.ts
import assert from 'node:assert/strict'
import { makeWhatsAppBridgeSource, whatsappEventFromMessage, whatsappSenderFor, markWhatsAppSeen, whatsappPairingErrorPatch, resolveBaileysModule, silentBaileysLogger, startWhatsAppBridge } from './whatsapp-bridge.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('whatsapp-bridge tests:')

// JIDs carry an '@' (user@server); build them so the private-leak guard does not
// mistake the test fixture for an email address.
const JID = ['15551234567', 's.whatsapp.net'].join('@')

{
  // A fresh source is unpaired with no phone number and a stable id.
  const s = makeWhatsAppBridgeSource('Work', 1000)
  assert.match(s.id, /^wa_/)
  assert.equal(s.paired, false)
  assert.equal(s.phoneNumber, null)
  assert.equal(s.credentialRotatedAt, null)
  assert.equal(s.enabled, true)
  assert.equal(s.label, 'Work')
  ok('makeWhatsAppBridgeSource creates an unpaired, enabled multi-account source')
}

{
  const source = makeWhatsAppBridgeSource('No plaintext fallback')
  await assert.rejects(
    () => startWhatsAppBridge(source, '/tmp/unused', () => {}, () => {}, () => {}),
    /protected credential store is unavailable/,
    'pairing fails closed unless the host injects the protected auth adapter',
  )
  ok('startWhatsAppBridge has no plaintext multi-file auth fallback')
}

{
  // An inbound WhatsApp message -> gray bubble (inbound event).
  const recv = whatsappEventFromMessage({ sourceId: 'wa_1', jid: JID, fromMe: false, text: 'are you coming?', sender: 'Sam', ts: 2000 }, 2000)
  assert.equal(recv.direction, 'inbound')
  assert.equal(recv.action, 'message_received')
  assert.equal(recv.connectorId, 'whatsapp')
  assert.equal(recv.actor, 'Sam')
  assert.equal(recv.threadKey, `whatsapp:wa_1:${JID}`)

  // A message you sent -> blue bubble (outbound event).
  const sent = whatsappEventFromMessage({ sourceId: 'wa_1', jid: JID, fromMe: true, text: 'on my way', sender: 'You', ts: 3000 }, 3000)
  assert.equal(sent.direction, 'outbound')
  assert.equal(sent.action, 'message_sent')
  ok('whatsappEventFromMessage: fromMe drives outbound/inbound bubble direction')
}

{
  // The WhatsApp message id must ride in the payload so events dedup against replays
  // (the bridge is push-based; baileys can re-deliver on reconnect).
  const e = whatsappEventFromMessage({ sourceId: 'wa_1', jid: JID, fromMe: false, text: 'hi', sender: 'Sam', ts: 1, messageId: 'wamid.ABC' })
  assert.equal((e.payload as any)?.messageId, 'wamid.ABC')
  assert.ok(e.payloadKeys.includes('messageId'))
  ok('whatsappEventFromMessage carries the message id for dedup')
}

{
  // Group messages must attribute the author (key.participant / pushName), never the group JID.
  const GROUP = ['120363000', 'g.us'].join('@')
  const PART = ['15559999999', 's.whatsapp.net'].join('@')
  assert.equal(whatsappSenderFor(GROUP, 'Sam', PART), 'Sam', 'prefers the display name')
  assert.equal(whatsappSenderFor(GROUP, undefined, PART), PART, 'falls back to the participant JID, not the group')
  assert.equal(whatsappSenderFor(GROUP, undefined, undefined), GROUP, 'group JID only as last resort')
  assert.equal(whatsappSenderFor(JID, undefined, undefined), JID, 'direct chat: the chat JID is the sender')
  ok('whatsappSenderFor attributes group messages to the participant, not the group')
}

{
  // The per-session seen-set must survive a reconnect (the bridge reuses sessions.get(id).seen
  // rather than a fresh Set), so a replay delivered after reconnect is still dropped.
  const session = { seen: new Set<string>() }
  assert.equal(markWhatsAppSeen(session, 'wamid.OLD'), true, 'first delivery is new')
  assert.equal(markWhatsAppSeen(session, 'wamid.OLD'), false, 'replay of the same id is dropped')
  // Simulate a reconnect carrying the existing set forward (what startWhatsAppBridge does).
  const reconnected = { seen: session.seen }
  assert.equal(markWhatsAppSeen(reconnected, 'wamid.OLD'), false, 'replay after reconnect still dropped (set carried over)')
  assert.equal(markWhatsAppSeen(reconnected, 'wamid.NEW'), true, 'a genuinely new message still ingests')
  ok('markWhatsAppSeen dedups within a session and across a carried-over reconnect')
}

{
  // PAIRING ERROR SURFACING: the bridge starts fire-and-forget, so a start failure must become a
  // renderer-pushable status patch (paired:false + a clipped message) — otherwise the UI shows
  // nothing and "Start pairing does nothing". Covers the real baileys-load-failure message, a bare
  // string, and an empty/garbage error (which still yields a non-empty fallback message).
  const fromError = whatsappPairingErrorPatch(new Error('WhatsApp bridge library failed to load (not available in this build).'))
  assert.equal(fromError.paired, false)
  assert.match(String(fromError.lastError), /failed to load/)

  const fromString = whatsappPairingErrorPatch('connection refused')
  assert.equal(fromString.lastError, 'connection refused')

  const fromEmpty = whatsappPairingErrorPatch(undefined)
  assert.equal(fromEmpty.paired, false)
  assert.ok(fromEmpty.lastError && fromEmpty.lastError.length > 0, 'an empty error still produces a clear message')

  // The message is clipped so a giant stack trace can't bloat the persisted/pushed status.
  const long = whatsappPairingErrorPatch(new Error('x'.repeat(1000)))
  assert.ok((long.lastError ?? '').length <= 300, 'lastError is clipped to a bounded length')
  ok('whatsappPairingErrorPatch turns a start failure into a paired:false status with a clipped message')
}

{
  // ESM-INTEROP: baileys is pure ESM. `await import()` returns the Module namespace where the
  // NAMED exports (makeWASocket / useMultiFileAuthState / DisconnectReason / …) sit on the
  // namespace, while `.default` is the makeWASocket FUNCTION alone. resolveBaileysModule must
  // keep the namespace (so the named exports the bridge destructures survive) — the old
  // `.default ?? namespace` picked the bare function, whose `.makeWASocket` is undefined, which
  // is exactly why pairing threw "library failed to load" even though baileys loaded fine.
  const makeWASocket = () => ({}) // stand-in for the default-exported factory
  const realNamespace = {
    default: makeWASocket, // baileys' default export = the bare factory function
    makeWASocket,
    useMultiFileAuthState: () => ({}),
    DisconnectReason: { loggedOut: 401 },
    fetchLatestBaileysVersion: () => ({ version: [2, 3000, 0] }),
  }
  const resolved = resolveBaileysModule(realNamespace)
  assert.equal(typeof resolved.makeWASocket, 'function', 'keeps makeWASocket from the namespace')
  assert.equal(typeof resolved.useMultiFileAuthState, 'function', 'keeps the NAMED useMultiFileAuthState (lost if .default were chosen)')
  assert.equal(typeof resolved.fetchLatestBaileysVersion, 'function', 'keeps fetchLatestBaileysVersion')
  assert.equal(resolved.DisconnectReason?.loggedOut, 401, 'keeps DisconnectReason')

  // INTEROP DOUBLE-WRAP fallback: if a future bundler/interop nests the whole API under
  // `.default` and the namespace itself has no makeWASocket, unwrap `.default`.
  const wrapped = { default: { makeWASocket, useMultiFileAuthState: () => ({}) } }
  const unwrapped = resolveBaileysModule(wrapped)
  assert.equal(typeof unwrapped.makeWASocket, 'function', 'unwraps .default when the namespace lacks makeWASocket')
  assert.equal(typeof unwrapped.useMultiFileAuthState, 'function', 'unwrapped form still carries the named exports')

  // EMPTY STUB: a bundler stub (no makeWASocket anywhere) resolves to something whose
  // makeWASocket is not a function, so loadBaileys()'s guard can fail loud + clear.
  const stub = resolveBaileysModule({})
  assert.notEqual(typeof stub?.makeWASocket, 'function', 'an empty stub yields no callable makeWASocket (guard then throws)')
  ok('resolveBaileysModule keeps the ESM namespace (named exports), unwraps a .default double-wrap, and rejects an empty stub')
}

{
  // SILENT LOGGER: baileys (>=7) calls logger.child() and the level methods during makeWASocket;
  // a bare `{ level: 'silent' }` throws "logger.child is not a function" and aborts pairing before
  // any QR. silentBaileysLogger must expose the full pino-shaped no-op surface, and child() must
  // return a logger with the same surface (baileys chains .child().child()).
  const log = silentBaileysLogger()
  for (const m of ['trace', 'debug', 'info', 'warn', 'error', 'fatal'] as const) {
    assert.equal(typeof log[m], 'function', `logger.${m} is a callable no-op`)
    assert.doesNotThrow(() => log[m]({ a: 1 }, 'msg'), `logger.${m}() is a no-op that never throws`)
  }
  assert.equal(typeof log.child, 'function', 'logger.child is a function (the method whose absence broke pairing)')
  const kid = log.child({ class: 'socket' })
  assert.equal(typeof kid.child, 'function', 'child() returns a logger that itself can child()')
  assert.equal(typeof kid.info, 'function', 'the child logger has the level methods too')
  assert.doesNotThrow(() => kid.child({}).warn('deep'), 'chained children stay no-op loggers')
  ok('silentBaileysLogger exposes the full pino-shaped surface baileys needs (child + level methods)')
}

console.log(`\n${passed} checks passed.`)
