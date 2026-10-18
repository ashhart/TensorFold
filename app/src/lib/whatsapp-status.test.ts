import assert from 'node:assert/strict'
import { applyWhatsAppRuntimeSnapshot, applyWhatsAppStatusPush } from './whatsapp-status.ts'

const draft = {
  theme: 'dark',
  whatsapp: {
    lastSyncAt: null,
    sources: [
      { id: 'wa1', label: 'Phone', enabled: true, phoneNumber: null, paired: false, qr: null, lastEventAt: null, lastError: 'connecting', createdAt: 1, updatedAt: 1 },
    ],
  },
} as any

const next = applyWhatsAppStatusPush(draft, { sourceId: 'wa1', paired: true }, 42)
assert.notEqual(next, draft)
assert.equal(next.theme, 'dark', 'unrelated in-progress draft fields are preserved')
assert.equal(next.whatsapp.sources[0].paired, true)
assert.equal(next.whatsapp.sources[0].lastError, null)
assert.equal(next.whatsapp.sources[0].lastEventAt, 42)
assert.equal(draft.whatsapp.sources[0].paired, false, 'the input draft is not mutated')

const credentials = applyWhatsAppStatusPush(draft, { sourceId: 'wa1', credentialRotatedAt: 123, phoneNumber: '+15551230000' }, 99)
assert.equal(credentials.whatsapp.sources[0].phoneNumber, '+15551230000')
assert.equal(credentials.whatsapp.sources[0].credentialRotatedAt, 123)
assert.equal(credentials.whatsapp.sources[0].lastError, 'connecting', 'a credential-only push cannot clear connection feedback')
assert.equal(credentials.whatsapp.sources[0].lastEventAt, null, 'credential and status updates are not new inbound messages')
const disconnected = applyWhatsAppStatusPush(next, { sourceId: 'wa1', enabled: false, lastError: null }, 99)
assert.equal(disconnected.whatsapp.sources[0].enabled, false)
assert.equal(disconnected.whatsapp.sources[0].paired, true, 'saved credentials and live disconnect intent are distinct')
assert.equal(disconnected.whatsapp.sources[0].lastEventAt, 42)

const pairing = { sourceId: 'wa1', phase: 'pairing', qr: 'synthetic-qr', qrExpiresAt: 100, phoneNumber: null, reconnectAttempt: 0, nextReconnectAt: null, lastError: null, updatedAt: 1 } as const
const connected = { ...pairing, phase: 'connected' as const, qr: null, qrExpiresAt: null, updatedAt: 2 }
let runtime = applyWhatsAppRuntimeSnapshot({}, pairing)
runtime = applyWhatsAppRuntimeSnapshot(runtime, connected)
assert.equal(applyWhatsAppRuntimeSnapshot(runtime, pairing), runtime, 'late initial snapshot cannot restore a QR after a newer connected push')
assert.equal(applyWhatsAppRuntimeSnapshot(runtime, { ...pairing, updatedAt: 2 }), runtime, 'equal revisions cannot override the accepted push')

console.log('WhatsApp status draft merge tests passed')
