import assert from 'node:assert/strict'
import { buildAssistantControlSnapshot } from './connector-spine'

const source = { id: 'wa_fixture', label: 'Private account', enabled: true, paired: true, phoneNumber: null, lastQrAt: null, lastEventAt: null, lastError: null, createdAt: 1, updatedAt: 1 }
function health(phase: string | null) {
  return buildAssistantControlSnapshot([], undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined,
    { sources: [source], lastSyncAt: null }, undefined, undefined, undefined, undefined,
    { whatsappRuntime: phase ? [{ sourceId: source.id, phase, lastError: null, updatedAt: 2 }] : [], now: 3 } as any).connectors.find((item) => item.id === 'whatsapp')!
}

assert.equal(health('connected').status, 'connected', 'An open QR account must not require unrelated Business API credentials')
assert.ok(health('connected').tools.some((item) => item.name === 'message_compose'), 'The connected QR route exposes its real message capability')
assert.notEqual(health(null).status, 'connected', 'Saved pairing alone is not a live session after app restart')
assert.notEqual(health('reconnecting').status, 'connected', 'Reconnect is not yet a usable connection')
assert.notEqual(health('error').status, 'connected', 'A failed QR account never looks ready')
