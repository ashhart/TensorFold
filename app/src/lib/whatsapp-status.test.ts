import assert from 'node:assert/strict'
import { applyWhatsAppStatusPush } from './whatsapp-status.ts'

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

console.log('WhatsApp status draft merge tests passed')
