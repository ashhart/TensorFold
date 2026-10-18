import assert from 'node:assert/strict'
import { defaultSettings } from '../store'
import { makeWhatsAppBridgeSource } from './whatsapp-bridge'
import { normalizeWhatsAppSettingsForSave } from './settings-save'

const current = defaultSettings()
const source = { ...makeWhatsAppBridgeSource('Linked fixture'), paired: true, phoneNumber: '+15551230000', enabled: false, allowSelfChat: false, allowedChatJids: [['15551230001', 's.whatsapp.net'].join('@')], credentialRotatedAt: 123, lastError: 'Disconnected by owner' }
current.whatsapp = { sources: [source], lastSyncAt: 123 }
const stale = { ...current, theme: 'light', whatsapp: { sources: [{ ...source, label: 'Stale rename', paired: false, phoneNumber: '+15551239999', enabled: true, allowSelfChat: true, allowedChatJids: [['15551230002', 's.whatsapp.net'].join('@')], credentialRotatedAt: 999, lastError: null }], lastSyncAt: 999 } } as typeof current
assert.deepEqual(normalizeWhatsAppSettingsForSave(stale, current), current.whatsapp, 'stale whole-settings saves preserve exact current scope, disconnect intent, identity, credential timestamp and status')
assert.throws(() => normalizeWhatsAppSettingsForSave({ ...current, whatsapp: { ...current.whatsapp, sources: [] } }, current), /dedicated Remove account/)
assert.throws(() => normalizeWhatsAppSettingsForSave({ ...current, whatsapp: { ...current.whatsapp, sources: [source, { ...source, id: 'fabricated' }] } }, current), /dedicated Add account/)
assert.throws(() => normalizeWhatsAppSettingsForSave({ ...current, whatsapp: { ...current.whatsapp, sources: [{ ...source, allowedChatJids: ['Untrusted name'] }] } }, current), /exact/)
assert.throws(() => normalizeWhatsAppSettingsForSave({ ...current, whatsapp: { ...current.whatsapp, sources: [{ ...source, allowSelfChat: 'yes' as any }] } }, current), /Invalid/)
console.log('WhatsApp stale whole-settings authority and scope validation tests passed')
