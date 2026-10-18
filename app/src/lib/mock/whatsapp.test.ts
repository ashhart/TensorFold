import assert from 'node:assert/strict'
import { createMockWhatsAppApi } from './whatsapp'
import { pairedSettingsDefaults } from '../paired-settings'
const settings = pairedSettingsDefaults()
let serial = 0
const api = createMockWhatsAppApi(() => settings, () => String(++serial))
const pushes: unknown[] = [], statuses: any[] = []
const off = api.onWhatsAppRuntime(value => pushes.push(value))
const offStatus = api.onWhatsAppStatus(value => statuses.push(value))
const added = await api.addWhatsAppSource('Synthetic account')
assert.equal(added.sources.length, 1)
const sourceId = added.sources[0].id
assert.equal(added.sources[0].paired, false)
assert.equal((await api.whatsappRuntime())[0].qr, null)
added.sources[0].label = 'external mutation'
assert.equal(settings.whatsapp.sources[0].label, 'Synthetic account')
await api.updateWhatsAppSource(sourceId, { enabled: true, allowSelfChat: true, allowedChatJids: ['synthetic-chat@example.test'] })
assert.equal(settings.whatsapp.sources[0].enabled, true)
const started = await api.startWhatsAppPairing(sourceId)
assert.equal(started.started, false)
const runtime = (await api.whatsappRuntime())[0]
assert.equal(runtime.phase, 'error')
assert.match(runtime.lastError!, /browser demo/)
assert.equal(runtime.qr, null)
assert.equal(runtime.phoneNumber, null)
assert.equal(statuses.at(-1).credentialRotatedAt, null)
assert.equal(statuses.at(-1).paired, false)
await assert.rejects(api.updateWhatsAppSource(sourceId, { paired: true } as any), /Unsupported/)
await api.disconnectWhatsApp(sourceId)
assert.equal((await api.whatsappRuntime())[0].phase, 'disconnected')
assert.equal(settings.whatsapp.sources[0].enabled, false)
off(); offStatus()
const count = pushes.length
await api.updateWhatsAppSource(sourceId, { label: 'Another synthetic account' })
assert.equal(pushes.length, count, 'unsubscribe prevents future callbacks')
await api.removeWhatsAppSource(sourceId)
assert.deepEqual(await api.whatsappRuntime(), [])
await assert.rejects(api.startWhatsAppPairing(sourceId), /not found/)
console.log('Offline WhatsApp demo account lifecycle and unavailable pairing passed without any phone connection')
