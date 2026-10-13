import assert from 'node:assert/strict'
import { defaultSettings } from '../store.ts'
import { assertDedicatedWhatsAppSourceRemoval, normalizeChangedRemoteDestinations } from './settings-save.ts'

const current = defaultSettings()
current.remoteClient = { ...current.remoteClient, enabled: false, serverUrl: 'https://stale.invalid', authToken: 'token' }
current.remoteOrchestrators = [{
  id: 'old', name: 'Old', serverUrl: 'https://stale.invalid', authToken: 'token', enabled: false,
  lastConnectedAt: null, lastError: null, createdAt: 1, updatedAt: 1,
}]

const calls: string[] = []
const normalize = async (raw: string) => {
  calls.push(raw)
  if (raw.includes('stale')) throw new Error('unsafe stale destination')
  return raw.replace(/\/$/, '')
}

const unrelated = await normalizeChangedRemoteDestinations({ ...current, theme: 'dark' }, current, normalize)
assert.equal(unrelated.theme, 'dark')
assert.equal(calls.length, 0, 'an unrelated save does not revalidate unchanged stale destinations')

await assert.rejects(
  () => normalizeChangedRemoteDestinations({ ...current, remoteClient: { ...current.remoteClient, enabled: true } }, current, normalize),
  /unsafe stale destination/,
  'newly enabling a stale destination still fails closed',
)

const changed = {
  ...current,
  remoteOrchestrators: [{ ...current.remoteOrchestrators[0], serverUrl: 'https://new.example.test/' }],
}
const normalized = await normalizeChangedRemoteDestinations(changed, current, normalize)
assert.equal(normalized.remoteOrchestrators[0].serverUrl, 'https://new.example.test')
assert.ok(calls.includes('https://new.example.test/'), 'a changed destination is validated and canonicalized')

const whatsappSource = { id: 'wa-1', label: 'Support', enabled: true, paired: true, createdAt: 1, updatedAt: 1 } as any
const withWhatsapp = { ...current, whatsapp: { ...current.whatsapp, sources: [whatsappSource] } }
assert.doesNotThrow(() => assertDedicatedWhatsAppSourceRemoval({ ...withWhatsapp, theme: 'light' }, withWhatsapp), 'unrelated settings saves preserve WhatsApp rows')
assert.doesNotThrow(() => assertDedicatedWhatsAppSourceRemoval({ ...withWhatsapp, whatsapp: { ...withWhatsapp.whatsapp, sources: [{ ...whatsappSource, label: 'Renamed' }] } }, withWhatsapp), 'editing an existing row remains allowed')
assert.throws(
  () => assertDedicatedWhatsAppSourceRemoval({ ...withWhatsapp, whatsapp: { ...withWhatsapp.whatsapp, sources: [] } }, withWhatsapp),
  /dedicated Remove account control/i,
  'whole-settings save cannot bypass native confirmation and credential shredding by dropping a row',
)

console.log('settings remote destination transaction tests passed')
