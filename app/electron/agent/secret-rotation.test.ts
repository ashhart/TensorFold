import assert from 'node:assert/strict'
import { defaultSettings } from '../store'
import { SECRET_STALE_AFTER_MS, secretFreshness, stampSettingsSecretRotations, stampWebhookSecretRotation } from './secret-rotation'

const before = defaultSettings()
before.remoteControl.authToken = 'same-token'
before.remoteControl.secretRotations = { authToken: 10 }
before.whatsapp.sources = [{
  id: 'wa_test', label: 'Bridge', enabled: true, phoneNumber: null, paired: true,
  lastQrAt: null, lastEventAt: null, lastError: null, credentialRotatedAt: 25,
  createdAt: 1, updatedAt: 1,
}]
const next = structuredClone(before)
next.remoteControl.serverName = 'renamed only'
next.whatsapp.sources[0].label = 'Status-only rename'
next.whatsapp.sources[0].credentialRotatedAt = 999_999 // untrusted renderer round-trip
next.botControl.telegram.botToken = 'new-bot-token'
next.slack.sources = [{ id: 's', label: 'Slack', path: '', kind: 'api_token', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, token: 'new-slack-token' }]

const stamped = stampSettingsSecretRotations(next, before, 1_000)
assert.equal(stamped.remoteControl.secretRotations?.authToken, 10, 'non-secret edits do not refresh token age')
assert.equal(stamped.whatsapp.sources[0].credentialRotatedAt, 25, 'status-only edits cannot refresh host-owned WhatsApp credential age')
assert.equal(stamped.botControl.telegram.secretRotations?.botToken, 1_000)
assert.equal(stamped.slack.sources[0].secretRotations?.token, 1_000)

const webhook: any = { id: 'w', secretRef: 'new', secretRotations: {}, updatedAt: 1 }
assert.equal(stampWebhookSecretRotation(webhook, { ...webhook, secretRef: 'old' }, 2_000).secretRotations?.secretRef, 2_000)
assert.equal(secretFreshness('secret', null, 10), 'unknown')
assert.equal(secretFreshness('secret', 10, 10 + SECRET_STALE_AFTER_MS + 1), 'stale')
assert.equal(secretFreshness('secret', 10, 11), 'fresh')
assert.equal(secretFreshness('', 10, 11), 'missing')

console.log('secret rotation metadata tests passed')
