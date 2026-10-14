import assert from 'node:assert/strict'
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const home = mkdtempSync(path.join(os.tmpdir(), 'vontra-external-store-'))
const externalKey = Buffer.alloc(32, 29)
const profileKey = Buffer.alloc(32, 31)
process.env.VONTRA_HOME = home
process.env.VONTRA_SECRET_KEY_BASE64 = externalKey.toString('base64')
process.env.VONTRA_REQUIRE_EXTERNAL_SECRET_KEY = '1'

const { decryptWithKey, encryptWithKey } = await import('./local-crypto.ts')
const store = await import('./store.ts')
mkdirSync(path.join(home, 'state'), { recursive: true })
writeFileSync(path.join(home, '.secretkey'), profileKey, { mode: 0o600 })

const oldEnvelope = {
  encrypted: `safe-json:v1:${encryptWithKey(JSON.stringify([{ id: 't1', title: 'profile-key thread' }]), profileKey)}`,
}
writeFileSync(path.join(home, 'state', 'threads.json'), JSON.stringify(oldEnvelope), { mode: 0o600 })
const threads = await store.loadThreads()
assert.equal(threads[0]?.title, 'profile-key thread')
const rewrappedEnvelope = JSON.parse(readFileSync(path.join(home, 'state', 'threads.json'), 'utf8'))
const rewrappedPayload = String(rewrappedEnvelope.encrypted).slice('safe-json:v1:'.length)
assert.deepEqual(JSON.parse(decryptWithKey(rewrappedPayload, externalKey)), threads, 'fallback-readable whole-file state is rewrapped under the external key')
assert.throws(() => decryptWithKey(rewrappedPayload, profileKey), 'the old profile key no longer decrypts active state')

const defaults = store.defaultSettings()
defaults.providers[0].apiKey = `safe:v1:${encryptWithKey('old-profile-api-key', profileKey)}`
writeFileSync(path.join(home, 'settings.json'), JSON.stringify(defaults), { mode: 0o600 })
const settings = await store.loadSettings()
assert.equal(settings.providers[0].apiKey, 'old-profile-api-key')
const settingsRaw = JSON.parse(readFileSync(path.join(home, 'settings.json'), 'utf8'))
const migratedApiKey = String(settingsRaw.providers[0].apiKey).slice('safe:v1:'.length)
assert.equal(decryptWithKey(migratedApiKey, externalKey), 'old-profile-api-key', 'per-field profile-key secrets rewrap under the external key')
assert.throws(() => decryptWithKey(migratedApiKey, profileKey))

const posture = store.localSecretStorageStatus()
assert.equal(posture.keyMode, 'external-env-key')
assert.equal(posture.externalKeyRequired, true)
assert.equal(posture.copiedProfileDecryptable, true, 'a retained old .secretkey keeps the transitioned profile in warning posture')
assert.equal(process.env.VONTRA_SECRET_KEY_BASE64, undefined, 'managed key was scrubbed before child processes can inherit it')

console.log('store-external-key.test.ts passed')
