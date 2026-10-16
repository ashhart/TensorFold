import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { encryptWithKey, decryptWithKey, loadOrCreateKeySync, parseExternalSecretKey, resolveLocalSecretKeySync } from './local-crypto.ts'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

console.log('local-crypto tests:')

{
  // App-managed key encryption replaces the macOS Keychain (safeStorage), so the
  // app never triggers the "wants to use your confidential information" prompt.
  const key = Buffer.alloc(32, 7)
  const secret = 'sk-vontra-test-API-KEY-äöü-🔐'
  const blob = encryptWithKey(secret, key)
  assert.notEqual(blob, secret)
  assert.equal(decryptWithKey(blob, key), secret)
  ok('encrypt → decrypt round-trips arbitrary UTF-8 with the app key')
}

{
  // Same plaintext encrypts to different ciphertext each time (random IV) but both decode.
  const key = Buffer.alloc(32, 3)
  const a = encryptWithKey('hello', key)
  const b = encryptWithKey('hello', key)
  assert.notEqual(a, b)
  assert.equal(decryptWithKey(a, key), 'hello')
  assert.equal(decryptWithKey(b, key), 'hello')
  ok('random IV makes ciphertext non-deterministic yet decryptable')
}

{
  // GCM auth tag rejects the wrong key and tampered ciphertext (no silent garbage).
  const key = Buffer.alloc(32, 1)
  const wrong = Buffer.alloc(32, 2)
  const blob = encryptWithKey('top secret', key)
  assert.throws(() => decryptWithKey(blob, wrong))
  const tampered = Buffer.from(blob, 'base64')
  tampered[tampered.length - 1] ^= 0xff
  assert.throws(() => decryptWithKey(tampered.toString('base64'), key))
  // A legacy safeStorage blob (not our format) must throw, not return garbage,
  // so the store's try/catch falls back to "re-enter your key" cleanly.
  assert.throws(() => decryptWithKey('bm90LW91ci1mb3JtYXQ=', key))
  ok('wrong key / tampered / foreign blobs throw instead of returning garbage')
}

await (async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-key-'))
  const keyPath = path.join(dir, '.secretkey')
  const k1 = loadOrCreateKeySync(keyPath)
  assert.equal(k1.length, 32)
  const k2 = loadOrCreateKeySync(keyPath)
  assert.deepEqual(k1, k2) // stable across reads
  const stat = await fs.stat(keyPath)
  assert.equal(stat.mode & 0o777, 0o600) // owner-only, never world-readable
  const secret = 'persisted-secret'
  assert.equal(decryptWithKey(encryptWithKey(secret, k1), k2), secret)
  await fs.rm(dir, { recursive: true, force: true })
  ok('loadOrCreateKeySync makes a stable 0600 key and survives a reload')
})()

await (async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-external-key-'))
  const keyPath = path.join(dir, '.secretkey')
  const external = Buffer.alloc(32, 42)
  const resolved = resolveLocalSecretKeySync(keyPath, external.toString('base64'))
  assert.equal(resolved.mode, 'external-env-key')
  assert.deepEqual(resolved.key, external)
  await assert.rejects(() => fs.stat(keyPath), /ENOENT/, 'external-key mode does not create a profile-adjacent key')
  assert.throws(() => parseExternalSecretKey('not base64'), /exactly 32 bytes/)
  assert.throws(() => parseExternalSecretKey(Buffer.alloc(31).toString('base64')), /exactly 32 bytes/)
  assert.throws(
    () => resolveLocalSecretKeySync(path.join(dir, 'required-key'), undefined, true),
    /VONTRA_REQUIRE_EXTERNAL_SECRET_KEY.*VONTRA_SECRET_KEY_BASE64 is missing/,
    'managed require-external policy fails closed instead of silently creating a profile key',
  )

  const oldProfileKey = Buffer.alloc(32, 11)
  await fs.writeFile(keyPath, oldProfileKey, { mode: 0o600 })
  const migrating = resolveLocalSecretKeySync(keyPath, external.toString('base64'))
  assert.deepEqual(migrating.profileFallbackKey, oldProfileKey, 'existing profile key is read only as a migration fallback')
  await fs.rm(dir, { recursive: true, force: true })
  ok('external deployment keys validate strictly and do not get copied into the profile')
})()

await (async () => {
  // Exclusive create: a key already on disk (written by another concurrently-booting process)
  // must be ADOPTED, never clobbered — otherwise the two processes encrypt under different
  // keys and make each other's at-rest data undecryptable (gap-audit TOCTOU).
  const { randomBytes } = await import('node:crypto')
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-key2-'))
  const keyPath = path.join(dir, '.secretkey')
  const pre = randomBytes(32)
  await fs.writeFile(keyPath, pre, { mode: 0o600 })
  const adopted = loadOrCreateKeySync(keyPath)
  assert.deepEqual(adopted, pre, 'adopts the existing key rather than overwriting it')
  await fs.rm(dir, { recursive: true, force: true })
  ok('loadOrCreateKeySync adopts a pre-existing key (no clobber under concurrent boot)')
})()

{
  // Crypto-refactor migration contract: the store treats a `safe:v1:` blob that the
  // app key can't decode as a LEGACY blob (from the pre-refactor safeStorage/Keychain
  // era) and migrates it via the safeStorage fallback. This pins the property that
  // detection relies on: a foreign ciphertext throws under the app key, so the store
  // can distinguish "legacy" from "ours" and re-save instead of dropping the secret.
  const appKey = Buffer.alloc(32, 9)
  const foreignKey = Buffer.alloc(32, 1)
  const foreignBlob = encryptWithKey('legacy-api-key', foreignKey) // Keychain-era shape
  // The store's isLegacySecretBlob predicate is: decryptWithKey throws.
  assert.throws(() => decryptWithKey(foreignBlob, appKey))
  // And our own blobs round-trip with the app key (so they are NOT misclassified).
  const ourBlob = encryptWithKey('current-api-key', appKey)
  assert.equal(decryptWithKey(ourBlob, appKey), 'current-api-key')
  ok('migration contract: a foreign-keyed blob throws under the app key (detectable as legacy)')
}

await (async () => {
  // Key-persistence invariant (the "settings lost on every reload" bug): a key
  // created by loadOrCreateKeySync MUST land on disk so a later process can read it
  // back. If two calls return DIFFERENT keys for the same path, every file encrypted
  // with the first would be unreadable after restart. We pin: create -> persist ->
  // fresh read returns the SAME bytes.
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-key-persist-'))
  const keyPath = path.join(dir, '.secretkey')
  const created = loadOrCreateKeySync(keyPath)
  // The bytes on disk must match what was returned...
  const onDisk = await fs.readFile(keyPath)
  assert.deepEqual(onDisk, created)
  // ...and a second, independent call (simulating a new process) must agree.
  const reread = loadOrCreateKeySync(keyPath)
  assert.deepEqual(reread, created)
  // A blob encrypted with the created key must decrypt with the re-read key.
  assert.equal(decryptWithKey(encryptWithKey('survives-restart', created), reread), 'survives-restart')
  await fs.rm(dir, { recursive: true, force: true })
  ok('loadOrCreateKeySync persists the key to disk; a re-read returns the same bytes')
})()

console.log(`\n${passed} checks passed.`)
