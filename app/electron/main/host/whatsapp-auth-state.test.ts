// Run: jiti electron/main/host/whatsapp-auth-state.test.ts
import assert from 'node:assert/strict'
import { existsSync, lstatSync, mkdirSync, readFileSync, statSync, symlinkSync, writeFileSync } from 'node:fs'
import { mkdtemp } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

const home = await mkdtemp(path.join(os.tmpdir(), 'vontra-whatsapp-auth-'))
process.env.VONTRA_HOME = home

const B = await import('@whiskeysockets/baileys')
const { overwriteAndUnlink, readProtectedJson, writeProtectedJson } = await import('../../store.ts')
const {
  authFolderFor,
  beginWhatsAppAuthStatePurge,
  createProtectedWhatsAppAuthState,
  endWhatsAppAuthStatePurge,
  protectedAuthFileFor,
  shredWhatsAppAuthState,
} = await import('./whatsapp-auth-state.ts')

const jid = ['15551230000', 's.whatsapp.net'].join('@')
const sourceId = 'wa_disk'
let now = 1_700_000_000_000
const deps = {
  readProtected: readProtectedJson,
  writeProtected: writeProtectedJson,
  shredFile: overwriteAndUnlink,
  now: () => now,
}

const auth = await createProtectedWhatsAppAuthState(B, home, sourceId, deps)
auth.state.creds.registered = true
auth.state.creds.me = { id: jid, name: 'Protected test' }
const secretSession = Buffer.from('SESSION-PLAINTEXT-SENTINEL')
await auth.state.keys.set({ session: { 'device:1': secretSession } })
assert.equal(await auth.saveCreds(), now, 'a real paired credential update receives the host timestamp')

const protectedFile = protectedAuthFileFor(home, sourceId)
const raw = readFileSync(protectedFile, 'utf8')
assert.match(raw, /safe-json:v1:/, 'auth is a TensorFold Studio app-key protected JSON envelope')
assert.doesNotMatch(raw, /SESSION-PLAINTEXT-SENTINEL|15551230000|advSecretKey/, 'neither creds nor Signal keys exist plaintext on disk')
assert.equal(statSync(protectedFile).mode & 0o777, 0o600, 'protected auth file is owner-only')
assert.equal(statSync(authFolderFor(home, sourceId)).mode & 0o777, 0o700, 'auth directory is owner-only')

const reopened = await createProtectedWhatsAppAuthState(B, home, sourceId, deps)
const restored = await reopened.state.keys.get('session', ['device:1'])
assert.ok(Buffer.isBuffer(restored['device:1']), 'BufferJSON revives stored Signal bytes as Buffer')
assert.deepEqual(restored['device:1'], secretSession, 'protected Signal key round-trips exactly')
assert.equal(reopened.state.creds.me.id, jid, 'protected credentials round-trip')
assert.equal(reopened.credentialUpdatedAt(), now, 'credential age survives restart')

// A label/status-only save does not call saveCreds and therefore cannot advance auth age.
now += 60_000
assert.equal(reopened.credentialUpdatedAt(), 1_700_000_000_000, 'status-only time passage does not refresh credential age')

// A wipe invalidates the handle before draining. It waits for a write that already entered,
// shreds those bytes afterwards, and permanently rejects events the old socket emits later.
const raceId = 'wa_race'
let holdWrites = false
let releaseWrite!: () => void
let markWriteEntered!: () => void
const writeGate = new Promise<void>((resolve) => { releaseWrite = resolve })
const writeEntered = new Promise<void>((resolve) => { markWriteEntered = resolve })
const raceDeps = {
  ...deps,
  writeProtected: async (file: string, value: unknown) => {
    if (holdWrites) {
      markWriteEntered()
      await writeGate
    }
    await writeProtectedJson(file, value)
  },
}
const raceAuth = await createProtectedWhatsAppAuthState(B, home, raceId, raceDeps)
raceAuth.state.creds.registered = true
holdWrites = true
const enteredSave = raceAuth.saveCreds()
await writeEntered
let wipeSettled = false
const racingWipe = shredWhatsAppAuthState(home, raceId).then(() => { wipeSettled = true })
await new Promise((resolve) => setTimeout(resolve, 0))
assert.equal(wipeSettled, false, 'wipe drains a credential write that had already entered storage')
await assert.rejects(
  () => createProtectedWhatsAppAuthState(B, home, raceId, deps),
  /no longer active|late write/i,
  'a concurrent reconnect cannot claim a fresh writer generation while wipe is in progress',
)
releaseWrite()
const [saveResult, wipeResult] = await Promise.allSettled([enteredSave, racingWipe])
assert.equal(saveResult.status, 'rejected', 'the in-flight stale handle observes generation invalidation')
assert.equal(wipeResult.status, 'fulfilled', 'expected stale-write rejection does not fail the credential wipe')
assert.equal(existsSync(authFolderFor(home, raceId)), false, 'drained bytes are shredded after the in-flight write settles')
await assert.rejects(() => raceAuth.saveCreds(), /no longer active|late write/i, 'late creds.update cannot recreate wiped auth')
await assert.rejects(
  () => raceAuth.state.keys.set({ session: { late: Buffer.from('LATE-KEY') } }),
  /no longer active|late write/i,
  'late Signal-key update cannot recreate wiped auth',
)
await assert.rejects(
  () => raceAuth.state.keys.get('session', ['late']),
  /no longer active|late write/i,
  'a wiped stale session cannot continue reading in-memory Signal keys',
)
assert.equal(existsSync(authFolderFor(home, raceId)), false, 'all post-wipe old-handle updates leave auth absent')

// Global local-state purge uses the same synchronous invalidation barrier for every source.
const globalId = 'wa_global'
const globalAuth = await createProtectedWhatsAppAuthState(B, home, globalId, deps)
globalAuth.state.creds.registered = true
await globalAuth.saveCreds()
beginWhatsAppAuthStatePurge()
try {
  await assert.rejects(
    () => createProtectedWhatsAppAuthState(B, home, 'wa_during_purge', deps),
    /no longer active|late write/i,
    'global purge blocks new auth handles for its full credential-wipe window',
  )
  await shredWhatsAppAuthState(home, globalId)
} finally {
  endWhatsAppAuthStatePurge()
}
await assert.rejects(() => globalAuth.saveCreds(), /no longer active|late write/i, 'pre-purge creds.update stays invalid after the global barrier reopens')
assert.equal(existsSync(authFolderFor(home, globalId)), false, 'global purge leaves old auth absent')

// Safely migrate Baileys' historical plaintext multi-file layout, including an id whose ':'
// was irreversibly normalized to '-' in its filename. The compatibility lookup preserves it.
const legacyId = 'wa_legacy'
const legacyFolder = authFolderFor(home, legacyId)
mkdirSync(legacyFolder, { recursive: true, mode: 0o700 })
const legacyCreds = B.initAuthCreds() as any
legacyCreds.registered = true
legacyCreds.me = { id: jid, name: 'Legacy test' }
writeFileSync(path.join(legacyFolder, 'creds.json'), JSON.stringify(legacyCreds, B.BufferJSON.replacer), { mode: 0o600 })
const legacySession = Buffer.from('LEGACY-SESSION-SENTINEL')
writeFileSync(path.join(legacyFolder, 'session-user-1.json'), JSON.stringify(legacySession, B.BufferJSON.replacer), { mode: 0o600 })

const migrated = await createProtectedWhatsAppAuthState(B, home, legacyId, deps)
assert.equal(existsSync(path.join(legacyFolder, 'creds.json')), false, 'legacy plaintext creds are shredded after verified migration')
assert.equal(existsSync(path.join(legacyFolder, 'session-user-1.json')), false, 'legacy plaintext Signal key is shredded after verified migration')
const migratedRaw = readFileSync(protectedAuthFileFor(home, legacyId), 'utf8')
assert.doesNotMatch(migratedRaw, /LEGACY-SESSION-SENTINEL|15551230000/, 'migrated state exists only as ciphertext')
const migratedSession = await migrated.state.keys.get('session', ['user:1'])
assert.deepEqual(migratedSession['user:1'], legacySession, 'legacy filename normalization remains readable after migration')
assert.ok((migrated.credentialUpdatedAt() ?? 0) > 0, 'legacy creds mtime seeds an honest age instead of pretending they are new')

// Source removal recursively shreds regular files and unlinks symlinks without following them.
const outside = path.join(home, 'outside-do-not-touch.txt')
writeFileSync(outside, 'OUTSIDE-SENTINEL', { mode: 0o600 })
symlinkSync(outside, path.join(legacyFolder, 'hostile-link'))
await shredWhatsAppAuthState(home, legacyId)
assert.equal(existsSync(legacyFolder), false, 'source auth folder is removed')
assert.equal(readFileSync(outside, 'utf8'), 'OUTSIDE-SENTINEL', 'credential shredding never follows an in-tree symlink')
await assert.rejects(
  () => migrated.saveCreds(),
  /no longer active|late write/i,
  'a creds.update queued by the stopped old socket cannot recreate state after wipe',
)
assert.equal(existsSync(legacyFolder), false, 'late save rejection leaves the wiped auth folder absent')

// A source directory that is itself a symlink is rejected for writes and only unlinked on wipe.
const linkedSource = 'wa_link'
const externalFolder = path.join(home, 'external-auth-target')
mkdirSync(externalFolder, { mode: 0o700 })
writeFileSync(path.join(externalFolder, 'keep.txt'), 'KEEP-EXTERNAL', { mode: 0o600 })
symlinkSync(externalFolder, authFolderFor(home, linkedSource))
await assert.rejects(
  () => createProtectedWhatsAppAuthState(B, home, linkedSource, deps),
  /unsafe WhatsApp credential path/,
  'protected auth refuses a symlinked source directory',
)
await shredWhatsAppAuthState(home, linkedSource)
assert.equal(lstatSync(externalFolder).isDirectory(), true)
assert.equal(readFileSync(path.join(externalFolder, 'keep.txt'), 'utf8'), 'KEEP-EXTERNAL', 'wiping a symlinked source does not touch its target')
assert.equal(existsSync(authFolderFor(home, linkedSource)), false, 'the hostile source symlink itself is removed')

console.log('protected WhatsApp auth disk, migration, rotation-age, and symlink-safe wipe tests passed')
