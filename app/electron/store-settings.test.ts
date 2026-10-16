import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync, lstatSync, symlinkSync, unlinkSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { decryptWithKey, encryptWithKey } from './local-crypto.ts'

// Isolate persistence to a temp dir (VONTRA_HOME) so we exercise the real save/load round-trip
// without an Electron app or touching ~/.vontra.
process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'vontra-store-'))

const { saveSettings, loadSettings, saveMcpServers, loadMcpServers, saveWebhooks, loadWebhooks, saveThreads, loadThreads, saveContactDirectory, shredPurgedStateFiles, writeArchiveRecord, readProtectedJson, writeProtectedJson, writeTensorFoldStudioProtectedJson, writeTensorFoldStudioStateJson, overwriteAndUnlink, vontraPath, seedProviders, persistSettingsMigration, localSecretStorageStatus } = await import('./store.ts')
const { archiveEntity } = await import('./main/host/archive.ts')
const { store } = await import('./main/state.ts')
const { emptyContactDirectory } = await import('./agent/contact-directory.ts')

async function main() {
  const base = await loadSettings() // no file yet → defaults
  const provId = base.providers[0].id
  const settings = {
    ...base,
    providers: base.providers.map((p, i) =>
      i === 0 ? { ...p, apiKey: 'sk-PLAINTEXT-SECRET-123', discoveredModels: ['model-a', 'model-b'] } : p,
    ),
  }
  await saveSettings(settings)

  // settings.json is readable JSON (NOT a wholly-encrypted blob), with the key encrypted in-place.
  const raw = readFileSync(path.join(process.env.VONTRA_HOME!, 'settings.json'), 'utf8')
  assert.ok(!raw.includes('"encrypted"'), 'settings.json is readable, not a {encrypted} blob')
  assert.ok(raw.includes('safe:v1:'), 'API key is encrypted in-place (safe:v1:)')
  assert.ok(!raw.includes('sk-PLAINTEXT-SECRET-123'), 'plaintext API key is NEVER written to disk')
  JSON.parse(raw) // must be valid JSON

  // models.json is a readable catalogue per provider.
  const modelsRaw = readFileSync(path.join(process.env.VONTRA_HOME!, 'models.json'), 'utf8')
  const models = JSON.parse(modelsRaw)
  assert.deepEqual(models[provId], ['model-a', 'model-b'], 'models.json holds the discovered catalogue')

  // load reveals the secret back to plaintext for the app.
  const reloaded = await loadSettings()
  assert.equal(reloaded.providers[0].apiKey, 'sk-PLAINTEXT-SECRET-123', 'secret round-trips (revealed on load)')

  // reload is stable — still readable, still no plaintext.
  await saveSettings(reloaded)
  const raw2 = readFileSync(path.join(process.env.VONTRA_HOME!, 'settings.json'), 'utf8')
  assert.ok(!raw2.includes('"encrypted"') && !raw2.includes('sk-PLAINTEXT-SECRET-123'), 'stays readable + protected across reload')
  assert.ok(existsSync(path.join(process.env.VONTRA_HOME!, 'models.json')))

  // atomic writes leave no stray temp files, and concurrent saves never corrupt the file.
  await Promise.all([
    saveSettings({ ...reloaded, activeModel: 'a' }),
    saveSettings({ ...reloaded, activeModel: 'b' }),
    saveSettings({ ...reloaded, activeModel: 'c' }),
  ])
  const leftovers = readdirSync(process.env.VONTRA_HOME!).filter((f) => f.includes('.tmp-'))
  assert.deepEqual(leftovers, [], 'no temp files left after writes')
  const afterConcurrent = JSON.parse(readFileSync(path.join(process.env.VONTRA_HOME!, 'settings.json'), 'utf8'))
  assert.ok(['a', 'b', 'c'].includes(afterConcurrent.activeModel), 'concurrent saves serialize to a valid, non-corrupt file')

  console.log('store-settings round-trip tests passed')
}

// Regression: a default provider the user DELETES must stay deleted across a reload. The old
// loadSettings seeded a Map with every default then overlaid the saved list (a union by id), so
// deleting a default (openrouter / z.ai / ollama…) silently came back on the next launch.
async function deletedDefaultProvidersStayDeleted() {
  const base = await loadSettings()
  assert.ok(base.providers.some((p) => p.id === 'zai'), 'precondition: z.ai is a seeded default')
  assert.ok(base.providers.some((p) => p.id === 'openrouter'), 'precondition: openrouter is a seeded default')

  // User deletes z.ai + openrouter and saves.
  const pruned = { ...base, providers: base.providers.filter((p) => p.id !== 'zai' && p.id !== 'openrouter') }
  await saveSettings(pruned)

  const reloaded = await loadSettings()
  assert.ok(!reloaded.providers.some((p) => p.id === 'zai'), 'deleted default (z.ai) stays deleted across reload')
  assert.ok(!reloaded.providers.some((p) => p.id === 'openrouter'), 'deleted default (openrouter) stays deleted across reload')
  // Providers the user KEPT are untouched (still present, default fields intact).
  assert.equal(reloaded.providers.find((p) => p.id === 'openai')?.name, 'OpenAI', 'kept default retains its fields')

  // Deleting EVERY provider is honoured too (empty list is authoritative, not re-seeded).
  await saveSettings({ ...base, providers: [] })
  assert.deepEqual((await loadSettings()).providers, [], 'an explicitly-empty provider list is not re-seeded with defaults')

  console.log('deleted-default-provider persistence tests passed')
}

function shippedProviderModelsMergeForward() {
  const defaults = [{ id: 'p', name: 'Provider', baseUrl: 'https://example.test', apiKey: '', models: ['new-model', 'shared'] }]
  const saved = [{ id: 'p', name: 'Renamed', baseUrl: 'https://custom.example.test', apiKey: 'secret', models: ['shared', 'custom-model'] }]
  const [merged] = seedProviders(defaults as any, saved as any)
  assert.equal(merged.name, 'Renamed', 'saved user configuration still wins')
  assert.deepEqual(merged.models, ['new-model', 'shared', 'custom-model'], 'new shipped models merge with saved/custom entries')
  console.log('provider default-model forward-merge test passed')
}

async function requiredMigrationFailureSurfaces() {
  let attempted = 0
  await assert.rejects(
    () => persistSettingsMigration(true, async () => { attempted++; throw new Error('disk full') }),
    /disk full/,
    'a required credential migration rejects instead of swallowing its failed save',
  )
  assert.equal(attempted, 1)
  await persistSettingsMigration(false, async () => { attempted++ })
  assert.equal(attempted, 1, 'no migration means no incidental write')
  console.log('required settings migration failure-surfacing test passed')
}

// Regression (SETTINGS-1): a corrupt/unreadable settings.json must fall back to defaults WITHOUT
// silently and unrecoverably destroying the original — the bytes are backed up first, so providers
// and API keys stay recoverable instead of being overwritten by a defaults save.
async function settingsCorruptIsRecoverable() {
  const p = path.join(process.env.VONTRA_HOME!, 'settings.json')
  const original = '{ corrupt : not valid json ]]]'
  writeFileSync(p, original)
  const loaded = await loadSettings()
  assert.ok(loaded.providers.length > 0, 'corrupt settings.json falls back to defaults (no throw)')
  const baks = readdirSync(process.env.VONTRA_HOME!).filter((f) => f.startsWith('settings.json') && f.endsWith('.bak'))
  assert.ok(baks.length > 0, 'the corrupt original is backed up for recovery, not silently lost')
  assert.ok(readFileSync(path.join(process.env.VONTRA_HOME!, baks[0]), 'utf8').includes('corrupt'), 'backup holds the original bytes')
  assert.ok(localSecretStorageStatus().quarantinedProtectedPaths.includes(p), 'corrupt settings are write-blocked after fallback')
  await assert.rejects(() => saveSettings(loaded), /Refusing to overwrite quarantined state/, 'defaults cannot overwrite corrupt settings')
  assert.equal(readFileSync(p, 'utf8'), original, 'the corrupt live settings bytes remain untouched')
  await overwriteAndUnlink(p)
  await saveSettings(loaded)
  console.log('settings corrupt-recovery tests passed')
}

async function settingsWrongTopLevelIsQuarantined() {
  const p = path.join(process.env.VONTRA_HOME!, 'settings.json')
  const original = '["parse-valid-but-not-settings"]'
  writeFileSync(p, original, { mode: 0o600 })
  const loaded = await loadSettings()
  assert.ok(loaded.providers.length > 0, 'wrong-shape settings fall back to defaults')
  const status = localSecretStorageStatus()
  const index = status.quarantinedProtectedPaths.indexOf(p)
  assert.ok(index >= 0, 'wrong-shape settings enter protected-state quarantine')
  assert.equal(readFileSync(status.quarantineBackupPaths[index], 'utf8'), original, 'wrong-shape settings recovery copy preserves exact bytes')
  await assert.rejects(() => saveSettings(loaded), /Refusing to overwrite quarantined state/)
  assert.equal(readFileSync(p, 'utf8'), original, 'parse-valid wrong-shape settings cannot be overwritten')
  await overwriteAndUnlink(p)
  await saveSettings(loaded)
  console.log('settings top-level-shape quarantine tests passed')
}

// Regression (NESTED-1): an empty MCP-server list must not clobber a populated-but-UNREADABLE file
// (the load fell back to []), but a legitimate delete-all on a readable file still persists.
async function mcpEmptyWipeGuard() {
  const p = vontraPath('state', 'mcp-servers.json')
  await saveMcpServers([{ id: 'srv1', name: 'My MCP', command: 'node', args: [], env: {} } as any])
  assert.equal((await loadMcpServers()).length, 1, 'precondition: one MCP server saved')
  // Legitimate clear: a readable populated file writes through.
  await saveMcpServers([])
  assert.deepEqual(await loadMcpServers(), [], 'delete-all on a readable file persists')
  // Re-populate, corrupt the file, then an empty save must NOT clobber the unreadable file.
  await saveMcpServers([{ id: 'srv2', name: 'Keep', command: 'node', args: [], env: {} } as any])
  writeFileSync(p, 'NOT JSON {{{')
  await saveMcpServers([])
  assert.equal(readFileSync(p, 'utf8'), 'NOT JSON {{{', 'empty save does NOT overwrite an unreadable mcp-servers.json')
  const baks = readdirSync(path.dirname(p)).filter((f) => f.startsWith('mcp-servers.json') && f.endsWith('.bak'))
  assert.ok(baks.length > 0, 'unreadable mcp-servers.json is backed up before being preserved')
  // The fixture intentionally created an unrecoverable file. Model the explicit
  // operator purge before later test cases reuse this store path.
  await overwriteAndUnlink(p)
  await saveMcpServers([])
  console.log('mcp empty-wipe guard tests passed')
}

async function integrationMetadataUsesWholeFileProtection() {
  const mcpPath = vontraPath('state', 'mcp-servers.json')
  const mcpSecret = 'MCP-INNER-SECRET-VALUE'
  await saveMcpServers([{
    id: 'metadata-mcp-id',
    name: 'SENSITIVE MCP SERVER NAME',
    command: '/sensitive/launch/path',
    args: ['--private-tenant=acme'],
    env: { SECRET_ENV_VAR_NAME: mcpSecret },
  } as any])
  const mcpRaw = readFileSync(mcpPath, 'utf8')
  assert.match(mcpRaw, /safe-json:v1:/, 'mcp-servers.json uses a whole-file protected envelope')
  assert.doesNotMatch(mcpRaw, /SENSITIVE MCP SERVER NAME|sensitive\/launch|private-tenant|SECRET_ENV_VAR_NAME|MCP-INNER-SECRET-VALUE/, 'MCP structural metadata and secret layout are not plaintext')
  assert.equal((await loadMcpServers())[0]?.env?.SECRET_ENV_VAR_NAME, mcpSecret, 'whole-file MCP storage preserves per-field secret reveal compatibility')

  const webhookPath = vontraPath('state', 'webhooks.json')
  const webhookSecret = 'WEBHOOK-INNER-SECRET-VALUE'
  await saveWebhooks([{
    id: 'metadata-webhook-id',
    name: 'SENSITIVE WEBHOOK NAME',
    direction: 'outbound',
    url: 'https://hooks.example.test/private-tenant-path',
    path: '/private-inbound-path',
    secretRef: webhookSecret,
    eventKinds: ['agent'],
    approvalRequired: true,
    approvedAt: null,
    enabled: true,
    createdAt: 1,
    updatedAt: 1,
  } as any])
  const webhookRaw = readFileSync(webhookPath, 'utf8')
  assert.match(webhookRaw, /safe-json:v1:/, 'webhooks.json uses a whole-file protected envelope')
  assert.doesNotMatch(webhookRaw, /SENSITIVE WEBHOOK NAME|private-tenant-path|private-inbound-path|WEBHOOK-INNER-SECRET-VALUE/, 'webhook structural metadata and endpoint layout are not plaintext')
  assert.equal((await loadWebhooks())[0].secretRef, webhookSecret, 'whole-file webhook storage preserves per-field secret reveal compatibility')

  // Backward-compatible read migration: use a real safe:v1 value written by the settings store to
  // model the former plaintext-array/per-field-secret format exactly. Loading must reveal the
  // inner secret and replace the old structural plaintext with a safe-json envelope.
  const settingsRaw = JSON.parse(readFileSync(path.join(process.env.VONTRA_HOME!, 'settings.json'), 'utf8'))
  const legacySecretBlob = settingsRaw.providers.find((provider: any) => provider.apiKey)?.apiKey
  assert.match(String(legacySecretBlob), /^safe:v1:/, 'precondition: a real per-field protected value is available')
  writeFileSync(mcpPath, JSON.stringify([{
    id: 'legacy-mcp', name: 'LEGACY MCP STRUCTURE', command: '/legacy/mcp-command', args: [],
    env: { LEGACY_TOKEN_NAME: legacySecretBlob },
  }]), { mode: 0o600 })
  const migratedMcp = await loadMcpServers()
  assert.equal(migratedMcp[0]?.env?.LEGACY_TOKEN_NAME, 'sk-PLAINTEXT-SECRET-123', 'legacy MCP safe:v1 field is revealed after migration')
  assert.doesNotMatch(readFileSync(mcpPath, 'utf8'), /LEGACY MCP STRUCTURE|LEGACY_TOKEN_NAME|legacy\/mcp-command/, 'legacy MCP structural plaintext is removed during read migration')

  writeFileSync(webhookPath, JSON.stringify([{
    id: 'legacy-webhook', name: 'LEGACY WEBHOOK STRUCTURE', direction: 'inbound',
    path: '/legacy-private-path', secretRef: legacySecretBlob, eventKinds: [],
    approvalRequired: false, approvedAt: null, enabled: true, createdAt: 1, updatedAt: 1,
  }]), { mode: 0o600 })
  const migratedWebhooks = await loadWebhooks()
  assert.equal(migratedWebhooks[0].secretRef, 'sk-PLAINTEXT-SECRET-123', 'legacy webhook safe:v1 field is revealed after migration')
  assert.doesNotMatch(readFileSync(webhookPath, 'utf8'), /LEGACY WEBHOOK STRUCTURE|legacy-private-path/, 'legacy webhook structural plaintext is removed during read migration')
  console.log('integration metadata whole-file protection + migration tests passed')
}

// Regression (STATE-3): a security purge must shred memory-collection state files — including an
// UNDECRYPTABLE one (which saveX([]) skips + backs up) and its recovery .bak — while preserving
// config the purge does not clear (MCP servers / webhooks).
async function purgeShredsUndecryptableState() {
  const dir = vontraPath('state')
  mkdirSync(dir, { recursive: true })
  const threads = path.join(dir, 'threads.json')
  writeFileSync(threads, 'GARBAGE-undecryptable-sensitive-bytes')
  writeFileSync(`${threads}.undecryptable-x.bak`, 'GARBAGE-undecryptable-sensitive-bytes')
  const mcp = path.join(dir, 'mcp-servers.json')
  writeFileSync(mcp, '[]')
  await shredPurgedStateFiles()
  assert.ok(existsSync(threads), 'purged collection is left as a clean empty placeholder (not absent)')
  assert.ok(!readFileSync(threads, 'utf8').includes('GARBAGE'), 'the old undecryptable bytes are gone')
  assert.ok(!existsSync(`${threads}.undecryptable-x.bak`), 'its recovery .bak is shredded too (no sensitive duplicate left)')
  assert.ok(existsSync(mcp), 'MCP config is preserved (the purge does not clear it)')

  // A malicious/replaced protected-state path must not turn purge into an overwrite of an
  // arbitrary target outside TensorFold Studio. The symlink itself is removed and replaced with the empty
  // protected placeholder; the external target's bytes remain unchanged.
  const externalDir = mkdtempSync(path.join(os.tmpdir(), 'vontra-shred-external-'))
  const externalFile = path.join(externalDir, 'precious.txt')
  writeFileSync(externalFile, 'EXTERNAL-USER-DATA-MUST-SURVIVE', { mode: 0o600 })
  unlinkSync(threads)
  symlinkSync(externalFile, threads)
  await shredPurgedStateFiles()
  assert.equal(readFileSync(externalFile, 'utf8'), 'EXTERNAL-USER-DATA-MUST-SURVIVE', 'protected-file shredding never follows an external symlink target')
  assert.equal(lstatSync(threads).isSymbolicLink(), false, 'the state-path symlink is replaced with a local protected placeholder')
  console.log('purge shred-state tests passed')
}

async function roomMetadataIsProtectedAndMigrated() {
  const parts = ['rooms', 'project-a', 'room-a', 'room.json']
  const p = vontraPath(...parts)
  mkdirSync(path.dirname(p), { recursive: true, mode: 0o700 })
  writeFileSync(p, JSON.stringify({ prompt: 'LEGACY-ROOM-PLAINTEXT-SECRET' }), { mode: 0o600 })
  await writeTensorFoldStudioProtectedJson(parts, { prompt: 'ROOM-PROTECTED-SECRET' })
  const raw = readFileSync(p, 'utf8')
  assert.match(raw, /safe-json:v1:/, 'room.json uses the protected JSON envelope')
  assert.doesNotMatch(raw, /LEGACY-ROOM-PLAINTEXT-SECRET|ROOM-PROTECTED-SECRET/, 'room metadata has no plaintext secret duplicate')
  console.log('room metadata protected-storage migration test passed')
}

// Regression: the boot re-write must NEVER overwrite a populated on-disk state file with an
// empty in-memory value (which happens when a load silently falls back to [] on a corrupt or
// transiently-unreadable file). That would turn recoverable corruption into permanent loss.
async function bootOverwriteGuard() {
  const parts = ['state', 'threads.json']
  const p = vontraPath(...parts)

  // Real data on disk (state collections are stored as encrypted envelopes, so compare bytes).
  await writeTensorFoldStudioProtectedJson(parts, [{ id: 't1', title: 'precious thread' }, { id: 't2', title: 'more work' }])
  const before = readFileSync(p, 'utf8')
  assert.ok(before.length > 10, 'precondition: data persisted')

  // Boot tries to re-persist an EMPTY in-memory collection (the load-fallback case).
  const errors: string[] = []
  const realError = console.error
  console.error = (...args: any[]) => { errors.push(args.map(String).join(' ')) }
  try {
    await writeTensorFoldStudioStateJson(parts, [])
  } finally {
    console.error = realError
  }
  assert.equal(readFileSync(p, 'utf8'), before, 'empty boot write must NOT change the populated file')
  // The threatened file is backed up so it stays recoverable.
  assert.ok(readdirSync(path.dirname(p)).some((f) => f.startsWith('threads.json') && f.includes('.bak')), 'existing data backed up before refusing the empty write')
  assert.ok(!errors.some((line) => /Could not decrypt/.test(line)), 'readable populated state is not misreported as undecryptable')

  // A real (non-empty) write still goes through.
  await writeTensorFoldStudioStateJson(parts, [{ id: 't3', title: 'new state' }])
  assert.notEqual(readFileSync(p, 'utf8'), before, 'non-empty writes persist normally (envelope changes)')

  // Empty write to an ABSENT file (fresh install) is fine — it creates the file.
  const freshParts = ['state', 'jobs-fresh.json']
  assert.ok(!existsSync(vontraPath(...freshParts)), 'precondition: fresh file absent')
  await writeTensorFoldStudioStateJson(freshParts, [])
  assert.ok(existsSync(vontraPath(...freshParts)), 'empty write to a fresh (absent) file creates it')

  console.log('boot-overwrite guard tests passed')
}

async function archiveRecordsAreProtected() {
  const secret = 'archive-secret-thread-content'
  await writeArchiveRecord('thread', 'thread-secret', { thread: { id: 'thread-secret', title: secret, timeline: [{ text: secret }] } })
  const raw = readFileSync(vontraPath('archives', 'threads', 'thread-secret.json'), 'utf8')
  assert.ok(raw.includes('"encrypted"'), 'archive record is stored as a protected JSON envelope')
  assert.ok(!raw.includes(secret), 'archive record must not duplicate thread content in plaintext')

  store.threads = [{ id: 'thread-index-secret', title: secret, projectId: null, timeline: [{ text: secret }], createdAt: 1, updatedAt: 1 } as any]
  store.projects = []
  store.rooms = []
  store.tickets = []
  await archiveEntity('thread', 'thread-index-secret')
  const indexRaw = readFileSync(vontraPath('archives', 'index.json'), 'utf8')
  assert.ok(indexRaw.includes('"encrypted"'), 'archive index is stored as a protected JSON envelope')
  assert.ok(!indexRaw.includes(secret), 'archive index must not duplicate archived thread content in plaintext')
  console.log('archive protected-storage tests passed')
}

// Regression: an empty save issued FIRST and a populated save issued SECOND (same tick, no await
// between the two calls) must NOT let the empty save win on disk. The empty-overwrite guard needs
// extra async work (stat + read) before it can register its write, while a populated save has none
// — if the guard's async work runs BEFORE the write joins the per-path serialization queue, the
// populated save (issued later, but reaching the queue first) can register AHEAD of the empty save,
// so the empty save then runs LAST and silently clobbers the newer data. "Last call wins" must hold
// regardless of which path needs more internal awaits to get there.

// Incognito threads are memory-only: saveThreads must strip them so threads.json
// never contains one — that's what makes them vanish on relaunch.
async function incognitoThreadsNeverPersist() {
  await saveThreads([
    { id: 't-normal', title: 'normal' } as any,
    { id: 't-ghost', title: 'incognito', incognito: true } as any,
  ])
  const persisted = await loadThreads()
  assert.deepEqual(persisted.map((t) => t.id), ['t-normal'], 'threads.json keeps only non-incognito threads')
  console.log('incognito persistence tests passed')
}

async function writeOrderRaceGuard() {
  // writeProtectedJson path for whole-file-protected MCP configuration.
  await saveMcpServers([{ id: 'pre', name: 'Pre-existing', command: 'node', args: [], env: {} } as any])
  const p1 = saveMcpServers([])
  const p2 = saveMcpServers([{ id: 'later', name: 'Later (should win)', command: 'node', args: [], env: {} } as any])
  await Promise.all([p1, p2])
  assert.deepEqual((await loadMcpServers()).map((s) => s.id), ['later'], 'mcp-servers.json: the LATER (populated) save wins over an earlier-issued empty save')

  // writeProtectedJson path (threads.json).
  await saveThreads([{ id: 't-pre', title: 'pre' } as any])
  const t1 = saveThreads([])
  const t2 = saveThreads([{ id: 't-later', title: 'later (should win)' } as any])
  await Promise.all([t1, t2])
  assert.deepEqual((await loadThreads()).map((t) => t.id), ['t-later'], 'threads.json: the LATER (populated) save wins over an earlier-issued empty save')

  console.log('write-order race guard tests passed')
}

// Regression: writeProtectedJson's empty-overwrite guard only recognized arrays (.length===0) or
// null/undefined as "empty" — an object-shaped protected collection (ContactDirectory: { records,
// updatedAt }) is NEVER null and never an array, so the guard never engaged for it at all, and a
// freshly-constructed empty directory (e.g. from a failed load) could silently overwrite a real,
// merely-transiently-unreadable contacts.json with no backup.
async function contactsEmptyGuardForObjectShape() {
  const p = vontraPath('state', 'contacts.json')
  await saveContactDirectory({ records: { 'whatsapp:+15551234': { platform: 'whatsapp', handle: '+15551234', name: 'Real Contact', confidence: 'high', updatedAt: Date.now() } as any }, updatedAt: Date.now() })
  writeFileSync(p, 'NOT JSON {{{')
  await saveContactDirectory(emptyContactDirectory())
  assert.equal(readFileSync(p, 'utf8'), 'NOT JSON {{{', 'an empty ContactDirectory save does NOT overwrite an unreadable contacts.json')
  assert.ok(readdirSync(path.dirname(p)).some((f) => f.startsWith('contacts.json') && f.endsWith('.bak')), 'unreadable contacts.json is backed up before being preserved')
  await overwriteAndUnlink(p)
  await saveContactDirectory(emptyContactDirectory())
  console.log('contacts object-shape empty-guard test passed')
}

// Regression: loadSettings()'s plaintext-secret migration trigger (hasPlaintextSecret) only
// inspected providers/botControl/voice/remoteControl/remoteClient/remoteOrchestrators — never the
// email/Slack/Discord/Mattermost/Matrix/Google Chat connector source secrets. A plaintext connector
// secret (manual edit, restored backup) would never trigger the forced re-encrypt-on-load migration.
async function connectorPlaintextMigrationCoverage() {
  const base = await loadSettings()
  const tampered = {
    ...base,
    email: { ...base.email, sources: [{ id: 'e1', label: 'x', path: '', kind: 'imap', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, password: 'plaintext-imap-pw' }] },
    slack: { ...base.slack, sources: [{ id: 's1', label: 'x', path: '', kind: 'api_token', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, token: 'plaintext-slack-token' }] },
    discord: { ...base.discord, sources: [{ id: 'd1', label: 'x', token: 'plaintext-discord-token', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: 1, updatedAt: 1 }] },
    mattermost: { ...base.mattermost, sources: [{ id: 'm1', label: 'x', baseUrl: 'https://mm.example.com', token: 'plaintext-mm-token', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: 1, updatedAt: 1 }] },
    matrix: { ...base.matrix, sources: [{ id: 'mx1', label: 'x', baseUrl: 'https://matrix.example.com', token: 'plaintext-matrix-token', allowedRooms: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: 1, updatedAt: 1 }] },
    googlechat: { ...base.googlechat, sources: [{ id: 'g1', label: 'x', webhookUrl: 'https://chat.googleapis.com/v1/x?token=plaintext-gc-webhook', accessToken: 'plaintext-gc-token', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: 1, updatedAt: 1 }] },
    botControl: { ...base.botControl, whatsapp: { ...base.botControl.whatsapp, appSecret: 'plaintext-meta-app-secret' } },
  } as any
  writeFileSync(path.join(process.env.VONTRA_HOME!, 'settings.json'), JSON.stringify(tampered, null, 2))
  await loadSettings()
  const raw = readFileSync(path.join(process.env.VONTRA_HOME!, 'settings.json'), 'utf8')
  assert.ok(!raw.includes('plaintext-imap-pw'), 'email source password is migrated to encrypted-at-rest')
  assert.ok(!raw.includes('plaintext-slack-token'), 'slack source token is migrated to encrypted-at-rest')
  assert.ok(!raw.includes('plaintext-discord-token'), 'discord source token is migrated to encrypted-at-rest')
  assert.ok(!raw.includes('plaintext-mm-token'), 'mattermost source token is migrated to encrypted-at-rest')
  assert.ok(!raw.includes('plaintext-matrix-token'), 'matrix source token is migrated to encrypted-at-rest')
  assert.ok(!raw.includes('plaintext-gc-webhook') && !raw.includes('plaintext-gc-token'), 'googlechat webhookUrl + accessToken are migrated to encrypted-at-rest')
  assert.ok(!raw.includes('plaintext-meta-app-secret'), 'WhatsApp Meta App Secret is migrated to encrypted-at-rest')
  console.log('connector plaintext-secret migration coverage test passed')
}

// A pre-refactor safeStorage blob cannot be decoded without touching the macOS
// Keychain (which this build intentionally never does). Loading it as an empty UI
// value must not let an unrelated settings save destroy the only ciphertext copy.
async function unresolvedLegacySecretsArePreserved() {
  const p = path.join(process.env.VONTRA_HOME!, 'settings.json')
  const current = await loadSettings()
  const foreign = `safe:v1:${encryptWithKey('legacy-keychain-secret', Buffer.alloc(32, 71))}`
  const seedProvider = current.providers[0] ?? { id: 'legacy-provider', name: 'Legacy provider', baseUrl: 'https://example.test/v1', apiKey: '', models: ['test-model'], kind: 'cloud' as const }
  const firstId = seedProvider.id
  const raw = {
    ...current,
    providers: [seedProvider, ...current.providers.filter((provider) => provider.id !== firstId)]
      .map((provider) => provider.id === firstId ? { ...provider, apiKey: foreign } : provider),
  }
  writeFileSync(p, JSON.stringify(raw, null, 2), { mode: 0o600 })

  const revealed = await loadSettings()
  assert.equal(revealed.providers.find((provider) => provider.id === firstId)?.apiKey, '', 'unreadable legacy ciphertext is not exposed as a secret value')
  const status = localSecretStorageStatus()
  assert.equal(status.legacySafeStorageSecrets, 1)
  assert.ok(status.legacySafeStorageBackupPath && existsSync(status.legacySafeStorageBackupPath), 'the exact original settings file is backed up')
  assert.equal(JSON.parse(readFileSync(p, 'utf8')).providers.find((provider: any) => provider.id === firstId).apiKey, foreign, 'load does not destructively auto-migrate the blob')

  await saveSettings({ ...revealed, theme: revealed.theme === 'dark' ? 'light' : 'dark' })
  assert.equal(JSON.parse(readFileSync(p, 'utf8')).providers.find((provider: any) => provider.id === firstId).apiKey, foreign, 'an unrelated save preserves the unresolved ciphertext verbatim')

  const replacement = {
    ...revealed,
    providers: revealed.providers.map((provider) => provider.id === firstId
      ? { ...provider, apiKey: 'replacement-api-key', serverPassword: 'opencode-password' }
      : provider),
  }
  await saveSettings(replacement)
  const replacedRaw = readFileSync(p, 'utf8')
  assert.ok(!replacedRaw.includes(foreign) && !replacedRaw.includes('replacement-api-key') && !replacedRaw.includes('opencode-password'), 'replacement API key and server password are encrypted with the active key')
  const reloaded = await loadSettings()
  const provider = reloaded.providers.find((item) => item.id === firstId)
  assert.equal(provider?.apiKey, 'replacement-api-key')
  assert.equal(provider?.serverPassword, 'opencode-password')
  assert.equal(localSecretStorageStatus().legacySafeStorageSecrets, 0, 're-entering the credential resolves the recovery warning')

  const mcpPath = vontraPath('state', 'mcp-servers.json')
  const webhookPath = vontraPath('state', 'webhooks.json')
  const legacyMcp = [{ id: 'legacy-mcp-preserve', name: 'Legacy MCP', command: 'node', args: [], env: { TOKEN: foreign } }]
  const legacyWebhook = [{ id: 'legacy-webhook-preserve', name: 'Legacy webhook', direction: 'inbound', path: '/legacy', secretRef: foreign, eventKinds: [], approvalRequired: false, approvedAt: null, enabled: true, createdAt: 1, updatedAt: 1 }]
  writeFileSync(mcpPath, JSON.stringify(legacyMcp), { mode: 0o600 })
  writeFileSync(webhookPath, JSON.stringify(legacyWebhook), { mode: 0o600 })
  const revealedMcp = await loadMcpServers()
  const revealedWebhooks = await loadWebhooks()
  assert.equal(revealedMcp[0]?.env?.TOKEN, '')
  assert.equal(revealedWebhooks[0]?.secretRef, '')
  assert.equal(localSecretStorageStatus().legacySafeStorageSecrets, 2, 'MCP and webhook legacy ciphertext contributes to the recovery status')

  await saveMcpServers([{ ...revealedMcp[0], name: 'Renamed MCP' }])
  await saveWebhooks([{ ...revealedWebhooks[0], name: 'Renamed webhook' }])
  const profileKey = readFileSync(vontraPath('.secretkey'))
  const revealEnvelope = (file: string) => {
    const envelope = JSON.parse(readFileSync(file, 'utf8'))
    return JSON.parse(decryptWithKey(String(envelope.encrypted).slice('safe-json:v1:'.length), profileKey))
  }
  assert.equal(revealEnvelope(mcpPath)[0].env.TOKEN, foreign, 'unrelated MCP save preserves the unresolved env ciphertext')
  assert.equal(revealEnvelope(webhookPath)[0].secretRef, foreign, 'unrelated webhook save preserves the unresolved HMAC ciphertext')

  await saveMcpServers([{ ...revealedMcp[0], env: { TOKEN: 'new-mcp-token' } }])
  await saveWebhooks([{ ...revealedWebhooks[0], secretRef: 'new-webhook-secret' }])
  assert.equal((await loadMcpServers())[0]?.env?.TOKEN, 'new-mcp-token')
  assert.equal((await loadWebhooks())[0]?.secretRef, 'new-webhook-secret')
  assert.equal(localSecretStorageStatus().legacySafeStorageSecrets, 0)
  console.log('legacy safeStorage preservation + provider server-password protection tests passed')
}

async function undecryptableWholeFileIsQuarantined() {
  const p = vontraPath('state', 'threads.json')
  const foreignEnvelope = {
    encrypted: `safe-json:v1:${encryptWithKey(JSON.stringify([{ id: 'lost-thread', title: 'recover me' }]), Buffer.alloc(32, 99))}`,
  }
  writeFileSync(p, JSON.stringify(foreignEnvelope), { mode: 0o600 })
  assert.deepEqual(await loadThreads(), [], 'unknown-key whole-file state is not exposed as garbage')
  const status = localSecretStorageStatus()
  assert.ok(status.quarantinedProtectedPaths.includes(p), 'unknown-key path is visible in the security snapshot')
  assert.ok(status.quarantineBackupPaths.some((backup) => existsSync(backup)), 'a verified recovery copy exists')
  await assert.rejects(
    () => saveThreads([{ id: 'replacement', title: 'must not overwrite' } as any]),
    /Refusing to overwrite quarantined/,
    'even a non-empty save is fail-closed while the only original is quarantined',
  )
  assert.equal(JSON.parse(readFileSync(p, 'utf8')).encrypted, foreignEnvelope.encrypted, 'the live original remains untouched')
  await overwriteAndUnlink(p) // explicit purge/recovery action clears the write block
  await saveThreads([{ id: 'after-purge', title: 'fresh state' } as any])
  assert.equal(localSecretStorageStatus().quarantinedProtectedFiles, 0)
  console.log('undecryptable whole-file quarantine tests passed')
}

async function malformedWholeFileIsQuarantined() {
  const assertQuarantinedWithExactBackup = (p: string, original: string) => {
    const status = localSecretStorageStatus()
    const index = status.quarantinedProtectedPaths.indexOf(p)
    assert.ok(index >= 0, `${path.basename(p)} is visible in the protected-state quarantine`)
    const backup = status.quarantineBackupPaths[index]
    assert.ok(backup && existsSync(backup), `${path.basename(p)} has a verified recovery copy`)
    assert.equal(readFileSync(backup, 'utf8'), original, 'the recovery copy preserves the exact malformed bytes')
  }

  // Exact closure-audit sequence: load falls back, then a populated save must not
  // replace the truncated file merely because the in-memory collection is non-empty.
  const afterLoadPath = vontraPath('state', 'malformed-after-load.json')
  const afterLoadBytes = '{"truncated-secret":'
  writeFileSync(afterLoadPath, afterLoadBytes, { mode: 0o600 })
  assert.deepEqual(await readProtectedJson(afterLoadPath, []), [], 'malformed protected state returns its caller fallback')
  assertQuarantinedWithExactBackup(afterLoadPath, afterLoadBytes)
  await assert.rejects(
    () => writeProtectedJson(afterLoadPath, [{ id: 'replacement-after-load' }]),
    /Refusing to overwrite quarantined protected state/,
    'a populated save after fallback is fail-closed',
  )
  assert.equal(readFileSync(afterLoadPath, 'utf8'), afterLoadBytes, 'the malformed live file remains untouched')

  // A caller is not required to load first: every populated write probes an
  // existing file from inside the serialized write slot before replacing it.
  const directWritePath = vontraPath('state', 'malformed-direct-write.json')
  const directWriteBytes = 'NOT JSON: direct populated save must not win'
  writeFileSync(directWritePath, directWriteBytes, { mode: 0o600 })
  await assert.rejects(
    () => writeProtectedJson(directWritePath, [{ id: 'replacement-without-load' }]),
    /Refusing to overwrite quarantined protected state/,
    'a direct populated save detects and quarantines malformed on-disk state',
  )
  assertQuarantinedWithExactBackup(directWritePath, directWriteBytes)
  assert.equal(readFileSync(directWritePath, 'utf8'), directWriteBytes, 'direct save cannot overwrite malformed bytes')

  // An authenticated envelope can still contain invalid JSON if an older/foreign
  // writer produced it. Successful AES-GCM decryption must not be mistaken for a
  // valid protected-state payload.
  const malformedPayloadPath = vontraPath('state', 'malformed-decrypted-payload.json')
  const profileKey = readFileSync(vontraPath('.secretkey'))
  const malformedPayloadBytes = JSON.stringify({ encrypted: `safe-json:v1:${encryptWithKey('{"still-truncated":', profileKey)}` })
  writeFileSync(malformedPayloadPath, malformedPayloadBytes, { mode: 0o600 })
  assert.deepEqual(await readProtectedJson(malformedPayloadPath, []), [], 'malformed decrypted JSON returns its caller fallback')
  assertQuarantinedWithExactBackup(malformedPayloadPath, malformedPayloadBytes)
  await assert.rejects(
    () => writeProtectedJson(malformedPayloadPath, [{ id: 'replacement-decrypted-payload' }]),
    /Refusing to overwrite quarantined protected state/,
  )

  // Syntax alone is insufficient: collection consumers expect an array and may
  // immediately call .map/.filter. A parse-valid object must not be migrated as if
  // it were a valid collection.
  const wrongShapePath = vontraPath('state', 'wrong-top-level-shape.json')
  const wrongShapeBytes = '{"parseValid":"but not an array"}'
  writeFileSync(wrongShapePath, wrongShapeBytes, { mode: 0o600 })
  assert.deepEqual(await readProtectedJson(wrongShapePath, []), [], 'wrong top-level collection shape returns its fallback')
  assertQuarantinedWithExactBackup(wrongShapePath, wrongShapeBytes)
  await assert.rejects(
    () => writeProtectedJson(wrongShapePath, [{ id: 'replacement-wrong-shape' }]),
    /Refusing to overwrite quarantined protected state/,
  )
  assert.equal(readFileSync(wrongShapePath, 'utf8'), wrongShapeBytes, 'wrong-shape state remains untouched')

  // Direct writes enforce the incoming value's top-level shape even if no loader ran.
  const wrongShapeDirectPath = vontraPath('state', 'wrong-top-level-shape-direct.json')
  writeFileSync(wrongShapeDirectPath, '{}', { mode: 0o600 })
  await assert.rejects(
    () => writeProtectedJson(wrongShapeDirectPath, [{ id: 'array-write' }]),
    /Refusing to overwrite quarantined protected state/,
  )
  assertQuarantinedWithExactBackup(wrongShapeDirectPath, '{}')

  // The higher-level wrapper used by boot/archive/room persistence previously
  // shredded legacy plaintext before writeProtectedJson could validate it.
  const wrapperMalformedParts = ['state', 'malformed-wrapper-write.json']
  const wrapperMalformedPath = vontraPath(...wrapperMalformedParts)
  const wrapperMalformedBytes = '{"wrapper-truncated":'
  writeFileSync(wrapperMalformedPath, wrapperMalformedBytes, { mode: 0o600 })
  await assert.rejects(
    () => writeTensorFoldStudioStateJson(wrapperMalformedParts, [{ id: 'wrapper-replacement' }]),
    /Refusing to overwrite quarantined protected state/,
  )
  assertQuarantinedWithExactBackup(wrapperMalformedPath, wrapperMalformedBytes)
  assert.equal(readFileSync(wrapperMalformedPath, 'utf8'), wrapperMalformedBytes, 'wrapper preserves malformed bytes before any legacy purge')

  const wrapperWrongShapeParts = ['state', 'wrong-shape-wrapper-write.json']
  const wrapperWrongShapePath = vontraPath(...wrapperWrongShapeParts)
  const wrapperWrongShapeBytes = '{"validJson":"wrong collection shape"}'
  writeFileSync(wrapperWrongShapePath, wrapperWrongShapeBytes, { mode: 0o600 })
  await assert.rejects(
    () => writeTensorFoldStudioStateJson(wrapperWrongShapeParts, [{ id: 'wrapper-array-replacement' }]),
    /Refusing to overwrite quarantined protected state/,
  )
  assertQuarantinedWithExactBackup(wrapperWrongShapePath, wrapperWrongShapeBytes)
  assert.equal(readFileSync(wrapperWrongShapePath, 'utf8'), wrapperWrongShapeBytes, 'wrapper preserves wrong-shape bytes before any legacy purge')

  const emptyWrapperWrongShapeParts = ['state', 'wrong-shape-empty-wrapper-write.json']
  const emptyWrapperWrongShapePath = vontraPath(...emptyWrapperWrongShapeParts)
  const emptyWrapperWrongShapeBytes = '{"validJson":"not the expected empty array"}'
  writeFileSync(emptyWrapperWrongShapePath, emptyWrapperWrongShapeBytes, { mode: 0o600 })
  await writeTensorFoldStudioStateJson(emptyWrapperWrongShapeParts, [])
  assertQuarantinedWithExactBackup(emptyWrapperWrongShapePath, emptyWrapperWrongShapeBytes)
  assert.equal(readFileSync(emptyWrapperWrongShapePath, 'utf8'), emptyWrapperWrongShapeBytes, 'empty wrapper write quarantines rather than silently accepting a wrong-shape file')

  for (const p of [afterLoadPath, directWritePath, malformedPayloadPath, wrongShapePath, wrongShapeDirectPath, wrapperMalformedPath, wrapperWrongShapePath, emptyWrapperWrongShapePath]) await overwriteAndUnlink(p)
  assert.equal(localSecretStorageStatus().quarantinedProtectedFiles, 0, 'explicit purge clears every focused-test quarantine')
  console.log('malformed whole-file quarantine tests passed')
}

await main()
await integrationMetadataUsesWholeFileProtection()
await deletedDefaultProvidersStayDeleted()
shippedProviderModelsMergeForward()
await requiredMigrationFailureSurfaces()
await mcpEmptyWipeGuard()
await purgeShredsUndecryptableState()
await roomMetadataIsProtectedAndMigrated()
await settingsCorruptIsRecoverable()
await settingsWrongTopLevelIsQuarantined()
await bootOverwriteGuard()
await archiveRecordsAreProtected()
await writeOrderRaceGuard()
await incognitoThreadsNeverPersist()
await contactsEmptyGuardForObjectShape()
await connectorPlaintextMigrationCoverage()
await unresolvedLegacySecretsArePreserved()
await malformedWholeFileIsQuarantined()
await undecryptableWholeFileIsQuarantined()
