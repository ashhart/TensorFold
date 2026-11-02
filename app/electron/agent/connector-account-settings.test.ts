import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { normalizeIMessageSettingsForSave, normalizeMicrosoftMailSettingsForSave } from './settings-save'
import { mergeSettingsSecrets, redactSettings, SECRET_MASK } from './settings-secrets'

const dir = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-connector-settings-'))
process.env.VONTRA_HOME = dir
try {
  const { defaultSettings, saveSettings, loadSettings, writeProtectedJson } = await import('../store')
  const current = defaultSettings()
  current.imessage = { ...current.imessage, enabled: true, transport: 'bluebubbles', blueBubblesUrl: 'https://messages.example.com',
    blueBubblesPassword: 'SYNTHETIC_BRIDGE_SECRET', allowedChatIds: 'iMessage;-;friend@example.com' }
  const redacted = redactSettings(current)
  assert.equal(redacted.imessage.blueBubblesPassword, SECRET_MASK)
  assert.equal(mergeSettingsSecrets(redacted, current).imessage.blueBubblesPassword, current.imessage.blueBubblesPassword)
  redacted.imessage.blueBubblesUrl = 'https://changed.example.com'
  assert.equal(mergeSettingsSecrets(redacted, current).imessage.blueBubblesPassword, '', 'saved secrets cannot follow an edited destination')
  await saveSettings(current)
  assert.doesNotMatch(await readFile(path.join(dir, 'settings.json'), 'utf8'), /SYNTHETIC_BRIDGE_SECRET/)
  assert.equal((await loadSettings()).imessage.blueBubblesPassword, 'SYNTHETIC_BRIDGE_SECRET')
  current.imessage.lastSyncAt = 42
  current.imessage.cursor = { lastPosition: 10, lastId: 'fixture', updatedAt: 1 } as any
  const changed = normalizeIMessageSettingsForSave({ ...current, imessage: { ...current.imessage, allowedChatIds: 'iMessage;+;chat999' } }, current)
  assert.equal(changed.cursor, undefined); assert.equal(changed.lastSyncAt, null)
  const graph: any = { id: 'ms-fixture', kind: 'microsoft_graph', user: 'account@example.com', microsoftRevision: 2, enabled: true, label: 'Original' }
  current.email.sources = [graph]
  const stale = normalizeMicrosoftMailSettingsForSave({ ...current, email: { ...current.email, sources: [] } }, current)
  assert.deepEqual(stale.sources, [graph], 'stale settings cannot silently remove credentials or account membership')
  const forged = normalizeMicrosoftMailSettingsForSave({ ...current, email: { ...current.email, sources: [{ ...graph, kind: 'eml_dir', user: 'other@example.com' }] } }, current)
  assert.deepEqual(forged.sources, [graph], 'a source id cannot be retargeted around the sign-in boundary')
  assert.throws(() => normalizeMicrosoftMailSettingsForSave({ ...current, email: { ...current.email, sources: [{ ...graph, id: 'forged' }] } }, current), /Microsoft sign-in/)
  const file = path.join(dir, 'state', 'microsoft-mail.protected.json')
  await writeProtectedJson(file, { version: 1, accounts: [{ accessToken: 'SYNTHETIC_ACCESS_SECRET', refreshToken: 'SYNTHETIC_REFRESH_SECRET' }] })
  const raw = await readFile(file, 'utf8')
  assert.match(raw, /safe-json:v1:/); assert.doesNotMatch(raw, /SYNTHETIC_ACCESS_SECRET|SYNTHETIC_REFRESH_SECRET/)
  console.log('actual storage encryption, renderer masking, URL binding, stale settings and connector account ownership passed')
} finally { await rm(dir, { recursive: true, force: true }) }
