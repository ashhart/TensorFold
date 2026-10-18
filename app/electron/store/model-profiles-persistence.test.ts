import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { defaultEngineModelProfile, updateEngineModelProfiles } from '../../shared/model-profiles'
process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-model-profiles-'))
const { loadSettings, saveSettings, closeSessionSearchIndexes } = await import('../store')
try {
  const base = await loadSettings()
  const profile = { ...defaultEngineModelProfile(8192), id: 'small', name: 'Small context' }
  const a = updateEngineModelProfiles(base, { model: 'example/model-a', expectedRevision: 0, activeProfileId: profile.id, profiles: [profile] })
  const b = updateEngineModelProfiles(a, { model: 'example/model-b', expectedRevision: 0, activeProfileId: 'large', profiles: [{ ...profile, id: 'large', name: 'Large context', contextTokens: 32768 }] })
  await saveSettings(b)
  const loaded = await loadSettings()
  assert.deepEqual(loaded.engineModelProfiles, b.engineModelProfiles, 'two model profiles and selections survive real restart persistence')
  await saveSettings({ ...loaded, theme: 'dark' })
  assert.deepEqual((await loadSettings()).engineModelProfiles, b.engineModelProfiles, 'unrelated theme save preserves profiles')
  const settingsPath = path.join(process.env.VONTRA_HOME!, 'settings.json')
  const raw = JSON.parse(readFileSync(settingsPath, 'utf8'))
  raw.engineModelProfiles[0].profiles[0].contextTokens = -100
  writeFileSync(settingsPath, JSON.stringify(raw))
  const damaged = await loadSettings()
  assert.deepEqual(damaged.engineModelProfiles, [])
  assert.deepEqual(damaged.engineModelProfilesRecovery, raw.engineModelProfiles)
  assert.match(damaged.engineModelProfilesProblem ?? '', /invalid/)
  await saveSettings(damaged)
  assert.deepEqual((await loadSettings()).engineModelProfilesRecovery, raw.engineModelProfiles, 'original damaged values remain recoverable across unrelated writes')
  await assert.rejects(() => saveSettings({ ...loaded, engineModelProfiles: [{ ...b.engineModelProfiles![0], profiles: [{ ...profile, temperature: Infinity }] }] }), /Temperature/)
  console.log('Real per-model settings persistence and malformed-data recovery passed')
} finally { closeSessionSearchIndexes() }
