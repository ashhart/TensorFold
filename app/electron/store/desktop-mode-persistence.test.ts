import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const profile = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-desktop-mode-'))
process.env.VONTRA_HOME = profile
const { loadSettings, saveSettings } = await import('../store')
try {
  const fresh = await loadSettings()
  assert.equal(fresh.desktopMode, 'full')
  assert.equal(fresh.desktopOnboardingComplete, false)
  await saveSettings({ ...fresh, desktopMode: 'minimal', desktopOnboardingComplete: true })
  const restarted = await loadSettings()
  assert.equal(restarted.desktopMode, 'minimal')
  assert.equal(restarted.desktopOnboardingComplete, true)
  assert.equal(restarted.remoteControl.enabled, fresh.remoteControl.enabled, 'Minimal mode must not enable a public or paired listener')
  const file = path.join(profile, 'settings.json')
  const raw = JSON.parse(readFileSync(file, 'utf8'))
  writeFileSync(file, JSON.stringify({ ...raw, desktopMode: 'background', desktopOnboardingComplete: 'true' }))
  const malformed = await loadSettings()
  assert.equal(malformed.desktopMode, 'full', 'unknown saved modes recover to a visible Full window')
  assert.equal(malformed.desktopOnboardingComplete, false, 'only an actual completion boolean allows background startup')
  writeFileSync(file, JSON.stringify({ ...raw, desktopMode: undefined, desktopOnboardingComplete: undefined }))
  const old = await loadSettings()
  assert.equal(old.desktopMode, 'full')
  assert.equal(old.desktopOnboardingComplete, false)
} finally { rmSync(profile, { recursive: true, force: true }) }
console.log('Desktop mode durable settings restart and malformed-state recovery tests passed')
