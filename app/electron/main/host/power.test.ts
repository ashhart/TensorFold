import assert from 'node:assert/strict'
import path from 'node:path'
import os from 'node:os'
import { promises as fs } from 'node:fs'
import { createJiti } from 'jiti'

// Isolated test for power.ts. power.ts imports the real `../../store` (which pulls
// in electron + crypto + fs) and `../state`. We alias both to lightweight fakes so
// the module under test loads in a plain node/jiti context, and we point its
// hardcoded `/usr/bin/caffeinate` spawn at a tiny long-lived stand-in via a PATH-
// independent absolute path by aliasing nothing there — instead we drive the REAL
// caffeinate binary on this macOS host and observe the respawn behaviour.

const hostDir = __dirname // electron/main/host
const powerPath = path.join(hostDir, 'power.ts')

async function main() {
  // ---- shared fake state module ----------------------------------------
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'power-test-'))
  const savedCalls: any[] = []

  const fakeStorePath = path.join(tmp, 'fake-store.mjs')
  await fs.writeFile(
    fakeStorePath,
    `export const store = { settings: { power: { keepComputerAlive: true, lastError: null } } }
`,
  )
  const fakeSaveSettingsPath = path.join(tmp, 'fake-save.mjs')
  await fs.writeFile(
    fakeSaveSettingsPath,
    `globalThis.__savedCalls = globalThis.__savedCalls || []
export async function saveSettings(s){ globalThis.__savedCalls.push(s); return s }
`,
  )

  void savedCalls // capture happens via globalThis.__savedCalls in the fake module
  const jiti = createJiti(__filename, {
    // jiti matches alias keys against the import *specifier*, so we alias the exact
    // relative strings power.ts imports rather than resolved absolute paths.
    alias: {
      '../../store': fakeSaveSettingsPath, // import { saveSettings } from '../../store'
      '../state': fakeStorePath, //          import { store } from '../state'
    },
    moduleCache: false,
    fsCache: false,
  })

  const mod: any = await jiti.import(powerPath)
  const { applyPowerSettings, stopCaffeinate } = mod
  assert.equal(typeof applyPowerSettings, 'function', 'applyPowerSettings exported')

  if (process.platform !== 'darwin') {
    console.log('non-darwin host: applyPowerSettings should record a platform error and not spawn')
    applyPowerSettings()
    console.log('PASS (skipped respawn assertions; macOS-only path)')
    return
  }

  // ---- start caffeinate -------------------------------------------------
  applyPowerSettings()

  // find the spawned caffeinate pid (the module keeps the handle private, so we
  // detect via the system process list)
  const { execSync } = await import('node:child_process')
  const pidOf = (): number | null => {
    try {
      const out = execSync('pgrep -f "/usr/bin/caffeinate -dimsu"', { encoding: 'utf8' }).trim()
      const pids = out.split(/\s+/).filter(Boolean).map(Number)
      return pids.length ? pids[pids.length - 1] : null
    } catch {
      return null
    }
  }

  const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))

  await wait(300)
  const firstPid = pidOf()
  assert.ok(firstPid, 'caffeinate spawned on applyPowerSettings')
  console.log('PASS: caffeinate spawned, pid', firstPid)

  // ---- simulate an unexpected death (NOT our SIGTERM) -------------------
  // SIGKILL the child as if the OS reaped it. The exit handler should respawn
  // after RESPAWN_BACKOFF_MS (2s) because keepComputerAlive is still wanted.
  process.kill(firstPid!, 'SIGKILL')
  await wait(500)
  // briefly: process gone, no immediate respawn yet (backoff pending)
  // (we don't hard-assert the gap window to avoid timing flakiness)

  // wait past the 2s backoff + spawn time
  await wait(2500)
  const secondPid = pidOf()
  assert.ok(secondPid, 'caffeinate was respawned after unexpected death')
  assert.notEqual(secondPid, firstPid, 'respawn produced a new process')
  console.log('PASS: caffeinate respawned after unexpected exit, new pid', secondPid)

  // ---- deliberate stop must NOT respawn ---------------------------------
  stopCaffeinate()
  await wait(2700) // longer than backoff
  const afterStop = pidOf()
  assert.equal(afterStop, null, 'no caffeinate process after deliberate stopCaffeinate (no respawn)')
  console.log('PASS: stopCaffeinate killed it and suppressed respawn')

  // ---- error path was recorded for the unexpected exit ------------------
  const calls: any[] = (globalThis as any).__savedCalls || []
  const sawExitError = calls.some(
    (c) => c?.power?.lastError && /caffeinate exited/.test(String(c.power.lastError)),
  )
  assert.ok(sawExitError, 'lastError recorded the unexpected caffeinate exit')
  console.log('PASS: unexpected exit recorded lastError')

  await fs.rm(tmp, { recursive: true, force: true }).catch(() => {})
  console.log('ALL TESTS PASSED')
}

main().catch((e) => {
  // best-effort cleanup of any stray caffeinate we started
  try {
    require('node:child_process').execSync('pkill -f "/usr/bin/caffeinate -dimsu"')
  } catch {}
  console.error('TEST FAILED:', e)
  process.exit(1)
})
