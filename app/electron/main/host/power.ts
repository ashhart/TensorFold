import { spawn } from 'node:child_process'
import { store } from '../state'
import { persistSettings } from './persistence'

// macOS "keep computer alive" via caffeinate. Self-contained: the child-process
// handle and intent flag are module-local; only the persisted settings are shared.
let caffeinateProcess: ReturnType<typeof spawn> | null = null
let caffeinateWanted = false
let respawnTimer: ReturnType<typeof setTimeout> | null = null
const RESPAWN_BACKOFF_MS = 2000

export function stopCaffeinate(): void {
  caffeinateWanted = false
  if (respawnTimer) {
    clearTimeout(respawnTimer)
    respawnTimer = null
  }
  const proc = caffeinateProcess
  caffeinateProcess = null
  if (proc && !proc.killed) proc.kill('SIGTERM')
}

function updatePowerError(message: string | null): void {
  if (!store.settings?.power) return
  store.settings = { ...store.settings, power: { ...store.settings.power, lastError: message } }
  persistSettings('power.lastError')
}

export function applyPowerSettings(): void {
  if (!store.settings?.power?.keepComputerAlive) {
    stopCaffeinate()
    return
  }
  caffeinateWanted = true
  if (process.platform !== 'darwin') {
    updatePowerError('Keep computer alive uses macOS caffeinate and is only available on macOS.')
    return
  }
  if (caffeinateProcess && !caffeinateProcess.killed) return

  const proc = spawn('/usr/bin/caffeinate', ['-dimsu'], { stdio: 'ignore' })
  caffeinateProcess = proc
  store.settings = {
    ...store.settings,
    power: {
      ...store.settings.power,
      lastStartedAt: Date.now(),
      lastError: null,
    },
  }
  persistSettings('power.lastStartedAt')

  proc.on('error', (err) => {
    if (caffeinateProcess === proc) caffeinateProcess = null
    updatePowerError(err.message || 'Failed to start caffeinate.')
  })
  proc.on('exit', (code, signal) => {
    if (caffeinateProcess === proc) caffeinateProcess = null
    if (!caffeinateWanted || signal === 'SIGTERM') return
    updatePowerError(code == null ? `caffeinate exited: ${signal ?? 'unknown signal'}` : `caffeinate exited with code ${code}`)
    // The user still wants the machine awake but caffeinate died unexpectedly
    // (OS kill, crash, sleep/wake reap). Nothing else re-applies power settings,
    // so respawn after a short backoff to avoid a tight loop if it keeps failing.
    respawnTimer = setTimeout(() => {
      respawnTimer = null
      if (caffeinateWanted && !caffeinateProcess) applyPowerSettings()
    }, RESPAWN_BACKOFF_MS)
    if (typeof respawnTimer.unref === 'function') respawnTimer.unref()
  })
}
