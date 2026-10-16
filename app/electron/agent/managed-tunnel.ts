// Supervise a cloudflared/ngrok child so a TensorFold remote-control server (see remote-control.ts) can
// be reached from outside the LAN without the user babysitting the tunnel process. The spawn factory
// is INJECTED (mirrors workspace-git's GitRunner and ssh-connection's clientFactory), so the whole
// supervise/restart/parse loop is unit-testable with a fake spawner and no real subprocess or network.
//
// Security: a tunnel exposes a local port to the public internet. Only start one when the user has
// explicitly opted in (integrator gates this behind a setting); default the local bind to 127.0.0.1.

import { spawn } from 'node:child_process'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'

// What the controller needs from a spawned tunnel child. Keeping it minimal lets tests inject a fake.
export interface TunnelChildHandle {
  // Register a callback for a line (or chunk) of the child's stdout/stderr.
  onStdout(cb: (chunk: string) => void): void
  // Register a callback for child exit (code is the OS exit code, or null if killed by signal).
  onExit(cb: (code: number | null) => void): void
  // Terminate the child.
  kill(): void
}

// (cmd, args) -> child handle. Default is a real spawn wrapper; tests pass a fake and never spawn.
export type TunnelSpawner = (cmd: string, args: string[]) => TunnelChildHandle

export type TunnelState = 'idle' | 'starting' | 'connected' | 'restarting' | 'stopped' | 'error'

export interface TunnelStatus {
  state: TunnelState
  url: string | null
  lastError: string | null
  // How many times the child has been (re)spawned this start() cycle. Useful for status surfaces.
  restarts: number
}

export interface ManagedTunnelOptions {
  spawner?: TunnelSpawner
  // The tunnel binary, e.g. 'cloudflared' or 'ngrok'.
  command?: string
  // Build the argv for a given local port. Defaults to a cloudflared quick-tunnel against localhost.
  buildArgs?: (localPort: number) => string[]
  // Backoff schedule (ms) between restarts; capped at the last entry. Injected so tests run instantly.
  backoffMs?: number[]
  // Schedule a delayed restart. Injected so tests can drive timing deterministically.
  setTimer?: (cb: () => void, ms: number) => unknown
  clearTimer?: (handle: unknown) => void
  // Notified on every status transition (state, url, error). Optional.
  onStatus?: (status: TunnelStatus) => void
}

// cloudflared prints the public URL on its own line, e.g.
//   2024-01-01T00:00:00Z INF +-----...| https://random-words-1234.trycloudflare.com |...
// ngrok (log mode) prints url=https://abcd.ngrok-free.app. Pull the first https URL we recognise.
const TUNNEL_URL_RE = /https:\/\/[A-Za-z0-9][A-Za-z0-9.-]*\.(?:trycloudflare\.com|ngrok(?:-free)?\.app|ngrok\.io|cfargotunnel\.com)\b[^\s"'|]*/

/** Pure helper: extract a public tunnel https URL from one line of provider output, or null. */
export function parseTunnelUrl(line: string): string | null {
  const m = TUNNEL_URL_RE.exec(line)
  return m ? m[0] : null
}

function defaultCloudflaredArgs(localPort: number): string[] {
  // Quick tunnel: no account needed, prints a *.trycloudflare.com URL. Bind target stays on loopback.
  return ['tunnel', '--no-autoupdate', '--url', `http://127.0.0.1:${localPort}`]
}

const DEFAULT_BACKOFF = [500, 1000, 2000, 5000, 10000, 30000]

export interface ManagedTunnel {
  // Begin desiring a tunnel for localPort and spawn the child. Idempotent-ish: a second call restarts.
  start(localPort: number): void
  // Stop desiring a tunnel: cancels any pending restart and kills the child. No more auto-restarts.
  stop(): void
  status(): TunnelStatus
}

export function createManagedTunnel(options: ManagedTunnelOptions = {}): ManagedTunnel {
  const spawner = options.spawner ?? defaultSpawner
  const command = options.command ?? 'cloudflared'
  const buildArgs = options.buildArgs ?? defaultCloudflaredArgs
  const backoff = options.backoffMs && options.backoffMs.length ? options.backoffMs : DEFAULT_BACKOFF
  const setTimer = options.setTimer ?? ((cb, ms) => setTimeout(cb, ms))
  const clearTimer = options.clearTimer ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>))

  let desired = false
  let localPort = 0
  let child: TunnelChildHandle | null = null
  let timer: unknown = null
  let restarts = 0
  // Number of consecutive restarts without a successful 'connected', for backoff selection.
  let failStreak = 0

  let state: TunnelState = 'idle'
  let url: string | null = null
  let lastError: string | null = null

  function emit(): void {
    options.onStatus?.(status())
  }

  function status(): TunnelStatus {
    return { state, url, lastError, restarts }
  }

  function backoffFor(streak: number): number {
    const idx = Math.min(streak, backoff.length - 1)
    return backoff[idx]
  }

  function spawnChild(): void {
    state = restarts === 0 ? 'starting' : 'restarting'
    url = null
    emit()
    restarts += 1
    const current = spawner(command, buildArgs(localPort))
    child = current

    current.onStdout((chunk) => {
      // A child may hand us multiple lines in one chunk; scan each.
      for (const line of String(chunk).split(/\r?\n/)) {
        if (!url) {
          const found = parseTunnelUrl(line)
          if (found) {
            url = found
            state = 'connected'
            failStreak = 0
            lastError = null
            emit()
          }
        }
      }
    })

    current.onExit((code) => {
      // Ignore exits from a stale child (e.g. one we already replaced or killed during stop()).
      if (child !== current) return
      child = null
      if (!desired) {
        state = 'stopped'
        emit()
        return
      }
      // Unexpected exit while we still want the tunnel: record + schedule a capped-backoff restart.
      if (!url) failStreak += 1
      else failStreak = 0
      lastError = code == null ? 'Tunnel process was terminated.' : `Tunnel process exited with code ${code}.`
      state = 'error'
      url = null
      emit()
      scheduleRestart()
    })
  }

  function scheduleRestart(): void {
    if (!desired) return
    clearPendingTimer()
    const delay = backoffFor(failStreak)
    timer = setTimer(() => {
      timer = null
      if (!desired) return
      spawnChild()
    }, delay)
  }

  function clearPendingTimer(): void {
    if (timer != null) {
      clearTimer(timer)
      timer = null
    }
  }

  function start(port: number): void {
    localPort = port
    desired = true
    restarts = 0
    failStreak = 0
    lastError = null
    url = null
    clearPendingTimer()
    // Kill any pre-existing child from a prior start() so we never leak a tunnel.
    if (child) {
      const stale = child
      child = null
      try { stale.kill() } catch { /* best-effort */ }
    }
    spawnChild()
  }

  function stop(): void {
    desired = false
    clearPendingTimer()
    const current = child
    child = null
    if (current) {
      try { current.kill() } catch { /* best-effort */ }
    }
    state = 'stopped'
    url = null
    emit()
  }

  return { start, stop, status }
}

/** Real spawn wrapper. Lazy use of node:child_process; tests inject a fake and never reach this. */
export function defaultSpawner(cmd: string, args: string[]): TunnelChildHandle {
  const proc = spawn(cmd, args, {
    detached: detachedProcessGroup(),
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  return {
    onStdout(cb) {
      // cloudflared prints the URL on stderr; ngrok varies. Watch both so the parser sees it.
      proc.stdout?.on('data', (d: Buffer) => cb(d.toString('utf8')))
      proc.stderr?.on('data', (d: Buffer) => cb(d.toString('utf8')))
    },
    onExit(cb) {
      proc.once('exit', (code) => cb(code))
      proc.once('error', () => cb(null))
    },
    kill() {
      terminateChildProcess(proc, { processGroup: true })
    },
  }
}
