// Run: npx jiti electron/agent/managed-tunnel.test.ts
import assert from 'node:assert'
import { createManagedTunnel, parseTunnelUrl, type TunnelChildHandle, type TunnelSpawner } from './managed-tunnel.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

// A fake spawned child we can drive: emit stdout lines, fire exit, and observe kill().
class FakeChild implements TunnelChildHandle {
  killed = false
  private stdoutCb: ((chunk: string) => void) | null = null
  private exitCb: ((code: number | null) => void) | null = null
  onStdout(cb: (chunk: string) => void): void { this.stdoutCb = cb }
  onExit(cb: (code: number | null) => void): void { this.exitCb = cb }
  kill(): void { this.killed = true }
  emit(line: string): void { this.stdoutCb?.(line) }
  exit(code: number | null): void { this.exitCb?.(code) }
}

// A spawner that records every spawn and hands out controllable children. setTimer is captured so the
// test fires restarts synchronously instead of waiting on real timers.
function harness() {
  const children: FakeChild[] = []
  const spawns: Array<{ cmd: string; args: string[] }> = []
  const spawner: TunnelSpawner = (cmd, args) => {
    spawns.push({ cmd, args })
    const c = new FakeChild()
    children.push(c)
    return c
  }
  const timers: Array<() => void> = []
  const setTimer = (cb: () => void) => { timers.push(cb); return timers.length - 1 }
  const clearTimer = () => { /* no-op for the test */ }
  const runPendingTimers = () => { const due = timers.splice(0); for (const t of due) t() }
  return { children, spawns, spawner, setTimer, clearTimer, runPendingTimers }
}

// parseTunnelUrl pulls a cloudflared URL out of a noisy log line.
{
  const line = '2024-01-01T00:00:00Z INF |  https://random-words-1234.trycloudflare.com  |'
  assert.equal(parseTunnelUrl(line), 'https://random-words-1234.trycloudflare.com')
  assert.equal(parseTunnelUrl('url=https://abcd.ngrok-free.app done'), 'https://abcd.ngrok-free.app')
  assert.equal(parseTunnelUrl('no url here'), null)
  assert.equal(parseTunnelUrl('http://insecure.example.com'), null)
  ok('parseTunnelUrl extracts https tunnel URLs and rejects non-tunnel lines')
}

// start() spawns the tunnel and the public URL is parsed from a scripted stdout line.
{
  const h = harness()
  const tunnel = createManagedTunnel({ spawner: h.spawner, setTimer: h.setTimer, clearTimer: h.clearTimer })
  tunnel.start(8765)
  assert.equal(h.spawns.length, 1, 'one spawn on start')
  assert.equal(h.spawns[0].cmd, 'cloudflared')
  assert.ok(h.spawns[0].args.some((a) => a.includes('127.0.0.1:8765')), 'binds the local port on loopback')
  assert.equal(tunnel.status().state, 'starting')
  assert.equal(tunnel.status().url, null)

  h.children[0].emit('INF Connection registered https://demo-9.trycloudflare.com')
  assert.equal(tunnel.status().state, 'connected')
  assert.equal(tunnel.status().url, 'https://demo-9.trycloudflare.com')
  assert.equal(tunnel.status().lastError, null)
  ok('start spawns the tunnel and parses the public URL from stdout')
}

// A child exit while desired triggers an auto-restart (spawn called again).
{
  const h = harness()
  const tunnel = createManagedTunnel({ spawner: h.spawner, setTimer: h.setTimer, clearTimer: h.clearTimer })
  tunnel.start(8765)
  h.children[0].emit('https://live-1.trycloudflare.com')
  assert.equal(tunnel.status().state, 'connected')

  // Child dies unexpectedly.
  h.children[0].exit(1)
  assert.equal(tunnel.status().state, 'error')
  assert.ok(tunnel.status().lastError && tunnel.status().lastError!.includes('1'), 'records exit code')
  assert.equal(h.spawns.length, 1, 'restart is deferred to a timer, not immediate')

  // Fire the scheduled restart.
  h.runPendingTimers()
  assert.equal(h.spawns.length, 2, 'exit while desired triggers a respawn')
  assert.equal(tunnel.status().state, 'restarting')

  // The fresh child reconnects.
  h.children[1].emit('https://live-2.trycloudflare.com')
  assert.equal(tunnel.status().url, 'https://live-2.trycloudflare.com')
  assert.equal(tunnel.status().state, 'connected')
  ok('child exit while desired auto-restarts the tunnel')
}

// stop() cancels desired + kills, and a subsequent exit does NOT restart.
{
  const h = harness()
  const tunnel = createManagedTunnel({ spawner: h.spawner, setTimer: h.setTimer, clearTimer: h.clearTimer })
  tunnel.start(8765)
  const first = h.children[0]
  first.emit('https://x.trycloudflare.com')

  tunnel.stop()
  assert.equal(first.killed, true, 'stop kills the child')
  assert.equal(tunnel.status().state, 'stopped')
  assert.equal(tunnel.status().url, null)

  // Even if the killed child reports its exit afterwards, no restart should be scheduled or spawned.
  first.exit(null)
  h.runPendingTimers()
  assert.equal(h.spawns.length, 1, 'stop prevents any further restarts')
  ok('stop cancels desired, kills the child, and blocks further restarts')
}

// A stale child (replaced by start()) exiting must not cause a spurious restart.
{
  const h = harness()
  const tunnel = createManagedTunnel({ spawner: h.spawner, setTimer: h.setTimer, clearTimer: h.clearTimer })
  tunnel.start(8765)
  const first = h.children[0]
  tunnel.start(9000) // replaces the first child
  assert.equal(first.killed, true, 'restarting via start() kills the prior child')
  assert.equal(h.spawns.length, 2)

  first.exit(1) // late exit from the now-stale child
  h.runPendingTimers()
  assert.equal(h.spawns.length, 2, 'stale child exit is ignored (no extra spawn)')
  ok('stale child exit after a restart is ignored')
}

console.log(`\n${passed} checks passed`)
