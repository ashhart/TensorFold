import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { streamClaudeCli } from './claude-cli'
import { startCliLogin } from './cli-auth-runtime'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { streamCodexAppServer } from './codex-app-server'
import { streamCodexCli } from './codex-cli'
import { streamCursorAgent } from './cursor-agent-cli'
import { streamGeminiCli } from './gemini-cli'
import { streamPiCli } from './pi-cli'

const waitFor = async (check: () => boolean, timeoutMs = 2_000): Promise<void> => {
  const deadline = Date.now() + timeoutMs
  while (!check()) {
    if (Date.now() >= deadline) throw new Error('condition did not become true before timeout')
    await new Promise((resolve) => setTimeout(resolve, 10))
  }
}

const processExists = (pid: number): boolean => {
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

if (process.platform !== 'win32') {
  const proc = spawn(process.execPath, ['-e', "process.on('SIGTERM', () => {}); process.stdout.write('ready\\n'); setInterval(() => {}, 1000)"], {
    detached: detachedProcessGroup(),
    stdio: ['ignore', 'pipe', 'ignore'],
  })
  await new Promise<void>((resolve) => proc.stdout.once('data', () => resolve()))
  terminateChildProcess(proc, { processGroup: true, graceMs: 50 })
  const result = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
    proc.once('close', (code, signal) => resolve({ code, signal }))
  })
  assert.equal(result.signal, 'SIGKILL', 'a SIGTERM-ignoring CLI group is forcibly reaped')

  // The leader can accept SIGTERM while a descendant ignores it. ChildProcess
  // then reports the leader closed, but that must not cancel the group-wide
  // SIGKILL escalation or the descendant becomes an orphan.
  const dir = mkdtempSync(path.join(tmpdir(), 'vontra-process-group-'))
  const childScript = path.join(dir, 'child.cjs')
  const leaderScript = path.join(dir, 'leader.cjs')
  const pidFile = path.join(dir, 'child.pid')
  const readyFile = path.join(dir, 'child.ready')
  writeFileSync(childScript, [
    "const { writeFileSync } = require('node:fs')",
    "process.on('SIGTERM', () => {})",
    "writeFileSync(process.argv[2], 'ready')",
    'setInterval(() => {}, 1000)',
  ].join('\n'))
  writeFileSync(leaderScript, [
    "const { spawn } = require('node:child_process')",
    "const { writeFileSync } = require('node:fs')",
    'const child = spawn(process.execPath, [process.argv[2], process.argv[4]], { stdio: \'ignore\' })',
    "writeFileSync(process.argv[3], String(child.pid))",
    'setInterval(() => {}, 1000)',
  ].join('\n'))
  const leader = spawn(process.execPath, [leaderScript, childScript, pidFile, readyFile], {
    detached: detachedProcessGroup(),
    stdio: 'ignore',
  })
  await waitFor(() => {
    try {
      return Number(readFileSync(pidFile, 'utf8')) > 0 && readFileSync(readyFile, 'utf8') === 'ready'
    } catch { return false }
  })
  const descendantPid = Number(readFileSync(pidFile, 'utf8'))
  terminateChildProcess(leader, { processGroup: true, graceMs: 50 })
  await new Promise<void>((resolve) => leader.once('close', () => resolve()))
  try {
    await waitFor(() => !processExists(descendantPid))
    assert.equal(processExists(descendantPid), false, 'a descendant that outlives its group leader is forcibly reaped')
  } finally {
    if (processExists(descendantPid)) {
      try { process.kill(descendantPid, 'SIGKILL') } catch { /* already gone */ }
    }
    rmSync(dir, { recursive: true, force: true })
  }
} else {
  assert.equal(detachedProcessGroup(), false)
}

// Cancellation that predates iteration must launch nothing. An impossible
// binary path makes this an executable assertion for every one-shot CLI: if a
// regression reaches spawn, the drain rejects with ENOENT.
const alreadyAborted = new AbortController()
alreadyAborted.abort()
const provider = { id: 'cancelled', cliPath: '/definitely/not/a/vontra-cli', cliEnv: {} } as any
const messages = [{ role: 'user', content: 'do not launch' }] as any
const drain = async (source: AsyncIterable<unknown>): Promise<void> => {
  for await (const _ of source) { /* no chunks expected */ }
}
await drain(streamCodexCli({ provider, model: 'codex-cli-default', messages, signal: alreadyAborted.signal }))
await drain(streamClaudeCli({ provider, model: 'claude-cli-default', messages, signal: alreadyAborted.signal }))
await drain(streamCursorAgent({ provider, model: 'cursor-default', messages, signal: alreadyAborted.signal }))
await drain(streamGeminiCli({ provider, model: 'gemini-default', messages, signal: alreadyAborted.signal }))
await drain(streamPiCli({ provider, model: 'pi-default', messages, signal: alreadyAborted.signal }))
let appServerSpawns = 0
await drain(streamCodexAppServer({
  provider,
  model: 'codex-cli-default',
  messages,
  sessionKey: 'already-aborted',
  signal: alreadyAborted.signal,
  spawnTransport: () => {
    appServerSpawns++
    throw new Error('must not spawn')
  },
}))
assert.equal(appServerSpawns, 0, 'an already-aborted app-server turn launches no transport')
assert.deepEqual(
  await startCliLogin({ ...provider, transport: 'codex-cli' }, { signal: alreadyAborted.signal }),
  { ok: false, loggedIn: false, message: 'Sign-in cancelled.' },
  'an already-aborted CLI login launches no auth process',
)

console.log('child-process-lifecycle.test.ts passed')
