import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { PullProgress } from '../types'
import { engineChildEnv } from './engine-env'
import { PROXY_ENV_KEYS } from './pull-network'
import { PullQueue, startPull, type PullHandle } from './pull'
import { makeHarness, processAlive, waitFor } from './test-fixtures/harness'
import { PullClock } from './test-fixtures/pull-clock'

async function main() {
  const h = await makeHarness({ FAKE_PULL_FILES: '2', FAKE_PULL_SIZE: '262144', FAKE_PULL_CHUNK: '4096', FAKE_PULL_DELAY_MS: '15' })
  const queue = new PullQueue()
  const owned: PullHandle[] = []
  const options = (repo: string) => ({ binary: h.binary, env: h.env, hub: h.hub, steps: [{ repo, revision: null }], totalBytes: 524288, pollMs: 20, onUpdate: (_progress: PullProgress) => {} })
  try {
    // The child has no cache writes, even while printing a connection status line.
    const clock = new PullClock()
    const updates: PullProgress[] = []
    let finalized = false
    const stalled = queue.enqueue({ ...options('Org/Stalled'), env: { ...h.env, FAKE_PULL_STALL: '1', FAKE_PULL_IGNORE_TERM: '1' }, inactivityTimer: clock.schedule, killGraceMs: 30, onUpdate: (progress) => updates.push(progress) }, async (progress) => { finalized = true; return progress })
    const next = queue.enqueue(options('Org/Next'))
    owned.push(stalled, next)
    const record = await waitFor(async () => (await h.records()).find((entry) => entry.kind === 'pull-stalled'), 5000, 'non-writing fake pull')
    await waitFor(() => stalled.progress().lastLine === 'Waiting for the Hub connection', 5000, 'stall handler ready')
    clock.advance(59_999)
    assert.equal(stalled.progress().state, 'running')
    assert.equal(stalled.progress().message, null)
    assert.equal(next.progress().state, 'queued')
    clock.advance(1)
    assert.ok(updates.some((progress) => /60 seconds/.test(progress.message ?? '')), 'the message appears at the deadline')
    assert.throws(() => queue.enqueue(options('Org/Stalled')), /already.*download/i, 'retry cannot overlap the stopping child')
    const failure = await stalled.done
    assert.equal(failure.state, 'failed')
    assert.equal(failure.receivedBytes, 0)
    assert.match(failure.message ?? '', /timed out.*Hugging Face Hub.*retry/)
    assert.equal(processAlive(record.pid), false, 'done waits for the owned cache writer to exit, including SIGKILL escalation')
    assert.ok(updates.some((progress) => progress.lastLine === 'Still waiting for the Hub connection'), 'the stalled child ignored SIGTERM')
    assert.equal(finalized, false)
    assert.equal(clock.pending, 0)
    assert.equal((await next.done).state, 'done', 'a failed job does not strand the queue')

    const retried = queue.enqueue(options('Org/Stalled'))
    owned.push(retried)
    assert.equal((await retried.done).state, 'done', 'the failed repository can be explicitly retried')

    // Real cache growth grants a fresh deadline and keeps the existing progress display working.
    const growingClock = new PullClock()
    const finishFile = path.join(h.root, 'finish-growing')
    const progress: PullProgress[] = []
    const growing = startPull({ ...options('Org/Growing'), env: { ...h.env, FAKE_PULL_FINISH_FILE: finishFile }, inactivityTimer: growingClock.schedule, onUpdate: (value) => progress.push(value) })
    owned.push(growing)
    await waitFor(() => progress.find((value) => value.receivedBytes > 0 && value.receivedBytes < 524288), 10_000, 'normal fake pull progress')
    const bytes = growing.progress().receivedBytes
    growingClock.advance(59_999)
    await waitFor(() => growing.progress().receivedBytes > bytes, 10_000, 'new cache bytes')
    growingClock.advance(1)
    assert.equal(growing.progress().state, 'running')
    assert.equal(growing.progress().message, null)
    await fs.writeFile(finishFile, '')
    assert.equal((await growing.done).state, 'done')
    assert.equal(growingClock.pending, 0)

    // Inspect the actual spawned child's environment, then emit deliberately unsafe diagnostics.
    const proxy = 'http://proxy-user:needle%3Apass@example.test:8080/private?token=hidden'
    const expected = Object.fromEntries(PROXY_ENV_KEYS.map((key) => [key, key.toLowerCase() === 'no_proxy' ? 'localhost,.example.test' : proxy]))
    const env = engineChildEnv({ ...h.env, ...expected, OPENAI_API_KEY: 'provider-secret' })
    const script = `
      const assert = require('node:assert/strict')
      const fs = require('node:fs')
      for (const [key, value] of Object.entries(${JSON.stringify(expected)})) assert.equal(process.env[key], value)
      assert.equal(process.env.OPENAI_API_KEY, undefined)
      console.log('proxy environment accepted')
      fs.writeSync(1, 'x'.repeat(66000) + 'https://unknown-user:unknown-secret@')
      setTimeout(() => fs.writeSync(1, 'example.test:8080/private\\n'), 20)
      fs.writeSync(2, 'proxy refused http://proxy-user:needle%3A')
      setTimeout(() => { fs.writeSync(2, 'pass@example.test:8080/private?token=hidden\\n'); process.exitCode = 1 }, 50)
    `
    const visible: PullProgress[] = []
    const redacted = startPull({ ...options('Org/Proxy'), binary: { command: process.execPath, args: ['-e', script, '--'] }, env, onUpdate: (value) => visible.push(value) })
    owned.push(redacted)
    const rejected = await redacted.done
    assert.equal(rejected.state, 'failed')
    assert.ok(visible.some((value) => value.lastLine === 'proxy environment accepted'), 'the real child received all eight exact values')
    assert.match(rejected.message ?? '', /proxy \(http:\/\/example\.test:8080\)/)
    assert.doesNotMatch(JSON.stringify(visible), /proxy-user|needle|unknown-user|unknown-secret|private|hidden|provider-secret/)
    console.log('5/5 pull liveness checks passed: bounded stall, stopped writer, retry, normal progress and spawned proxy redaction')
  } finally {
    queue.cancelAll()
    for (const handle of owned) handle.cancel()
    await Promise.all(owned.map((handle) => handle.done))
    await h.cleanup()
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
