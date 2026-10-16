import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { PullProgress } from '../types'
import { hasCachedSnapshot } from './paths'
import { startPull } from './pull'
import { makeHarness, processAlive, waitFor } from './test-fixtures/harness'

async function main() {
  const h = await makeHarness({ FAKE_PULL_FILES: '3', FAKE_PULL_SIZE: String(256 * 1024), FAKE_PULL_CHUNK: String(16 * 1024), FAKE_PULL_DELAY_MS: '15' })
  try {
    const total = 3 * 256 * 1024
    const updates: PullProgress[] = []
    const first = startPull({ binary: h.binary, env: h.env, hub: h.hub, steps: [{ repo: 'Org/Model', revision: null }], totalBytes: total, pollMs: 40, onUpdate: (p) => updates.push(p) })
    // Byte progress grows from the cache folder while the process runs; then cancel mid-download.
    await waitFor(() => updates.some((u) => u.receivedBytes > 0 && u.receivedBytes < total), 10_000, 'partial byte progress')
    const partial = first.progress().receivedBytes
    first.cancel()
    const cancelled = await first.done
    assert.equal(cancelled.state, 'cancelled')
    assert.match(cancelled.message ?? '', /resume/)
    assert.ok(partial > 0 && partial < total, `partial progress ${partial}`)
    assert.equal(await hasCachedSnapshot(h.hub, 'Org/Model'), false, 'a cancelled pull leaves no refs/main')

    // Pulling again resumes and finishes, with lines and a full total.
    const lines: string[] = []
    const second = startPull({ binary: h.binary, env: h.env, hub: h.hub, steps: [{ repo: 'Org/Model', revision: null }, { repo: 'z-lab/Drafter', revision: null }], totalBytes: total * 2, pollMs: 40, onUpdate: (p) => { if (p.lastLine) lines.push(p.lastLine) } })
    const finished = await second.done
    assert.equal(finished.state, 'done', finished.message ?? '')
    assert.equal(finished.receivedBytes, total * 2)
    assert.ok(lines.some((line) => line.startsWith('stored Org/Model@')), 'pull lines reach the renderer')
    assert.ok(lines.some((line) => line.startsWith('stored z-lab/Drafter@')), 'the companion pull runs after the model')
    assert.equal(await hasCachedSnapshot(h.hub, 'Org/Model'), true)
    assert.equal(await hasCachedSnapshot(h.hub, 'z-lab/Drafter'), true)

    // A failing pull reports its last line.
    const failing = startPull({ binary: h.binary, env: { ...h.env, FAKE_PULL_FAIL: '1' }, hub: h.hub, steps: [{ repo: 'Org/Broken', revision: 'main' }], totalBytes: null, onUpdate: () => {} })
    const failed = await failing.done
    assert.equal(failed.state, 'failed')
    assert.match(failed.message ?? '', /file tree is unreachable/)

    // Cancel kills the whole process group, not just the direct child.
    const slow = startPull({ binary: h.binary, env: { ...h.env, FAKE_PULL_DELAY_MS: '200' }, hub: h.hub, steps: [{ repo: 'Org/Slow', revision: null }], totalBytes: total, pollMs: 40, onUpdate: () => {} })
    await waitFor(async () => (await fs.readdir(path.join(h.hub, 'models--Org--Slow', 'blobs')).catch(() => [])).length > 0, 10_000, 'slow pull to start')
    const stdout = execFileSync('/bin/ps', ['-axo', 'pid=,command=']).toString()
    const pid = Number(stdout.split('\n').find((line) => line.includes('fake-engine.mjs pull Org/Slow'))?.trim().split(/\s+/)[0])
    assert.ok(pid > 0, 'found the pull process')
    slow.cancel()
    assert.equal((await slow.done).state, 'cancelled')
    await waitFor(() => !processAlive(pid), 5000, 'pull process to exit')
  } finally {
    await h.cleanup()
  }
  console.log('engine pull tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
