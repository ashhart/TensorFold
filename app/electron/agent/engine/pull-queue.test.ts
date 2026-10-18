import assert from 'node:assert/strict'
import { PullQueue } from './pull'
import { makeHarness, waitFor } from './test-fixtures/harness'

async function main() {
  const h = await makeHarness({ FAKE_PULL_FILES: '1', FAKE_PULL_SIZE: '65536', FAKE_PULL_CHUNK: '4096', FAKE_PULL_DELAY_MS: '10' })
  const queue = new PullQueue(3)
  const updates: string[] = []
  const options = (repo: string, companion?: string) => ({ binary: h.binary, env: h.env, hub: h.hub, steps: [{ repo, revision: null }, ...(companion ? [{ repo: companion, revision: null }] : [])], totalBytes: null, pollMs: 25, onUpdate: (p: any) => updates.push(`${p.repo}:${p.state}`) })
  try {
    let finish!: () => void
    const durableWrite = new Promise<void>((resolve) => { finish = resolve })
    const first = queue.enqueue(options('Org/First', 'Org/Drafter'), async (progress) => { await durableWrite; return progress })
    const cancelled = queue.enqueue(options('Org/Cancelled'))
    const second = queue.enqueue(options('Org/Second'))
    assert.equal(second.progress().state, 'queued')
    assert.equal(second.progress().queuePosition, 2)
    assert.equal(queue.hasRepo('Org/Drafter'), true, 'a companion is reserved with its parent job')
    assert.throws(() => queue.enqueue(options('Org/Drafter')), /already.*download/i, 'a companion cannot race a separate download')
    assert.throws(() => queue.enqueue(options('Org/Full')), /queue.*full/i)
    cancelled.cancel()
    assert.equal((await cancelled.done).state, 'cancelled')
    assert.equal(second.progress().queuePosition, 1, 'positions update after cancelling a queued item')
    await waitFor(() => first.progress().state === 'verifying', 10_000, 'durable completion boundary')
    assert.deepEqual((await h.records()).filter((r) => r.kind === 'pull').map((r) => r.repo), ['Org/First', 'Org/Drafter'], 'next job never overlaps model, companion or finalization')
    finish()
    assert.equal((await first.done).state, 'done')
    assert.equal((await second.done).state, 'done')
    assert.deepEqual((await h.records()).filter((r) => r.kind === 'pull').map((r) => r.repo), ['Org/First', 'Org/Drafter', 'Org/Second'])
    assert.ok(updates.includes('Org/First:verifying'))
    assert.equal(queue.hasRepo('Org/Drafter'), false)

    const failure = queue.enqueue(options('Org/CommitFailed'), async () => { throw new Error('Registry could not be saved') })
    const recovery = queue.enqueue(options('Org/Recovery'))
    assert.equal((await failure.done).state, 'failed')
    assert.match(failure.progress().message ?? '', /Registry could not be saved/)
    assert.equal((await recovery.done).state, 'done', 'failure does not strand explicitly queued downloads')

    const running = queue.enqueue(options('Org/ActiveCancel'))
    const successor = queue.enqueue(options('Org/ConsentedNext'))
    await waitFor(async () => (await h.records()).some((r) => r.kind === 'pull' && r.repo === 'Org/ActiveCancel'), 10_000, 'active pull')
    running.cancel()
    assert.equal((await running.done).state, 'cancelled')
    assert.equal((await successor.done).state, 'done', 'cancelling the current item respects the separate next-item consent')

    const stopFirst = queue.enqueue(options('Org/ShutdownFirst'))
    const stopNext = queue.enqueue(options('Org/ShutdownNeverStarted'))
    queue.cancelAll()
    await Promise.all([stopFirst.done, stopNext.done])
    assert.equal(stopNext.progress().state, 'cancelled')
    assert.equal((await h.records()).some((r) => r.kind === 'pull' && r.repo === 'Org/ShutdownNeverStarted'), false)
    assert.throws(() => queue.enqueue(options('Org/AfterShutdown')), /closed/i)
  } finally {
    queue.cancelAll()
    await h.cleanup()
  }
  console.log('Download queue: serial model/companion/finalize, bounds, cancellation, recovery and shutdown passed')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
