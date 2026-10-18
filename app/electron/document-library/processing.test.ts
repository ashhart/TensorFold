import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DocumentProcessingQueue } from './processing'
import { readDocumentFile } from './scanner'
import { makeDocumentSource } from '../agent/document-connector'
import type { DocumentReadResult } from '../agent/types'

const deferred = () => { let resolve!: () => void; const promise = new Promise<void>(done => { resolve = done }); return { promise, resolve } }
async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-document-processing-'))
  process.env.VONTRA_HOME = path.join(root, 'profile')
  try {
    const folder = path.join(root, 'documents'); await fs.mkdir(folder)
    await fs.writeFile(path.join(folder, 'alpha.txt'), 'Amber lark evidence')
    await fs.writeFile(path.join(folder, 'broken.csv'), 'Name,Body\nA,"unterminated\n')
    await fs.writeFile(path.join(folder, 'gamma.txt'), 'Sapphire swift evidence')
    const settings = { sources: [makeDocumentSource(folder)], maxDocumentBytes: 8_192, maxDocuments: 20, lastSyncAt: null }
    const committed = new Map<string, DocumentReadResult>(), outcomes: string[] = [], reads: string[] = []
    const blocked = deferred(), release = deferred()
    let physical = 0, maximum = 0, blockedOnce = false
    const options = { directory: path.join(root, 'profile', 'library'), getSettings: () => settings, commit: async (_source: any, outcome: any, doc?: DocumentReadResult) => { outcomes.push(`${outcome.relativePath}:${outcome.status}`); if (doc) committed.set(outcome.id, doc) }, read: async (...args: Parameters<typeof readDocumentFile>) => {
      physical++; maximum = Math.max(maximum, physical); reads.push(path.basename(args[1]))
      try {
        if (committed.size === 1 && !blockedOnce) { blockedOnce = true; blocked.resolve(); await release.promise }
        return await readDocumentFile(...args)
      } finally { physical-- }
    } }
    const queue = new DocumentProcessingQueue(options)
    await queue.control({ action: 'start' }); await blocked.promise
    const beforePause = await queue.snapshot(), running = beforePause.jobs.find(job => job.state === 'processing')!
    assert.ok(running)
    await queue.control({ action: 'pause' }); release.resolve(); await queue.wait()
    const paused = await queue.snapshot()
    assert.equal(paused.paused, true); assert.equal(paused.jobs.filter(job => job.state === 'ready').length, 1)
    assert.equal(committed.size, 1, 'A parse finishing after Pause must be discarded')
    assert.ok(paused.jobs.filter(job => job.state !== 'ready').every(job => job.state === 'paused'))
    const raw = await fs.readFile(path.join(options.directory, 'processing.json'), 'utf8')
    assert.match(raw, /safe-json:v1:/); assert.ok(!raw.includes('alpha.txt') && !raw.includes(folder), 'Filenames and source paths require protected profile storage')
    const checkpointReady = paused.jobs.find(job => job.state === 'ready')!
    const restartedReads: string[] = []
    const restarted = new DocumentProcessingQueue({ ...options, read: async (...args) => { restartedReads.push(path.basename(args[1])); return readDocumentFile(...args) } })
    const recovery = await restarted.snapshot()
    assert.equal(recovery.paused, true); assert.equal(recovery.running, false, 'Restart never silently resumes indexing')
    await restarted.control({ action: 'resume' }); await restarted.wait()
    assert.ok(!restartedReads.includes(checkpointReady.relativePath), 'Restart resumes at per-file checkpoints without rereading completed documents')
    const finished = await restarted.snapshot()
    assert.equal(finished.jobs.filter(job => job.state === 'ready').length, 2)
    assert.equal(finished.jobs.filter(job => job.state === 'failed').length, 1)
    assert.match(finished.jobs.find(job => job.state === 'failed')!.error!, /CSV|quoted|unterminated|closed/i)
    assert.equal(maximum, 1)
    const failed = finished.jobs.find(job => job.state === 'failed')!
    await fs.writeFile(path.join(folder, failed.relativePath), 'Name,Body\nA,Repaired text\n')
    await restarted.control({ action: 'retry', jobId: failed.id }); await restarted.wait()
    assert.equal((await restarted.snapshot()).jobs.find(job => job.id === failed.id)?.state, 'ready')
    assert.equal(committed.size, 3, 'A failed file does not discard other successes; individual retry replaces only its checkpoint')
    const started = deferred(), finish = deferred(), noCommit: string[] = []
    const cancelling = new DocumentProcessingQueue({ ...options, directory: path.join(root, 'profile', 'cancel-library'), read: async (...args) => { started.resolve(); await finish.promise; return readDocumentFile(...args) }, commit: async (_source, outcome) => { noCommit.push(outcome.id) } })
    await cancelling.control({ action: 'start' }); await started.promise
    const active = (await cancelling.snapshot()).jobs.find(job => job.state === 'processing')!
    await cancelling.control({ action: 'cancel', jobId: active.id }); await cancelling.control({ action: 'pause' })
    finish.resolve(); await cancelling.wait()
    assert.equal((await cancelling.snapshot()).jobs.find(job => job.id === active.id)?.state, 'cancelled')
    assert.equal(noCommit.length, 0, 'Cancellation discards the physical in-flight parser result')
    const revokedStart = deferred(), revokedFinish = deferred()
    const revoked = new DocumentProcessingQueue({ ...options, directory: path.join(root, 'profile', 'revoked-library'), read: async (...args) => { revokedStart.resolve(); await revokedFinish.promise; return readDocumentFile(...args) }, commit: async () => { throw new Error('Revoked extraction must never commit') } })
    await revoked.control({ action: 'start' }); await revokedStart.promise
    settings.sources[0].enabled = false; revokedFinish.resolve(); await revoked.wait()
    assert.ok((await revoked.snapshot()).jobs.every(job => job.state === 'failed'))
    assert.ok(outcomes.some(outcome => outcome.endsWith(':failed')))
    const damagedDirectory = path.join(root, 'profile', 'damaged-library'); await fs.mkdir(damagedDirectory); await fs.writeFile(path.join(damagedDirectory, 'processing.json'), '{damaged queue')
    const damaged = new DocumentProcessingQueue({ ...options, directory: damagedDirectory })
    await assert.rejects(damaged.snapshot(), /queue could not be read/)
    await damaged.quiesce(); damaged.resumeHost(); await damaged.clear()
    assert.deepEqual((await damaged.snapshot()).jobs, [], 'An explicit whole-library purge can recover damaged queue metadata without blocking workspace quiescence')
    console.log('PASS protected document queue: per-file checkpoints, pause/discard, restart, mixed outcomes, independent retry, physical concurrency, cancellation and revoked collections')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
