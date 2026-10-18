import assert from 'node:assert/strict'
import type { PullProgress } from '../../lib/types'
import { downloadEstimate, downloadStateLabel, isDownloadPending, orderedDownloads } from './download-display'

const base: PullProgress = { repo: 'Org/Example', revision: null, state: 'running', phase: 'Downloading Org/Example', receivedBytes: 1024 ** 3, totalBytes: 4 * 1024 ** 3, lastLine: null, message: null, startedAt: 0, finishedAt: null, updatedAt: 5000, bytesPerSecond: 1024 ** 2, etaSeconds: 3020 }
assert.equal(downloadEstimate(base, 5000), '1 MB/s · about 51 min left')
assert.equal(downloadEstimate({ ...base, etaSeconds: 19 }, 5000), '1 MB/s · about 20 sec left')
assert.equal(downloadEstimate({ ...base, totalBytes: null, etaSeconds: null }, 5000), '1 MB/s', 'unknown sizes show observed speed without promising an ETA')
for (const state of ['queued', 'verifying', 'done', 'cancelled', 'failed'] as const) assert.equal(downloadEstimate({ ...base, state }, 5000), null)
for (const patch of [{ updatedAt: undefined }, { updatedAt: 0 }, { updatedAt: 30_000 }, { bytesPerSecond: null }, { bytesPerSecond: 0 }, { bytesPerSecond: Number.NaN }]) assert.equal(downloadEstimate({ ...base, ...patch }, 20_001), null)
for (const state of ['queued', 'running', 'verifying'] as const) assert.equal(isDownloadPending({ state }), true)
for (const state of ['done', 'cancelled', 'failed'] as const) assert.equal(isDownloadPending({ state }), false)
assert.equal(downloadStateLabel({ ...base, state: 'queued', queuePosition: 2 }), 'Queued · 2 ahead')
assert.equal(downloadStateLabel({ ...base, state: 'queued', queuePosition: 0 }), 'Starting download')
assert.equal(downloadStateLabel({ ...base, state: 'done' }), 'Ready to load')
assert.equal(downloadStateLabel({ ...base, phase: 'Downloading companion Org/Drafter' }), 'Downloading companion model')
const jobs = [{ ...base, repo: 'Org/QueuedTwo', state: 'queued' as const, queuePosition: 2 }, { ...base, repo: 'Org/Done', state: 'done' as const, finishedAt: 1000 }, { ...base, repo: 'Org/QueuedOne', state: 'queued' as const, queuePosition: 1 }, { ...base, repo: 'Org/Active' }]
assert.deepEqual(orderedDownloads(jobs).map((job) => job.repo), ['Org/Active', 'Org/QueuedOne', 'Org/QueuedTwo', 'Org/Done'])
assert.equal(jobs[0].repo, 'Org/QueuedTwo', 'sorting never mutates the engine snapshot')
console.log('Download display: pending states, honest/stale/unknown ETA, queue position and immutable ordering passed')
