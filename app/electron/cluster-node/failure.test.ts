import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createClusterNodeServer } from './server'
import { DiskNodeJournal, commandDigest } from './journal'
import type { NodeJournal, StoredNodeState } from './types'
import { FakeEngine, model, deferred, until, sleep } from './testing/fixture'

class FailingJournal implements NodeJournal {
  fail = false
  constructor(readonly disk: DiskNodeJournal) {}
  read() { return this.disk.read() }
  update<T>(edit: (state: StoredNodeState) => T) { if (this.fail) return Promise.reject(new Error('Synthetic journal disk failure')); return this.disk.update(edit) }
  drain() { return this.disk.drain() }
}
async function fixture(root: string, engine: FakeEngine, journal?: NodeJournal) {
  const node = await createClusterNodeServer({ config: { schema: 1, dataDir: root, host: '127.0.0.1', port: 0 }, engine, ...(journal ? { journal } : {}) })
  const endpoint = await node.listen(), token = (await fs.readFile(path.join(root, 'management.token'), 'utf8')).trim(), api = '/api/cluster/v1'
  const call = (route: string, method = 'GET', body?: unknown, credential = token, signal?: AbortSignal) => fetch(endpoint.url + route, { method, headers: { authorization: `Bearer ${credential}`, ...(body ? { 'content-type': 'application/json' } : {}) }, ...(body ? { body: JSON.stringify(body) } : {}), ...(signal ? { signal } : {}) })
  const nodeId = (await (await call(api + '/identity?nonce=synthetic_nonce')).json() as any).nodeId
  const getJob = async (id: string) => (await call(api + '/jobs/' + id)).json() as Promise<any>
  const finished = async (id: string) => { await until(async () => ['completed', 'cancelled', 'failed', 'needs-review'].includes((await getJob(id)).phase)); return getJob(id) }
  return { node, endpoint, api, call, nodeId, getJob, finished }
}
async function main() {
  const base = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-node-failures-')))
  const cleanup: Array<() => Promise<void>> = []
  try {
    const root = path.join(base, 'cancel'), engine = new FakeEngine(); engine.ensureGate = deferred(); engine.ensureDrain = deferred()
    const f = await fixture(root, engine); cleanup.push(() => f.node.close())
    const first = await (await f.call(f.api + '/models/ensure', 'POST', { commandId: 'ensure-cancelled-0001', expectedNodeId: f.nodeId, model })).json() as any
    await until(() => engine.ensureCalls === 1)
    const cancelled = await f.call(f.api + '/jobs/' + first.id + '/cancel', 'POST', { commandId: 'cancel-first-command', expectedNodeId: f.nodeId })
    assert.equal(cancelled.status, 202); await until(() => engine.cancelled)
    const second = await (await f.call(f.api + '/models/ensure', 'POST', { commandId: 'ensure-second-command', expectedNodeId: f.nodeId, model })).json() as any
    await sleep(30); assert.equal(engine.ensureCalls, 1); assert.equal(engine.physicalParsers, 1, 'Cancellation keeps the physical queue owned until its parser drains')
    engine.ensureGate = null; engine.ensureDrain.resolve(); engine.ensureDrain = null
    assert.equal((await f.finished(first.id)).phase, 'cancelled'); assert.equal((await f.finished(second.id)).phase, 'completed'); assert.equal(engine.maximumParsers, 1)
    const launch = { deploymentId: 'stream-deployment', commandId: 'start-stream-command', expectedNodeId: f.nodeId, model, profileId: 'single-native', role: 'headless', generation: { contextTokens: 4096, maxOutputTokens: 64, temperature: 0 }, transport: 'tcp', rank: 0, worldSize: 1, links: [] }
    const start = await (await f.call(f.api + '/deployments/start', 'POST', launch)).json() as any; assert.equal((await f.finished(start.id)).phase, 'completed')
    const canary = await (await f.call(f.api + '/canary', 'POST', { commandId: 'canary-stream-command', expectedNodeId: f.nodeId, deploymentId: launch.deploymentId })).json() as any; assert.equal((await f.finished(canary.id)).phase, 'completed')
    const credential = (await (await f.call(f.api + '/endpoint-credentials')).json() as any).apiKey
    engine.cancelled = false; engine.mode = 'slow'; engine.requestDrain = deferred(); const controller = new AbortController()
    const body = { model: model.repo, messages: [{ role: 'user', content: 'Synthetic cancellable request' }], stream: true }
    const streaming = await f.call('/v1/chat/completions', 'POST', body, credential, controller.signal)
    const reader = streaming.body!.getReader(); await reader.read(); controller.abort(); await reader.cancel().catch(() => {})
    await until(() => engine.cancelled)
    assert.equal((await f.call('/v1/chat/completions', 'POST', body, credential)).status, 409)
    assert.equal(engine.physicalRequests, 1)
    engine.requestDrain.resolve(); engine.requestDrain = null
    await until(() => engine.physicalRequests === 0); engine.mode = 'normal'
    await until(async () => (JSON.parse(await fs.readFile(path.join(root, 'state.json'), 'utf8')).request.phase !== 'running'))
    await until(async () => { const resumed = await f.call('/v1/chat/completions', 'POST', { ...body, stream: false }, credential); const value = await resumed.json(); if (resumed.status === 409) return false; assert.equal(resumed.status, 200, JSON.stringify(value)); return true }, 'physical request receipt drain')
    assert.equal(engine.maximumRequests, 1)
    await f.node.close()

    const diskRoot = path.join(base, 'failed-before-effect'), disk = new FailingJournal(new DiskNodeJournal(diskRoot)), noEffects = new FakeEngine()
    const d = await fixture(diskRoot, noEffects, disk); cleanup.push(() => d.node.close()); disk.fail = true
    assert.equal((await d.call(d.api + '/models/ensure', 'POST', { commandId: 'never-effect-command', expectedNodeId: d.nodeId, model })).status, 503)
    assert.equal(noEffects.ensureCalls, 0, 'A failed accepted-receipt write must prevent every engine effect')
    assert.equal((await (await d.call(d.api + '/status')).json() as any).engine.available, false)
    assert.equal((await d.call(d.api + '/models/ensure', 'POST', { commandId: 'different-effect-command', expectedNodeId: d.nodeId, model })).status, 503)
    await d.node.close()

    const partialRoot = path.join(base, 'failed-after-effect'), partialDisk = new FailingJournal(new DiskNodeJournal(partialRoot)), partialEngine = new FakeEngine()
    const p = await fixture(partialRoot, partialEngine, partialDisk); cleanup.push(() => p.node.close()); partialEngine.afterEnsure = () => { partialDisk.fail = true }
    const accepted = await (await p.call(p.api + '/models/ensure', 'POST', { commandId: 'partial-effect-command', expectedNodeId: p.nodeId, model })).json() as any
    assert.equal((await p.finished(accepted.id)).phase, 'needs-review'); assert.equal(partialEngine.ensureCalls, 1)
    assert.equal((await p.call(p.api + '/models/ensure', 'POST', { commandId: 'partial-effect-command', expectedNodeId: p.nodeId, model })).status, 503)
    await p.node.close()
    const afterRestartEngine = new FakeEngine(), r = await fixture(partialRoot, afterRestartEngine); cleanup.push(() => r.node.close())
    const recovered = await r.getJob(accepted.id); assert.equal(recovered.phase, 'needs-review'); assert.equal(afterRestartEngine.ensureCalls, 0)
    const review = await (await r.call(r.api + '/models/ensure', 'POST', { commandId: 'new-after-review-command', expectedNodeId: r.nodeId, model })).json() as any
    assert.equal((await r.finished(review.id)).phase, 'failed'); assert.equal(afterRestartEngine.ensureCalls, 0, 'Restart cannot guess whether an earlier physical downloader finished')
    console.log('PASS node physical ownership: persisted cancellation, parser and stream drain, one request per node, pre-effect disk refusal, post-effect uncertainty and restart without replay')
  } finally { for (const close of cleanup.reverse()) await close(); await fs.rm(base, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
