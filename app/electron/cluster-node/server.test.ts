import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createClusterNodeServer } from './server'
import { FakeEngine, model, until, sleep } from './testing/fixture'
import type { ClusterNodeJob } from '../../shared/cluster-workspace'

async function main() {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-node-http-'))), engine = new FakeEngine()
  const node = await createClusterNodeServer({ config: { schema: 1, dataDir: root, host: '127.0.0.1', port: 0 }, engine, maxResponseBytes: 1024 * 1024 })
  const endpoint = await node.listen(), management = (await fs.readFile(path.join(root, 'management.token'), 'utf8')).trim(), seed = (await fs.readFile(path.join(root, 'inference.token'), 'utf8')).trim()
  const call = (route: string, method = 'GET', body?: unknown, token = management, extra: Record<string, string> = {}) => fetch(endpoint.url + route, { method, headers: { authorization: `Bearer ${token}`, ...(body === undefined ? {} : { 'content-type': 'application/json' }), ...extra }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) })
  const api = '/api/cluster/v1'
  const job = async (id: string): Promise<ClusterNodeJob> => (await call(api + '/jobs/' + id)).json()
  const finished = async (value: ClusterNodeJob) => { await until(async () => ['completed', 'failed', 'cancelled', 'needs-review'].includes((await job(value.id)).phase), 'durable node job'); return job(value.id) }
  try {
    assert.equal((await call(api + '/status', 'GET', undefined, seed)).status, 401, 'Inference credentials never authorize management')
    const identity = await (await call(api + '/identity?nonce=synthetic_nonce')).json() as any
    assert.equal(identity.authenticated, true); assert.equal(identity.nonce, 'synthetic_nonce')
    const nodeId = identity.nodeId
    assert.equal((await call(api + '/identity?nonce=bad')).status, 400)
    assert.equal((await call(api + '/status', 'GET', undefined, management, { origin: 'https://example.invalid' })).status, 403)
    assert.equal((await call(api + '/endpoint-credentials')).status, 409)
    const admission = { expectedNodeId: nodeId, role: 'headless', model, profileId: 'single-native', generation: { contextTokens: 4096, maxOutputTokens: 64, temperature: 0 }, transport: 'tcp', rank: 0, worldSize: 1, links: [] }
    assert.equal((await (await call(api + '/admission', 'POST', admission)).json() as any).allowed, false)
    assert.equal((await call(api + '/models/ensure', 'POST', { commandId: 'bad-revision-request', expectedNodeId: nodeId, model: { ...model, revision: 'main' } })).status, 400)
    assert.equal((await call(api + '/models/ensure', 'POST', { commandId: 'wrong-node-request', expectedNodeId: 'different-node', model })).status, 409)
    const ensureInput = { commandId: 'ensure-first-request', expectedNodeId: nodeId, model }
    const ensure = await (await call(api + '/models/ensure', 'POST', ensureInput)).json() as ClusterNodeJob
    assert.equal((await finished(ensure)).phase, 'completed'); assert.equal(engine.ensureCalls, 1)
    const repeated = await (await call(api + '/models/ensure', 'POST', ensureInput)).json() as ClusterNodeJob
    assert.equal(repeated.id, ensure.id); assert.equal(engine.ensureCalls, 1)
    assert.equal((await call(api + '/models/ensure', 'POST', { ...ensureInput, model: { ...model, revision: 'b'.repeat(40) } })).status, 409, 'A changed payload cannot reuse a durable command ID')
    const startInput = { ...admission, deploymentId: 'deployment-one', commandId: 'start-first-request' }
    const start = await (await call(api + '/deployments/start', 'POST', startInput)).json() as ClusterNodeJob
    assert.equal((await finished(start)).phase, 'completed'); assert.equal(engine.startCalls, 1)
    assert.equal((await call(api + '/endpoint-credentials')).status, 409)
    const canary = await (await call(api + '/canary', 'POST', { commandId: 'canary-first-request', expectedNodeId: nodeId, deploymentId: startInput.deploymentId })).json() as ClusterNodeJob
    assert.equal((await finished(canary)).phase, 'completed')
    const credentials = await (await call(api + '/endpoint-credentials')).json() as any
    assert.equal(credentials.model, model.repo); assert.match(credentials.apiKey, /^[a-f0-9]{64}$/)
    assert.ok(credentials.apiKey !== seed && credentials.apiKey !== management)
    assert.equal((await call('/v1/models', 'GET', undefined, management)).status, 401)
    const models = await (await call('/v1/models', 'GET', undefined, credentials.apiKey)).json() as any
    assert.equal(models.data[0].id, model.repo)
    const input = { model: model.repo, messages: [{ role: 'user', content: 'SYNTHETIC-PRIVATE-PROMPT-NOT-FOR-JOURNAL' }], max_tokens: 8 }
    const response = await call('/v1/chat/completions', 'POST', input, credentials.apiKey)
    assert.equal(response.status, 200); assert.equal((await response.json() as any).choices[0].finish_reason, 'stop')
    assert.equal(engine.requests.at(-1)?.model, startInput.deploymentId, 'The friendly repo alias is rewritten only to its owned native session')
    const stored = await fs.readFile(path.join(root, 'state.json'), 'utf8')
    assert.ok(!stored.includes(input.messages[0].content) && !stored.includes(management) && !stored.includes(credentials.apiKey), 'Receipts contain no prompts or credentials')
    assert.equal((await call('/v1/chat/completions', 'POST', { ...input, model: 'Example/Other' }, credentials.apiKey)).status, 409)
    engine.mode = 'sse'
    const streaming = await call('/v1/chat/completions', 'POST', { ...input, stream: true }, credentials.apiKey)
    assert.match(await streaming.text(), /\[DONE\]/)
    engine.mode = 'partial-sse'
    const partial = await call('/v1/chat/completions', 'POST', { ...input, stream: true }, credentials.apiKey)
    await assert.rejects(partial.text(), /terminated|abort|fetch/i)
    await until(async () => JSON.parse(await fs.readFile(path.join(root, 'state.json'), 'utf8')).request.phase === 'failed')
    engine.mode = 'normal'
    const stop = await (await call(api + '/deployments/stop', 'POST', { commandId: 'stop-first-request', expectedNodeId: nodeId, deploymentId: startInput.deploymentId })).json() as ClusterNodeJob
    assert.equal((await finished(stop)).phase, 'completed'); assert.equal(engine.stopCalls, 1)
    const nextInput = { ...startInput, commandId: 'start-second-request', deploymentId: 'deployment-two' }
    assert.equal((await finished(await (await call(api + '/deployments/start', 'POST', nextInput)).json() as ClusterNodeJob)).phase, 'completed')
    assert.equal((await finished(await (await call(api + '/canary', 'POST', { commandId: 'canary-second-request', expectedNodeId: nodeId, deploymentId: nextInput.deploymentId })).json() as ClusterNodeJob)).phase, 'completed')
    const secondKey = await (await call(api + '/endpoint-credentials')).json() as any
    assert.ok(secondKey.apiKey !== credentials.apiKey)
    assert.equal((await call('/v1/models', 'GET', undefined, credentials.apiKey)).status, 401, 'A stale deployment credential cannot silently use a new deployment of the same model repo')
    engine.mode = 'partial-json'
    const failedCanary = await (await call(api + '/canary', 'POST', { commandId: 'canary-incomplete-request', expectedNodeId: nodeId, deploymentId: nextInput.deploymentId })).json() as ClusterNodeJob
    assert.equal((await finished(failedCanary)).phase, 'failed'); assert.equal((await call(api + '/endpoint-credentials')).status, 409)
    engine.mode = 'normal'
    assert.equal(engine.maximumRequests, 1)
    await assert.rejects(createClusterNodeServer({ config: { schema: 1, dataDir: root, host: '0.0.0.0', port: 0 }, engine }), /TLS/)
    await assert.rejects(createClusterNodeServer({ config: { schema: 1, dataDir: root, host: '127.0.0.1', port: 0 }, engine }), /live process/)
    assert.equal((await fs.stat(path.join(root, 'management.token'))).mode & 0o077, 0)
    console.log('PASS real HTTP node: isolated credentials and identity, immutable staging, durable dedupe/conflicts, model admission, owned launch/canary, friendly proxy, complete/partial streams, deployment-key rotation, private receipts, TLS boundary and exclusive process lease')
  } finally { await node.close(); await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
