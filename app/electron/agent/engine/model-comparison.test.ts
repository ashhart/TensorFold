import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { EngineService } from './service'
import { makeHarness, seedModel, waitFor } from './test-fixtures/harness'
import { comparisonId, comparisonInput } from './model-comparison'
import { defaultEngineModelProfile } from '../../../shared/model-profiles'
import { builtinEngineProvider } from './provider'
import { streamChat } from '../openai'

const a = 'Fixture/CompareA', b = 'Fixture/CompareB', previous = 'Fixture/Previous'
let serial = 0
const input = () => ({ id: `comparison_test_${++serial}`, models: [a, b] as [string, string], prompt: 'Synthetic same prompt: write a short greeting.' })
for (const bad of [null, {}, { ...input(), models: [a, a] }, { ...input(), models: [a, '/tmp/model'] }, { ...input(), prompt: '🎉'.repeat(257) }, { ...input(), tools: [] }, { ...input(), prompt: '\0' }]) assert.throws(() => comparisonInput(bad))
for (const bad of ['', '../secret', 12, 'a'.repeat(81)]) assert.throws(() => comparisonId(bad))
const h = await makeHarness({ FAKE_REPLY_TIMINGS: '1', FAKE_FIRST_TOKEN_MS: '100' })
const profiles = { [a]: { ...defaultEngineModelProfile(4096), temperature: 0.8, maxOutputTokens: 128, speculativeDecoding: 'off' as const }, [b]: { ...defaultEngineModelProfile(8192), temperature: 0.2, maxOutputTokens: 512, speculativeDecoding: 'off' as const } }
let hostBusy = false
const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, childEnv: h.env, appHome: h.root, emit: () => {}, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 10, modelSettings: (model) => profiles[model as keyof typeof profiles] ?? null, assertComparisonAllowed: () => { if (hostBusy) throw new Error('Synthetic local work is busy.') } })
const weightsHash = async () => createHash('sha256').update(await fs.readFile(path.join(h.hub, 'models--Fixture--CompareA', 'blobs', 'weights'))).digest('hex')
try {
  await Promise.all([seedModel(h.hub, a), seedModel(h.hub, b), seedModel(h.hub, previous)])
  const before = await weightsHash()
  service.install(); await service.load(previous)
  const settingsBefore = structuredClone(profiles)
  const requested = input(), result = await service.compareModels(requested)
  assert.equal(result.state, 'complete'); assert.equal(result.restored, true); assert.equal(result.previousModel, previous)
  assert.deepEqual(result.rows.map((row) => row.state), ['complete', 'complete'])
  assert.equal(service.supervisor.status().model, previous)
  assert.equal((await service.snapshot()).performance, null, 'comparison is not chat throughput')
  const requests = (await h.records()).filter((entry) => entry.kind === 'chat')
  assert.deepEqual(requests.map((entry) => entry.model), [a, b]); assert.ok(requests.every((entry) => entry.prompt === requested.prompt && entry.messages === 1 && entry.tools === undefined && entry.temperature === 0 && entry.max_tokens === 256 && entry.draft === false))
  assert.equal(result.rows[0].result?.decodeTokensPerSecond, 200); assert.equal(result.rows[0].result?.completionTokens, 3)
  assert.equal(result.rows[0].result?.profile?.contextTokens, 4096); assert.equal(result.rows[1].result?.profile?.contextTokens, 8192)
  assert.equal((await service.connection())?.model, previous)
  assert.throws(() => service.compareModels(input()), /shared API session/)
  await service.unload(); await service.load(previous)
  assert.equal(result.rows[0].result?.peakMemoryBytes, null); assert.equal(await weightsHash(), before)
  assert.deepEqual(profiles, settingsBefore, 'saved profiles and original checkpoint bytes are unchanged')
  const lease = await service.supervisor.acquire(previous, { start: false })
  const refused = await service.compareModels(input()); assert.equal(refused.state, 'failed'); assert.match(refused.rows[0].error ?? '', /active TensorFold requests/); lease.release()
  hostBusy = true; assert.throws(() => service.compareModels(input()), /busy/); hostBusy = false
  const work = input(), pending = service.compareModels(work)
  await waitFor(() => service.comparisonStatus(work.id)?.rows[0].state === 'running')
  assert.throws(() => service.compareModels(input()), /current model comparison/)
  await assert.rejects(service.connection(), /API sharing is paused/)
  await assert.rejects(service.load(b), /model comparison/)
  await assert.rejects(service.applyModelSettings(b, async () => {}, async () => {}), /model comparison/)
  await assert.rejects(service.tryModel(a), /lesson|preview|comparison/)
  await assert.rejects(service.setLearning(false), /model comparison/)
  const outsider = async () => { for await (const _ of streamChat({ provider: builtinEngineProvider(), model: a, messages: [{ role: 'user', content: 'Unrelated local request' }], tools: [] })) { /* must not run */ } }
  await assert.rejects(outsider(), /model comparison/)
  assert.equal(service.cancelComparison('unrelated_request_123'), false)
  assert.equal(service.cancelComparison(work.id), true)
  const cancelled = await pending; assert.equal(cancelled.state, 'cancelled'); assert.equal(cancelled.restored, true); assert.equal(service.supervisor.status().model, previous); assert.equal(service.supervisor.activeRequestCount, 0)
  assert.throws(() => service.compareModels(work), /already been submitted/)
  const unloaded = input(); await service.unload(); const zero = await service.compareModels(unloaded); assert.equal(zero.restored, true); assert.equal(service.supervisor.status().state, 'stopped')
  // Recheck host admission after async metadata, before fetch, and per stream chunk.
  await service.load(previous)
  const changed = input(), changing = service.compareModels(changed)
  await waitFor(() => service.comparisonStatus(changed.id)?.rows[0].state === 'running'); hostBusy = true
  const interrupted = await changing; hostBusy = false
  assert.equal(interrupted.state, 'failed'); assert.equal(interrupted.restored, true); assert.equal(service.supervisor.status().model, previous)
  // Quit and purge synchronously close admission, cancel and drain comparisons before cleanup.
  const closing = input(), closingWork = service.compareModels(closing)
  await waitFor(() => service.comparisonStatus(closing.id)?.rows[0].state === 'running'); await service.prepareToQuit()
  assert.equal((await closingWork).state, 'cancelled'); assert.equal(service.supervisor.activeRequestCount, 0)
  assert.equal(service.comparisonStatus(closing.id), null, 'quit clears retained comparison data after draining')
} finally { await service.supervisor.stop(); service.shutdown(); await h.cleanup() }

for (const mode of ['response-failure', 'load-failure'] as const) {
  const h = await makeHarness(mode === 'response-failure' ? { FAKE_CHAT_FAIL_MODEL: a, FAKE_FINISH_REASON: 'length' } : { FAKE_SERVE_FAIL_MODEL: a })
  const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, childEnv: h.env, appHome: h.root, emit: () => {}, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 10, readyTimeoutMs: 1000 })
  try {
    await Promise.all([seedModel(h.hub, a), seedModel(h.hub, b), seedModel(h.hub, previous)])
    service.install(); await service.load(previous)
    const result = await service.compareModels(input())
    assert.equal(result.rows[0].state, 'failed'); assert.equal(result.rows[1].state, 'complete', 'second model still runs after one model fails')
    if (mode === 'response-failure') assert.equal(result.rows[1].result?.finishReason, 'length', 'transport exposes reported reply cutoff without claiming a full answer')
    assert.equal(result.restored, true); assert.equal(service.supervisor.status().model, previous)
    assert.equal(service.supervisor.activeRequestCount, 0)
  } finally { await service.supervisor.stop(); service.shutdown(); await h.cleanup() }
}
// The actual memory-purge admission hook cancels/drains and drops retained prompt/results.
{
  const h = await makeHarness({ FAKE_FIRST_TOKEN_MS: '100' })
  const events: any[] = []
  const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, childEnv: h.env, appHome: h.root, emit: (event) => events.push(event), totalMemoryBytes: 32 * 1024 ** 3, pollMs: 10 })
  try {
    await Promise.all([seedModel(h.hub, a), seedModel(h.hub, b)])
    service.install()
    const request = input(), work = service.compareModels(request)
    await waitFor(() => service.comparisonStatus(request.id)?.rows[0].state === 'running')
    const draining = service.beginLearningPurge()
    assert.throws(() => service.compareModels(input()), /erased/)
    await draining; await work
    assert.equal(service.comparisonStatus(request.id), null)
    assert.ok(events.some((event) => event.kind === 'comparison-cleared'))
    assert.equal(service.supervisor.activeRequestCount, 0)
    service.endLearningPurge()
    assert.equal(service.comparisonStatus(request.id), null, 'reopening admission cannot republish pre-purge results')
  } finally { await service.supervisor.stop(); service.shutdown(); await h.cleanup() }
}
// A restoration failure stays visible and releases ownership for a user-directed recovery.
{
  const h = await makeHarness()
  const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, childEnv: h.env, appHome: h.root, emit: () => {}, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 10, readyTimeoutMs: 1000 })
  try {
    await Promise.all([seedModel(h.hub, a), seedModel(h.hub, b), seedModel(h.hub, previous)])
    service.install(); await service.load(previous); h.env.FAKE_SERVE_FAIL_MODEL = previous
    const result = await service.compareModels(input())
    assert.equal(result.state, 'failed'); assert.equal(result.restored, false); assert.ok(result.restoreError)
    assert.deepEqual(result.rows.map((row) => row.state), ['complete', 'complete'], 'completed answers survive a separate restore failure')
    delete h.env.FAKE_SERVE_FAIL_MODEL; await service.load(previous)
    assert.equal(service.supervisor.status().model, previous, 'restoration failure does not leave the sequence owner locked')
  } finally { await service.supervisor.stop(); service.shutdown(); await h.cleanup() }
}
console.log('Actual comparison service: same prompt/budget, honest metrics, profile isolation, cancellation, busy/owner/replay/host guards, unloaded and prior-model restoration, failure isolation and quit drain passed.')
