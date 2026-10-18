import assert from 'node:assert/strict'
import { promises as fs, writeFileSync, rmSync } from 'node:fs'
import path from 'node:path'
import type { EngineEvent } from '../types'
import { EngineService } from './service'
import { makeHarness, waitFor } from './test-fixtures/harness'

const SHA = 'd'.repeat(40)
const fetchHub = (async (input: Parameters<typeof fetch>[0]) => {
  const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url)
  if (/\/api\/models\/[^/]+\/[^/]+\/revision\//.test(url.pathname)) return Response.json({ sha: SHA, gated: false })
  if (url.pathname.endsWith(`/tree/${SHA}`)) return Response.json([{ type: 'file', path: 'model.safetensors', size: 65536 }])
  if (url.pathname.endsWith(`/resolve/${SHA}/config.json`)) return Response.json({ model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 } })
  return new Response('missing', { status: 404 })
}) as typeof fetch

async function main() {
  const h = await makeHarness({ FAKE_PULL_FILES: '1', FAKE_PULL_SIZE: '65536', FAKE_PULL_CHUNK: '4096', FAKE_PULL_DELAY_MS: '20' })
  const events: EngineEvent[] = []
  let corruptOnVerify = false
  const registry = path.join(h.root, 'engine', 'model-revisions.json')
  const service = new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, fetch: fetchHub, pollMs: 30, totalMemoryBytes: 32 * 1024 ** 3, emit(event) {
    events.push(event)
    if (event.kind !== 'pull' || event.pull.repo !== 'Example/CommitFailure') return
    if (event.pull.state === 'verifying' && corruptOnVerify) writeFileSync(registry, '{invalid')
    if (event.pull.state === 'failed' && corruptOnVerify) { rmSync(registry); corruptOnVerify = false }
  } })
  try {
    const first = await service.pull('Example/First')
    assert.equal(first.state, 'queued')
    const queued = await service.pull('Example/Removed')
    assert.equal(queued.state, 'queued')
    const next = await service.pull('Example/Next')
    assert.equal(next.state, 'queued')
    assert.equal(service.cancelPull('Example/Removed'), true)
    await assert.rejects(() => service.load('Example/Next'), /download|queued/i, 'queued model cannot load while its files will change')
    await waitFor(() => service.pullList().find((pull) => pull.repo === 'Example/Next')?.state === 'done', 10_000, 'serial service downloads')
    const pulls = (await h.records()).filter((record) => record.kind === 'pull')
    assert.deepEqual(pulls.map((record) => record.repo), ['Example/First', 'Example/Next'])
    assert.equal(pulls.every((record) => record.revision === SHA), true, 'queue preserves immutable revision selection')
    assert.equal(service.pullList().find((pull) => pull.repo === 'Example/Removed')?.state, 'cancelled')
    assert.ok(events.some((event) => event.kind === 'pull' && event.pull.state === 'verifying'))
    assert.equal((await h.records()).some((record) => record.kind === 'serve'), false, 'finished downloads never start a model automatically')

    await service.pull('Example/BlocksDeletion')
    await assert.rejects(() => service.remove('Example/First'), /download|queue/i, 'cache deletion cannot race any active or queued download')
    await waitFor(() => service.pullList().find((pull) => pull.repo === 'Example/BlocksDeletion')?.state === 'done', 10_000, 'deletion blocker to complete')
    assert.ok((await fs.readFile(registry, 'utf8')).includes('Example/First'))

    corruptOnVerify = true
    await service.pull('Example/CommitFailure')
    await service.pull('Example/AfterCommitFailure')
    await waitFor(() => service.pullList().find((pull) => pull.repo === 'Example/AfterCommitFailure')?.state === 'done', 10_000, 'queue recovery after registry failure')
    assert.equal(service.pullList().find((pull) => pull.repo === 'Example/CommitFailure')?.state, 'failed')
    const corruptEvents = events.filter((event) => event.kind === 'pull' && event.pull.repo === 'Example/CommitFailure')
    assert.equal(corruptEvents.some((event) => event.kind === 'pull' && event.pull.state === 'done'), false, 'done is never emitted before the durable registry succeeds')
  } finally {
    service.shutdown()
    await waitFor(() => !service.pullList().some((pull) => ['queued', 'running', 'verifying'].includes(pull.state)), 5000, 'download cleanup')
    await h.cleanup()
  }
  console.log('EngineService download queue: immutable serial jobs, queued cancel/load guards, global deletion guard, durable finalization and no automatic serve passed')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
