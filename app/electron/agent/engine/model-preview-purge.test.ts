import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineService } from './service'
import { makeHarness, seedModel, waitFor } from './test-fixtures/harness'
import { WorkspaceMutationGuard } from '../../main/host/workspace-mutation-guard'

// Real preview transport and supervisor; synthetic checkpoint and localhost-only fake engine.
for (const phase of ['loading', 'request'] as const) {
  const model = 'Fixture/PurgePreview'
  const h = await makeHarness({ FAKE_SERVE_LOAD_MS: phase === 'loading' ? '250' : '20', FAKE_FIRST_TOKEN_MS: '250' })
  const events: any[] = []
  const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 10, emit: (event) => events.push(event) })
  const guard = new WorkspaceMutationGuard()
  const operation = guard.reserveStart({ workspaceKeys: ['engine:local'] })
  let previewSettled = false, purgeSettled = false
  const original = path.join(h.hub, 'models--Fixture--PurgePreview', 'blobs', 'weights')
  const hash = async () => createHash('sha256').update(await fs.readFile(original)).digest('hex')
  try {
    await seedModel(h.hub, model)
    const before = await hash()
    service.install()
    const preview = service.tryModel(model).then(
      (result) => ({ result, error: null as Error | null }),
      (error) => ({ result: null, error: error as Error }),
    ).finally(() => { previewSettled = true; operation.release() })
    if (phase === 'loading') await waitFor(() => service.supervisor.status().state === 'starting', 10_000, 'actual preview model loading')
    else await waitFor(async () => (await h.records()).some((record) => record.kind === 'chat'), 10_000, 'actual preview request before delayed response')
    const draining = service.beginLearningPurge().then(() => { purgeSettled = true })
    const globalLease = guard.acquireGlobal('Synthetic local memory purge')
    assert.equal(purgeSettled, false, 'purge cannot report settlement synchronously while its owned preview still exists')
    assert.throws(() => service.tryModel(model), /erased/, 'new previews are synchronously refused throughout purge')
    await draining
    assert.equal(previewSettled, true, 'purge drainage completes only after the owned preview and its host operation settle')
    assert.equal(service.isLearningBusy(), false, 'no preview promise remains admitted after the purge barrier resolves')
    assert.equal(service.supervisor.activeRequestCount, 0)
    const outcome = await preview
    assert.equal(outcome.result, null, 'a late delayed response cannot be published as preview success')
    assert.match(outcome.error?.message ?? '', /cancel|abort/i)
    assert.equal(service.supervisor.activeRequestCount, 0, 'purge drainage releases the request lease before engine stop')
    const lease = await globalLease // The preview IPC operation can release; no global/engine lease deadlock.
    try { await service.purgeLearningData() } finally { lease.release() }
    assert.equal(service.supervisor.status().state, 'stopped')
    assert.equal(service.supervisor.live(), null)
    assert.equal(service.isLearningBusy(), false)
    assert.equal(await hash(), before, 'global private-memory purge leaves the original downloaded checkpoint intact')
    assert.ok(events.some((event) => event.kind === 'comparison-cleared'))
    service.endLearningPurge()
    // Reopening admission alone cannot restart an aborted model or resurrect its delayed output.
    await new Promise((resolve) => setTimeout(resolve, 300))
    assert.equal(service.supervisor.status().state, 'stopped')
    assert.equal(service.supervisor.activeRequestCount, 0)
    assert.equal(service.supervisor.live(), null)
    assert.equal((await service.snapshot()).performance, null)
  } finally { operation.release(); await service.supervisor.stop(); service.shutdown(); await h.cleanup() }
}
console.log('Actual standalone preview purge: loading and delayed request cancellation, captured promise drain, no global lease deadlock, no late output/runtime resurrection and original checkpoint preservation passed.')
