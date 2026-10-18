import { dialog } from 'electron'
import type { EngineService } from '../../../agent/engine/service'
import type { ModelBenchmarkInput } from '../../../../shared/model-benchmark'
import { isRepoId } from '../../../agent/engine/paths'
import { handle } from '../ipc'
import { auditEvent, requireSecurityRole } from '../audit'
import { saveBenchmarkExport } from '../model-benchmark-export'

export function benchmarkId(value: unknown): string {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9_-]{16,80}$/.test(value)) throw new Error('Choose a valid local benchmark run.')
  return value
}
export function benchmarkInput(value: unknown): ModelBenchmarkInput {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !['id', 'model', 'serial'].includes(key))) throw new Error('Choose an installed model and benchmark mode.')
  const input = value as Record<string, unknown>
  if (!isRepoId(input.model) || input.serial !== undefined && typeof input.serial !== 'boolean') throw new Error('Choose an installed model and benchmark mode.')
  return { id: benchmarkId(input.id), model: input.model, ...(input.serial !== undefined ? { serial: input.serial } : {}) }
}
export function registerModelBenchmarkIpc(engine: () => EngineService | null, options: {
  reserveOperation: () => { release(): void }
  assertCanBenchmark: () => void
}): void {
  const service = () => { const value = engine(); if (!value) throw new Error('The built-in engine is unavailable.'); return value }
  handle('engineBenchmark:start', async (_event, raw: unknown) => {
    requireSecurityRole('operator', 'benchmark a cached local model')
    const input = benchmarkInput(raw)
    options.assertCanBenchmark()
    const instance = service()
    const lease = options.reserveOperation()
    let released = false
    const release = () => { if (!released) { released = true; lease.release() } }
    let admitted = false
    try {
      instance.setBenchmarkAdmission(options.assertCanBenchmark)
      const started = instance.startBenchmark(input)
      const initial = await started
      admitted = true
      // Start returns promptly; keep purge/update admission closed through restoration.
      const settled = instance.settleBenchmarks(input.id)
      void settled.then(release, release)
      try { await auditEvent('engine.benchmark', { id: input.id, model: input.model, serial: input.serial ?? false }) }
      catch (error) { instance.cancelBenchmark(input.id); await settled.catch(() => {}); throw error }
      return await instance.benchmarkStatus(input.id) ?? initial
    } catch (error) {
      if (!admitted) release()
      throw error
    }
  })
  handle('engineBenchmark:status', async (_event, id?: unknown) => { requireSecurityRole('operator', 'read local benchmark progress'); return service().benchmarkStatus(id === undefined ? undefined : benchmarkId(id)) })
  handle('engineBenchmark:history', async () => { requireSecurityRole('operator', 'read local benchmark results'); return service().benchmarkHistory() })
  handle('engineBenchmark:cancel', async (_event, id: unknown) => { requireSecurityRole('operator', 'cancel a local benchmark'); return service().cancelBenchmark(benchmarkId(id)) })
  handle('engineBenchmark:export', async (_event, id: unknown) => {
    requireSecurityRole('owner', 'export a local benchmark receipt')
    const runId = benchmarkId(id)
    const receipt = await service().benchmarkReceiptForExport(runId)
    const json = JSON.stringify(receipt, null, 2) + '\n'
    const selected = await dialog.showSaveDialog({ title: 'Export benchmark JSON', defaultPath: `tensorfold-benchmark-${runId}.json`, filters: [{ name: 'Benchmark JSON', extensions: ['json'] }] })
    if (selected.canceled || !selected.filePath) return { saved: false }
    await saveBenchmarkExport(selected.filePath, json, () => requireSecurityRole('owner', 'export a local benchmark receipt'))
    return { saved: true }
  })
}
