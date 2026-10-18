import { isRepoId } from '../paths'
import type { ModelBenchmarkInput, ModelBenchmarkReceipt, ModelBenchmarkSnapshot } from '../../../../shared/model-benchmark'
import { benchmarkCells, benchmarkProtocolComplete, parseBenchmarkReceipt } from './receipt'
import type { BenchmarkProcessHandle } from './process'
import type { ModelBenchmarkStore, SavedModelBenchmark } from './storage'
import { BENCHMARK_ENGINE_VERSION } from './version'

export interface BenchmarkLease { previousModel: string | null; pause(): Promise<void>; restore(signal?: AbortSignal): Promise<void>; release(): void }
export interface ModelBenchmarkDependencies {
  assertAllowed(): void
  validateModel(model: string): Promise<void | { revision: string; configSha256: string; engineVersion: string }>
  reserve(): Promise<BenchmarkLease>
  workspace(): Promise<{ receiptPath: string; directory: string; remove(): Promise<void> }>
  process(input: Required<ModelBenchmarkInput>, workspace: { receiptPath: string; directory: string }, progress: (value: ModelBenchmarkSnapshot['progress']) => void, signal: AbortSignal): BenchmarkProcessHandle
  readReceipt(path: string, input: Required<ModelBenchmarkInput>): Promise<ModelBenchmarkReceipt | null>
  store: ModelBenchmarkStore
  changed?(snapshot: ModelBenchmarkSnapshot): void
  now?: () => number
}
export function modelBenchmarkId(id: unknown): string {
  if (typeof id !== 'string' || !/^[A-Za-z0-9_-]{16,80}$/.test(id)) throw new Error('Expected a valid local benchmark identifier.')
  return id
}
export function modelBenchmarkInput(value: unknown): Required<ModelBenchmarkInput> {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !['id', 'model', 'serial'].includes(key))) throw new Error('Choose an installed model for a local benchmark.')
  const input = value as Record<string, unknown>
  if (!isRepoId(input.model) || input.model.startsWith('TensorFoldLocal/') || input.serial !== undefined && typeof input.serial !== 'boolean') throw new Error('Benchmarks currently support cached Hugging Face checkpoints; linked-folder models are unavailable.')
  return { id: modelBenchmarkId(input.id), model: input.model, serial: input.serial === true }
}
interface Job { snapshot: ModelBenchmarkSnapshot; receipt: ModelBenchmarkReceipt | null; controller: AbortController; restoration: AbortController; process?: BenchmarkProcessHandle; promise: Promise<void> }
export class ModelBenchmarks {
  private current: Job | null = null
  private readonly completions = new Map<string, Promise<void>>()
  private readonly unsaved = new Map<string, SavedModelBenchmark>()
  private disposed = false
  constructor(private readonly deps: ModelBenchmarkDependencies) {}
  get currentId(): string | null { return this.current?.snapshot.id ?? null }
  get busy(): boolean { return Boolean(this.current && ['preparing', 'running', 'restoring'].includes(this.current.snapshot.state)) }
  private now(): number { return this.deps.now?.() ?? Date.now() }
  private change(job: Job): void { job.snapshot.revision++; try { this.deps.changed?.(structuredClone(job.snapshot)) } catch {} }
  async start(raw: ModelBenchmarkInput): Promise<ModelBenchmarkSnapshot> {
    const input = modelBenchmarkInput(raw)
    if (this.disposed || this.busy) throw new Error('Finish or cancel the current local benchmark before starting another.')
    if (this.completions.has(input.id) || this.unsaved.has(input.id)) throw new Error('This benchmark has already been submitted. Start a new run.')
    if (this.unsaved.size >= 50) throw new Error('Unsaved benchmark history is full. Export the remaining receipts before closing TensorFold.')
    this.deps.assertAllowed()
    const snapshot: ModelBenchmarkSnapshot = { ...input, state: 'preparing', revision: 0, startedAt: this.now(), finishedAt: null, progress: { completedRequests: null, totalRequests: 24, fixture: null, temperature: null, repeat: null, phase: 'loading' }, cells: benchmarkCells(), receiptAvailable: false, protocolComplete: false, previousModel: null, restored: null, restoreError: null, error: null, storageError: null, warnings: ['Results stay local; no benchmark data is uploaded.'], saved: false }
    const job: Job = { snapshot, receipt: null, controller: new AbortController(), restoration: new AbortController(), promise: Promise.resolve() }
    this.current = job
    job.promise = this.run(job, input)
    this.completions.set(input.id, job.promise)
    if (this.completions.size > 128) this.completions.delete(this.completions.keys().next().value!)
    this.change(job)
    return structuredClone(snapshot)
  }
  cancel(id: string, force = false): boolean {
    modelBenchmarkId(id)
    const job = this.current
    if (!job || job.snapshot.id !== id || !this.busy || job.snapshot.state === 'restoring') return false
    job.controller.abort(); job.process?.cancel(force)
    return true
  }
  settle(id?: string): Promise<void> {
    if (id !== undefined) { modelBenchmarkId(id); return this.completions.get(id) ?? Promise.resolve() }
    return this.current?.promise ?? Promise.resolve()
  }
  dispose(force = true): void { this.disposed = true; if (this.current && this.busy) { this.current.controller.abort(); this.current.restoration.abort(); this.current.process?.cancel(force) } }
  async history(): Promise<ModelBenchmarkSnapshot[]> {
    let saved: SavedModelBenchmark[] = []
    let unavailable = false
    try { saved = await this.deps.store.list() }
    catch { if (!this.unsaved.size) throw new Error('Local benchmark history could not be unlocked or validated. Restore access to this profile before continuing.'); unavailable = true }
    const records = new Map(saved.map(record => [record.snapshot.id, record]))
    for (const [id, record] of this.unsaved) records.set(id, record)
    return [...records.values()].sort((a, b) => (b.snapshot.finishedAt ?? b.snapshot.startedAt) - (a.snapshot.finishedAt ?? a.snapshot.startedAt)).map(record => {
      const snapshot = structuredClone(record.snapshot)
      if (unavailable) snapshot.warnings.push('Saved benchmark history is unavailable; only unsaved results from this session are shown.')
      return snapshot
    })
  }
  async snapshot(id?: string): Promise<ModelBenchmarkSnapshot | null> {
    if (id !== undefined) modelBenchmarkId(id)
    if (this.current && (id === undefined || this.current.snapshot.id === id)) return structuredClone(this.current.snapshot)
    const history = await this.history()
    return id === undefined ? history[0] ?? null : history.find(item => item.id === id) ?? null
  }
  async receiptForExport(id: string): Promise<ModelBenchmarkReceipt> {
    modelBenchmarkId(id)
    let receipt: ModelBenchmarkReceipt | null | undefined
    let snapshot: ModelBenchmarkSnapshot | undefined
    if (this.current?.snapshot.id === id && (this.current.receipt || this.busy)) { receipt = this.current.receipt; snapshot = this.current.snapshot }
    else if (this.unsaved.has(id)) { const record = this.unsaved.get(id)!; receipt = record.receipt; snapshot = record.snapshot }
    else {
      try { const record = (await this.deps.store.list()).find(item => item.snapshot.id === id); receipt = record?.receipt; snapshot = record?.snapshot }
      catch { throw new Error('The local benchmark receipt could not be unlocked or validated.') }
    }
    if (!receipt || !snapshot || ['preparing', 'running', 'restoring'].includes(snapshot.state)) throw new Error('This benchmark has no finished receipt available for export.')
    return parseBenchmarkReceipt(JSON.stringify(receipt), { model: snapshot.model, serial: snapshot.serial }, this.now())
  }
  private async run(job: Job, input: Required<ModelBenchmarkInput>): Promise<void> {
    let lease: BenchmarkLease | null = null
    let workspace: Awaited<ReturnType<ModelBenchmarkDependencies['workspace']>> | null = null
    let code: number | null = null
    let duplicate = false
    let unsafeCleanup = false
    let phase: 'validation' | 'process' | 'receipt' = 'validation'
    const assertAllowed = () => { if (job.controller.signal.aborted || this.disposed) throw new Error('Benchmark cancelled.'); this.deps.assertAllowed() }
    try {
      const history = await this.deps.store.list()
      if (history.some(record => record.snapshot.id === input.id)) { duplicate = true; throw new Error('Duplicate benchmark identifier.') }
      assertAllowed(); const checkpoint = await this.deps.validateModel(input.model); assertAllowed()
      lease = await this.deps.reserve(); job.snapshot.previousModel = lease.previousModel
      assertAllowed(); await lease.pause(); assertAllowed()
      workspace = await this.deps.workspace(); assertAllowed()
      phase = 'process'; job.snapshot.state = 'running'; this.change(job)
      job.process = this.deps.process(input, workspace, progress => { if (this.current === job) { job.snapshot.progress = structuredClone(progress); this.change(job) } }, job.controller.signal)
      if (job.controller.signal.aborted) job.process.cancel()
      const result = await job.process.done
      code = result.code
      unsafeCleanup = result.cleanupFailed === true
      if (result.cancelled) job.controller.abort()
      if (result.failed || result.overflow) job.snapshot.error = 'The bundled benchmark process could not complete safely.'
      phase = 'receipt'
      const receipt = await this.deps.readReceipt(workspace.receiptPath, input)
      job.receipt = receipt ? parseBenchmarkReceipt(JSON.stringify(receipt), { ...input, engineVersion: checkpoint?.engineVersion ?? BENCHMARK_ENGINE_VERSION }, this.now()) : null
      if (job.receipt) {
        if (checkpoint && (job.receipt.model.revision !== checkpoint.revision || job.receipt.model.config_sha256 !== checkpoint.configSha256)) { job.receipt = null; throw new Error('The cached checkpoint changed during the benchmark.') }
        job.snapshot.metadata = structuredClone({ runtime: job.receipt.runtime, model: job.receipt.model, hardware: job.receipt.hardware, settings: job.receipt.settings })
        job.snapshot.cells = benchmarkCells(job.receipt)
        job.snapshot.receiptAvailable = true
        job.snapshot.protocolComplete = benchmarkProtocolComplete(job.receipt)
        if (!job.snapshot.cells.every(cell => cell.complete)) job.snapshot.warnings.push('Incomplete cells retain their failures and have no aggregate speed.')
        if (!job.snapshot.protocolComplete) job.snapshot.warnings.push('This receipt is not a complete community comparison.')
      } else if (!job.controller.signal.aborted) job.snapshot.error = 'The local benchmark did not save a usable receipt.'
    } catch {
      if (!job.controller.signal.aborted) job.snapshot.error = phase === 'validation' ? 'The installed model or local benchmark history could not be validated.' : phase === 'receipt' ? 'The local benchmark receipt failed validation.' : 'The local benchmark could not finish.'
    } finally {
      job.snapshot.state = 'restoring'; this.change(job)
      if (lease) {
        if (unsafeCleanup) { job.snapshot.restored = false; job.snapshot.restoreError = 'The benchmark process did not stop cleanly. Restart TensorFold before loading another model.' }
        else if (this.disposed) { job.snapshot.restored = false; job.snapshot.restoreError = 'TensorFold closed before the previous local model could be restored.' }
        else try { await lease.restore(job.restoration.signal); job.snapshot.restored = true } catch { job.snapshot.restored = false; job.snapshot.restoreError = 'The previous local model could not be restored. Reload it from My models.' }
        lease.release()
      }
      const finalState = job.controller.signal.aborted ? 'cancelled' : job.snapshot.error || job.snapshot.restoreError || code !== 0 || !job.snapshot.cells.every(cell => cell.complete) ? 'failed' : 'complete'
      if (finalState === 'failed' && !job.snapshot.error && !job.snapshot.restoreError) job.snapshot.error = 'The local benchmark did not complete every measured attempt.'
      job.snapshot.finishedAt = this.now()
      const record: SavedModelBenchmark = { snapshot: { ...structuredClone(job.snapshot), state: finalState, saved: true }, receipt: job.receipt }
      try { if (!duplicate) { await this.deps.store.save(record); job.snapshot.saved = true } }
      catch { job.snapshot.storageError = job.receipt ? 'Benchmark results could not be saved to this profile. Export the validated receipt before closing TensorFold.' : 'Benchmark results could not be saved to this profile. Restore access to the local profile and try again.' }
      try { await workspace?.remove() } catch { job.snapshot.warnings.push('Temporary benchmark files could not be removed from this profile.') }
      job.snapshot.state = finalState
      this.change(job)
      if (job.snapshot.storageError && !duplicate) this.unsaved.set(input.id, { snapshot: structuredClone(job.snapshot), receipt: structuredClone(job.receipt) })
    }
  }
}
