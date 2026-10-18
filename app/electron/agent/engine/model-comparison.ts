import type { EngineModelComparisonInput, EngineModelComparisonSnapshot } from '../types/model-comparison'
import type { EngineModelPreviewResult } from '../types/model-preview'
import { isRepoId } from './paths'

export function comparisonId(value: unknown): string {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9_-]{16,80}$/.test(value)) throw new Error('Expected a model comparison identifier.')
  return value
}
export function comparisonInput(value: unknown): EngineModelComparisonInput {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some((key) => !['id', 'models', 'prompt'].includes(key))) throw new Error('Expected two installed models and a plain comparison prompt.')
  const input = value as Record<string, unknown>
  const id = comparisonId(input.id)
  if (!Array.isArray(input.models) || input.models.length !== 2 || input.models.some((repo) => !isRepoId(repo)) || input.models[0] === input.models[1]) throw new Error('Choose two different installed models.')
  if (typeof input.prompt !== 'string' || !input.prompt.trim() || Buffer.byteLength(input.prompt, 'utf8') > 1024 || /\u0000/.test(input.prompt)) throw new Error('Enter a plain prompt up to 1 KB.')
  return { id, models: [...input.models] as [string, string], prompt: input.prompt.trim() }
}

export interface ModelComparisonDependencies {
  assertAllowed(): void
  reserve(): Promise<{ owner: symbol; previousModel: string | null; restore(): Promise<void>; release(): void }>
  validateModels(models: [string, string]): Promise<void>
  measure(model: string, prompt: string, signal: AbortSignal, owner: symbol): Promise<EngineModelPreviewResult>
}

// Ephemeral and local: comparison prompts/answers are never written to chat, memory or learning.
export class ModelComparisons {
  private current: { snapshot: EngineModelComparisonSnapshot; controller: AbortController; promise: Promise<EngineModelComparisonSnapshot> } | null = null
  private readonly seen = new Set<string>()
  constructor(private readonly deps: ModelComparisonDependencies) {}
  get currentId(): string | null { return this.current?.snapshot.id ?? null }
  get busy(): boolean { return this.current != null && ['running', 'restoring'].includes(this.current.snapshot.state) }
  snapshot(id: string): EngineModelComparisonSnapshot | null { comparisonId(id); return this.current?.snapshot.id === id ? structuredClone(this.current.snapshot) : null }
  cancel(id: string): boolean {
    comparisonId(id)
    if (this.current?.snapshot.id !== id || !this.busy) return false
    this.current.controller.abort(new Error('Model comparison cancelled.'))
    return true
  }
  clear(): void { if (this.busy) throw new Error('Drain the comparison before clearing its data.'); this.current = null; this.seen.clear() }
  settle(): Promise<unknown> { return this.current?.promise.catch(() => {}) ?? Promise.resolve() }
  start(raw: EngineModelComparisonInput): Promise<EngineModelComparisonSnapshot> {
    const input = comparisonInput(raw)
    this.deps.assertAllowed()
    if (this.busy) throw new Error('Finish or cancel the current model comparison first.')
    if (this.seen.has(input.id)) throw new Error('This comparison has already been submitted. Start a new comparison.')
    this.seen.add(input.id)
    if (this.seen.size > 64) this.seen.delete(this.seen.values().next().value!)
    const snapshot: EngineModelComparisonSnapshot = { id: input.id, revision: 1, state: 'running', startedAt: Date.now(), finishedAt: null, prompt: input.prompt, maxOutputTokens: 256, temperature: 0,
      rows: input.models.map((model) => ({ model, state: 'queued', result: null, error: null })) as EngineModelComparisonSnapshot['rows'], previousModel: null, restored: false, restoreError: null }
    const job = { snapshot, controller: new AbortController(), promise: Promise.resolve(snapshot) }
    this.current = job // Synchronous admission before any validation, load or IPC await.
    job.promise = this.run(job, input)
    return job.promise.then((result) => structuredClone(result))
  }
  private async run(job: NonNullable<ModelComparisons['current']>, input: EngineModelComparisonInput): Promise<EngineModelComparisonSnapshot> {
    let session: Awaited<ReturnType<ModelComparisonDependencies['reserve']>> | null = null
    const timer = setTimeout(() => job.controller.abort(new Error('Model comparison timed out. Try a smaller model or prompt.')), 360_000)
    const assertAllowed = () => { if (job.controller.signal.aborted) throw job.controller.signal.reason; this.deps.assertAllowed() }
    try {
      assertAllowed()
      session = await this.deps.reserve()
      job.snapshot.previousModel = session.previousModel
      job.snapshot.revision++
      assertAllowed()
      await this.deps.validateModels(input.models)
      assertAllowed()
      for (const row of job.snapshot.rows) {
        assertAllowed()
        row.state = 'running'; job.snapshot.revision++
        try { row.result = await this.deps.measure(row.model, input.prompt, job.controller.signal, session.owner); row.state = 'complete'; job.snapshot.revision++ }
        catch (error) {
          row.error = error instanceof Error ? error.message : 'The local model could not complete this comparison.'
          row.state = job.controller.signal.aborted ? 'cancelled' : 'failed'
          job.snapshot.revision++
          if (job.controller.signal.aborted) throw error
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'The model comparison could not finish.'
      for (const row of job.snapshot.rows) if (row.state === 'queued' || row.state === 'running') { row.state = job.controller.signal.aborted ? 'cancelled' : 'failed'; row.error = message }
    } finally {
      clearTimeout(timer)
      job.snapshot.state = 'restoring'
      job.snapshot.revision++
      try { if (session) { await session.restore(); job.snapshot.restored = true } }
      catch (error) { job.snapshot.restoreError = error instanceof Error ? error.message : 'The previous engine state could not be restored.' }
      finally { session?.release() }
      job.snapshot.state = job.controller.signal.aborted ? 'cancelled' : job.snapshot.rows.some((row) => row.state === 'failed') || job.snapshot.restoreError ? 'failed' : 'complete'
      job.snapshot.finishedAt = Date.now()
      job.snapshot.revision++
    }
    return job.snapshot
  }
}
