import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { DocumentConnectorSettings, DocumentReadResult, DocumentSourceConfig } from '../agent/types'
import type { DocumentFileOutcome } from '../../shared/document-library'
import type { DocumentProcessingJob, DocumentProcessingRequest, DocumentProcessingSnapshot } from '../../shared/document-processing'
import { stableDocumentId, readFileBytesBounded } from '../agent/document-parser'
import { isProtectedJsonEnvelope, protectJsonPayload, revealJsonPayload } from '../store/app-crypto'
import { atomicWriteJson } from '../store/write-infra'
import { collectCandidates, DOCUMENT_LIBRARY_BOUNDS, documentByteLimit, readDocumentFile, relativeDocumentPath } from './scanner'

type Job = DocumentProcessingJob & { path: string; source: DocumentSourceConfig; generation: number }
interface StoredQueue { version: 1; jobs: Job[]; paused: boolean; updatedAt: number | null; truncated: boolean }
interface Options {
  directory: string
  getSettings(): DocumentConnectorSettings
  commit(source: DocumentSourceConfig, outcome: DocumentFileOutcome, document?: DocumentReadResult): Promise<void>
  read?: typeof readDocumentFile
}
const empty = (): StoredQueue => ({ version: 1, jobs: [], paused: false, updatedAt: null, truncated: false })
const terminal = new Set(['ready', 'failed', 'unsupported', 'skipped', 'cancelled'])
const errorMessage = (error: unknown) => String((error as Error)?.message ?? error).slice(0, 300)
const matches = (left: DocumentSourceConfig, right: DocumentSourceConfig) => left.id === right.id && path.resolve(left.path) === path.resolve(right.path) && left.kind === right.kind && left.enabled

/** One physical parser at a time; control changes discard its result at the next file checkpoint. */
export class DocumentProcessingQueue {
  private data: StoredQueue | null = null
  private loading: Promise<StoredQueue> | null = null
  private persistence: Promise<void> = Promise.resolve()
  private operations: Promise<unknown> = Promise.resolve()
  private work: Promise<void> | null = null
  private accepting = true
  constructor(private options: Options) {}

  async snapshot(): Promise<DocumentProcessingSnapshot> {
    const data = await this.load()
    return { jobs: data.jobs.map(({ path: _path, source: _source, generation: _generation, ...job }) => ({ ...job })), paused: data.paused, running: !!this.work, updatedAt: data.updatedAt, concurrency: 1, checkpoint: 'between-files', truncated: data.truncated }
  }
  control(request: DocumentProcessingRequest): Promise<DocumentProcessingSnapshot> {
    const operation = this.operations.then(async () => {
      if (!this.accepting) throw new Error('Document processing is paused while workspace data is being changed.')
      if (!request || !['start', 'pause', 'resume', 'retry', 'cancel'].includes(request.action) || request.jobId != null && (typeof request.jobId !== 'string' || request.jobId.length > 256) || request.sourceId != null && (typeof request.sourceId !== 'string' || request.sourceId.length > 256)) throw new Error('Choose a valid document processing action.')
      const data = await this.load()
      if (request.action === 'start') await this.inventory(request.sourceId)
      else {
        const selected = data.jobs.filter(job => (!request.jobId || request.jobId === job.id) && (!request.sourceId || request.sourceId === job.sourceId))
        if (request.jobId && !selected.length) throw new Error('This document job is no longer available.')
        if (!request.jobId && !request.sourceId) data.paused = request.action === 'pause' ? true : request.action === 'resume' || request.action === 'retry' ? false : data.paused
        for (const job of selected) {
          if (request.action === 'pause' && !terminal.has(job.state)) { job.state = 'paused'; job.generation++ }
          if (request.action === 'cancel' && !terminal.has(job.state)) { job.state = 'cancelled'; job.generation++; job.error = 'Cancelled before its next file checkpoint; completed files remain indexed.' }
          if (request.action === 'resume' && job.state === 'paused' || request.action === 'retry' && ['failed', 'cancelled', 'skipped'].includes(job.state)) { job.state = 'queued'; job.generation++; delete job.error; if (request.jobId || request.sourceId) data.paused = false }
          job.updatedAt = Date.now()
        }
        data.updatedAt = Date.now()
        await this.save()
      }
      this.pump()
      return this.snapshot()
    })
    this.operations = operation.catch(() => {})
    return operation
  }
  async wait(): Promise<void> { await this.operations; while (this.work) await this.work }
  async quiesce(): Promise<void> {
    this.accepting = false
    await this.operations
    const data = await this.load().catch(() => null)
    if (!data) { await this.wait(); return }
    data.paused = true
    for (const job of data.jobs) if (!terminal.has(job.state)) { job.state = 'paused'; job.generation++; job.updatedAt = Date.now() }
    await this.save(); await this.wait()
  }
  resumeHost(): void { this.accepting = true }
  async clear(sourceId?: string): Promise<void> {
    const data = await this.load().catch(error => { if (sourceId) throw error; this.data = empty(); return this.data })
    for (const job of data.jobs) if (!sourceId || job.sourceId === sourceId) job.generation++
    data.jobs = sourceId ? data.jobs.filter(job => job.sourceId !== sourceId) : []
    await this.save(); await this.wait()
  }
  private async inventory(sourceId?: string): Promise<void> {
    const data = await this.load()
    if (data.jobs.some(job => !terminal.has(job.state))) throw new Error('Finish or cancel the current document queue before starting another scan.')
    const settings = structuredClone(this.options.getSettings())
    const sources = settings.sources.filter(source => source.enabled && (!sourceId || source.id === sourceId))
    if (sourceId && !sources.length) throw new Error('Choose an enabled document collection.')
    const jobs: Job[] = [], now = Date.now()
    let visited = 0, truncated = false
    for (const source of sources) {
      if (jobs.length >= DOCUMENT_LIBRARY_BOUNDS.maxFiles || visited >= DOCUMENT_LIBRARY_BOUNDS.maxFiles * 2) { truncated = true; break }
      const add = (file: string, state: Job['state'], error?: string) => jobs.push({ id: stableDocumentId(source.id, relativeDocumentPath(source, file)), sourceId: source.id, relativePath: relativeDocumentPath(source, file), path: file, source: { id: source.id, label: source.label, path: source.path, kind: source.kind, enabled: source.enabled, createdAt: source.createdAt, updatedAt: source.updatedAt, lastSyncAt: null }, state, updatedAt: now, error, generation: 0 })
      try {
        const inventory = await collectCandidates(source, now, DOCUMENT_LIBRARY_BOUNDS.maxFiles - jobs.length, DOCUMENT_LIBRARY_BOUNDS.maxFiles * 2 - visited)
        visited += inventory.visited; truncated ||= inventory.truncated
        for (const file of inventory.files) add(file, 'queued')
        for (const item of inventory.outcomes) {
          const file = source.kind === 'file' ? source.path : path.join(source.path, item.relativePath)
          add(file, item.status, item.error)
        }
      } catch (error) { add(source.path, 'failed', errorMessage(error)); visited++ }
    }
    if (!this.accepting) throw new Error('Document processing was interrupted before its inventory checkpoint.')
    for (const source of sources) if (!this.options.getSettings().sources.some(current => matches(current, source))) throw new Error('A document collection changed while its inventory was being prepared.')
    data.jobs = jobs; data.paused = false; data.updatedAt = now; data.truncated = truncated
    await this.save()
    for (const job of jobs.filter(job => terminal.has(job.state))) await this.commit(job)
  }
  private pump(): void {
    if (this.work || !this.accepting || !this.data || this.data.paused || !this.data.jobs.some(job => job.state === 'queued')) return
    this.work = this.run().finally(() => { this.work = null; this.pump() })
    // Failures are visible on the active job; no rejected background promises escape.
    void this.work.catch(() => {})
  }
  private async run(): Promise<void> {
    while (this.accepting && !this.data!.paused) {
      const job = this.data!.jobs.find(item => item.state === 'queued')
      if (!job) return
      const generation = job.generation
      job.state = 'processing'; job.updatedAt = Date.now()
      try {
        await this.save()
        const source = this.options.getSettings().sources.find(current => matches(current, job.source))
        if (!source) throw new Error('This collection was disabled, removed or changed; its queued file was not read.')
        const doc = await (this.options.read ?? readDocumentFile)(source, job.path, documentByteLimit(this.options.getSettings()))
        if (!doc) throw new Error('No readable text was extracted.')
        if (job.generation !== generation || !this.data!.jobs.includes(job) || !this.accepting) continue
        if (!this.options.getSettings().sources.some(current => matches(current, job.source))) throw new Error('The collection changed during extraction; its result was discarded.')
        await this.checkpoint(job, generation, doc)
      } catch (error) {
        if (job.generation !== generation || !this.data!.jobs.includes(job)) continue
        await this.checkpoint(job, generation, undefined, error)
      }
    }
  }
  private checkpoint(job: Job, generation: number, document?: DocumentReadResult, error?: unknown): Promise<void> {
    const next = this.operations.then(async () => {
      if (job.generation !== generation || !this.data!.jobs.includes(job) || !this.accepting) return
      if (error) { job.state = 'failed'; job.error = errorMessage(error) }
      try { await this.commit(job, document) }
      catch (failure) { job.state = 'failed'; job.error = errorMessage(failure); document = undefined }
      if (document) { job.state = 'ready'; job.revision = document.revision; delete job.error }
      job.updatedAt = Date.now(); this.data!.updatedAt = job.updatedAt
      await this.save()
    })
    this.operations = next.catch(() => {})
    return next
  }
  private async commit(job: Job, document?: DocumentReadResult): Promise<void> {
    if (!this.options.getSettings().sources.some(current => matches(current, job.source))) return
    await this.options.commit(job.source, { id: job.id, sourceId: job.sourceId, relativePath: job.relativePath, status: document ? 'ready' : ['failed', 'unsupported', 'skipped'].includes(job.state) ? job.state as 'failed' | 'unsupported' | 'skipped' : 'failed', error: document ? undefined : job.error, retryable: !document && job.state !== 'unsupported', scannedAt: Date.now(), revision: document?.revision }, document)
  }
  private async load(): Promise<StoredQueue> {
    if (this.data) return this.data
    if (this.loading) return this.loading
    this.loading = (async () => {
      try {
        if ((await fs.lstat(this.options.directory)).isSymbolicLink()) throw new Error('Managed document storage cannot be a symbolic link.')
        const envelope = JSON.parse((await readFileBytesBounded(path.join(this.options.directory, 'processing.json'), 3_000_000)).toString('utf8'))
        if (!isProtectedJsonEnvelope(envelope)) throw new Error('Document queue requires protected profile storage.')
        const data = revealJsonPayload<StoredQueue | null>(envelope, null)
        if (!data || data.version !== 1 || !Array.isArray(data.jobs) || data.jobs.length > DOCUMENT_LIBRARY_BOUNDS.maxFiles || typeof data.paused !== 'boolean') throw new Error('Invalid document queue schema.')
        const ids = new Set<string>()
        for (const job of data.jobs) {
          if (typeof job.id !== 'string' || job.id.length > 256 || ids.has(job.id) || typeof job.path !== 'string' || typeof job.relativePath !== 'string' || !job.source || job.source.id !== job.sourceId || typeof job.source.path !== 'string' || !['file', 'dir'].includes(job.source.kind) || !['queued', 'processing', 'paused', 'ready', 'failed', 'unsupported', 'skipped', 'cancelled'].includes(job.state)) throw new Error('Invalid document queue job.')
          ids.add(job.id); job.generation = Number.isSafeInteger(job.generation) ? job.generation : 0
          if (!terminal.has(job.state)) { job.state = 'paused'; job.error = 'Interrupted at a file checkpoint; resume to continue without reprocessing completed files.'; data.paused = true }
        }
        this.data = data
        return data
      } catch (error) {
        if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') { this.data = empty(); return this.data }
        throw new Error(`The protected document queue could not be read: ${errorMessage(error)}`)
      }
    })()
    try { return await this.loading } finally { this.loading = null }
  }
  private save(): Promise<void> {
    const envelope = protectJsonPayload(this.data!)
    if (Buffer.byteLength(JSON.stringify(envelope)) > 3_000_000) return Promise.reject(new Error('Document queue storage limit reached.'))
    const next = this.persistence.then(async () => {
      await fs.mkdir(this.options.directory, { recursive: true, mode: 0o700 })
      if ((await fs.lstat(this.options.directory)).isSymbolicLink()) throw new Error('Managed document storage cannot be a symbolic link.')
      await atomicWriteJson(path.join(this.options.directory, 'processing.json'), envelope)
    })
    this.persistence = next.catch(() => {})
    return next
  }
}
