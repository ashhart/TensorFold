import type { DocumentProcessingRequest } from '../../shared/document-processing'
import { DocumentProcessingQueue } from './processing'
import { readVerifiedSource } from './source-preview'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { DocumentConnectorSettings, DocumentReadResult, DocumentSourceConfig } from '../agent/types'
import type { DocumentFileOutcome, DocumentEvidenceRequest, DocumentEvidenceResult, DocumentLibraryFile, DocumentLibraryPassage, DocumentLibrarySearchOptions, DocumentLibrarySearchResult, DocumentLibrarySnapshot, DocumentLibrarySource } from '../../shared/document-library'
import { readFileBytesBounded } from '../agent/document-parser'
import { isProtectedJsonEnvelope, protectJsonPayload, revealJsonPayload } from '../store/app-crypto'
import { atomicWriteJson } from '../store/write-infra'
import { assertDocumentPath, DOCUMENT_LIBRARY_BOUNDS, documentByteLimit, readDocumentFile } from './scanner'

interface StoredDocument extends Omit<DocumentReadResult, 'content'> { indexedCharacters: number }
interface StoredSource extends DocumentLibrarySource { kind: DocumentSourceConfig['kind'] }
interface StoredLibrary { version: 1; indexedAt: number | null; sources: StoredSource[]; files: DocumentLibraryFile[]; documents: StoredDocument[] }
export interface DocumentLibraryOptions { directory: string; getSettings: () => DocumentConnectorSettings }
const MAX_CACHE_BYTES = 200_000_000
const empty = (): StoredLibrary => ({ version: 1, indexedAt: null, sources: [], files: [], documents: [] })
const fold = (text: string) => text.normalize('NFKC').toLowerCase()
const compatible = (source: DocumentSourceConfig, indexed: StoredSource) => source.id === indexed.id && path.resolve(source.path) === path.resolve(indexed.path) && source.kind === indexed.kind

/** Local lexical index; no model, embedding service, network or original file copy. */
export class DocumentLibrary {
  private data: StoredLibrary | null = null
  private queue: Promise<unknown> = Promise.resolve()
  private accepting = true
  private processingQueue: DocumentProcessingQueue
  constructor(private options: DocumentLibraryOptions) {
    this.processingQueue = new DocumentProcessingQueue({ ...options, commit: (source, outcome, document) => this.commitProcessed(source, outcome, document) })
  }

  processing() { return this.processingQueue.snapshot() }
  process(request: DocumentProcessingRequest) { return this.processingQueue.control(request) }
  waitForProcessing() { return this.processingQueue.wait() }
  readSource(request: DocumentEvidenceRequest) { return readVerifiedSource(this, request, this.options.getSettings) }

  async sync(): Promise<DocumentLibrarySnapshot> {
    await this.processingQueue.control({ action: 'start' })
    await this.processingQueue.wait()
    return this.list()
  }

  async drain(): Promise<void> { await this.processingQueue.wait(); await this.queue }
  async quiesce(): Promise<void> { await this.processingQueue.quiesce(); this.accepting = false; await this.queue; this.data = null }
  resume(): void { this.accepting = true; this.processingQueue.resumeHost() }

  list(): Promise<DocumentLibrarySnapshot> { return this.serial(async () => this.verifiedSnapshot(await this.load())) }

  search(options: DocumentLibrarySearchOptions, allowedSourceIds?: readonly string[]): Promise<DocumentLibrarySearchResult> {
    return this.serial(async () => {
      const data = await this.load()
      const query = typeof options?.query === 'string' ? options.query.trim().slice(0, 256) : ''
      const result: DocumentLibrarySearchResult = { query, passages: [], indexedAt: data.indexedAt, truncated: false }
      if (!query) return result
      const limit = Math.max(1, Math.min(Number(options?.limit) || 20, 100))
      const needle = fold(query)
      const sources = this.activeSources(data)
      for (const doc of data.documents) {
        const source = sources.get(doc.sourceId)
        if (!source || (allowedSourceIds !== undefined && !allowedSourceIds.includes(source.id)) || (options.sourceId && source.id !== options.sourceId)) continue
        if (!(doc.sections ?? []).some(section => fold(section.text).includes(needle))) continue
        const check = await this.verifyRevision(doc, source)
        if (check.status !== 'current') { (result.warnings ??= []).push(`${doc.relativePath}: ${check.message}`); continue }
        if (!this.activeSources(data).has(source.id)) continue
        for (const section of doc.sections ?? []) {
          if (!fold(section.text).includes(needle)) continue
          if (result.passages.length >= limit) { result.truncated = true; return result }
          result.passages.push(this.passage(doc, section.id, source.label))
        }
      }
      return result
    })
  }

  readEvidence(request: DocumentEvidenceRequest, allowedSourceIds?: readonly string[]): Promise<DocumentEvidenceResult> {
    return this.serial(async () => {
      if (!request || Object.keys(request).some(key => !['documentId', 'revision', 'passageId'].includes(key)) || typeof request.documentId !== 'string' || request.documentId.length > 256 || typeof request.passageId !== 'string' || request.passageId.length > 256 || !/^[a-f0-9]{64}$/.test(request.revision ?? '')) return { status: 'refused', message: 'A valid indexed passage and file revision are required.' }
      const data = await this.load()
      const doc = data.documents.find(item => item.id === request.documentId)
      const source = doc ? this.activeSources(data).get(doc.sourceId) : undefined
      if (!doc || !source || allowedSourceIds !== undefined && !allowedSourceIds.includes(source.id)) return { status: 'unavailable', message: 'This collection is disabled, removed, changed or purged; sync an enabled collection to read its evidence.' }
      if (doc.revision !== request.revision) return { status: 'changed', message: 'This citation uses an older indexed revision; search the current library again.' }
      if (!doc.sections?.some(section => section.id === request.passageId)) return { status: 'refused', message: 'This passage is not in the indexed document revision.' }
      try {
        await assertDocumentPath(source, doc.path)
        const current = await readDocumentFile(source, doc.path, documentByteLimit(this.options.getSettings()))
        if (!current || current.revision !== request.revision) return { status: 'changed', message: 'The original file has changed since indexing; sync this collection before using the citation.' }
        const currentSection = current.sections?.find(section => section.id === request.passageId)
        const cachedSection = doc.sections.find(section => section.id === request.passageId)!
        if (!currentSection || currentSection.text !== cachedSection.text || JSON.stringify(currentSection.location) !== JSON.stringify(cachedSection.location)) return { status: 'changed', message: 'The extraction or source coordinates have changed; sync this collection before reading evidence.' }
        if (!this.activeSources(data).has(source.id)) return { status: 'unavailable', message: 'The source changed or was disabled while evidence was being verified.' }
        return { status: 'ready', passage: this.passage({ ...doc, sections: [currentSection] }, request.passageId, source.label), verifiedAt: Date.now() }
      } catch (error) {
        const code = (error as NodeJS.ErrnoException)?.code
        if (code === 'ENOENT' || code === 'ENOTDIR') return { status: 'missing', message: 'The original file or its collection folder is missing; no cached evidence is shown.' }
        return { status: 'unavailable', message: `The original cannot be safely verified: ${String((error as Error)?.message ?? error).slice(0, 240)}` }
      }
    })
  }

  /** Removes cache for sources whose path/kind changed or whose configuration was removed. */
  reconcile(): Promise<DocumentLibrarySnapshot> {
    return this.serial(async () => {
      const data = await this.load()
      const current = this.options.getSettings()
      const sources = data.sources.filter(indexed => current.sources.some(source => compatible(source, indexed)))
      const ids = new Set(sources.map(source => source.id))
      const next = { ...data, sources, files: data.files.filter(file => ids.has(file.sourceId)), documents: data.documents.filter(doc => ids.has(doc.sourceId)) }
      await this.save(next)
      this.data = next
      return this.snapshot(next)
    })
  }

  async purge(sourceId?: string): Promise<DocumentLibrarySnapshot> {
    await this.processingQueue.clear(sourceId)
    return this.serial(async () => {
      const data = sourceId ? await this.load() : empty()
      const next = sourceId ? { ...data, sources: data.sources.filter(source => source.id !== sourceId), files: data.files.filter(file => file.sourceId !== sourceId), documents: data.documents.filter(doc => doc.sourceId !== sourceId) } : empty()
      await this.save(next)
      this.data = next
      return this.snapshot(next)
    }, true)
  }

  connectorItems(): Promise<{ documents: import('../agent/types').DocumentItem[]; sources: DocumentSourceConfig[] }> {
    return this.serial(async () => {
      const data = await this.load(), snapshot = await this.verifiedSnapshot(data)
      const ready = new Set(snapshot.files.filter(file => file.status === 'ready' && file.verification === 'current').map(file => file.id))
      const documents = data.documents.filter(doc => ready.has(doc.id)).map(({ sections: _sections, revision: _revision, indexedCharacters: _characters, locationNote: _note, ...metadata }) => metadata)
      const now = Date.now()
      const sources = this.options.getSettings().sources.map(source => {
        const indexed = snapshot.sources.find(item => item.id === source.id)
        return { ...source, lastScan: indexed?.lastScan, lastError: indexed?.lastError, lastFileOutcomes: snapshot.files.filter(file => file.sourceId === source.id).map(({ id, sourceId, relativePath, status, error, retryable, scannedAt, revision }) => ({ id, sourceId, relativePath, status, error, retryable, scannedAt, revision })), lastSyncAt: source.enabled ? now : source.lastSyncAt, updatedAt: now }
      })
      return { documents, sources }
    })
  }

  private commitProcessed(source: DocumentSourceConfig, outcome: DocumentFileOutcome, document?: DocumentReadResult): Promise<void> {
    return this.serial(async () => {
      const current = this.options.getSettings()
      if (!current.sources.some(active => active.enabled && compatible(active, source))) throw new Error('The document collection changed before its checkpoint; extraction was discarded.')
      const data = await this.load()
      const next = structuredClone(data)
      next.sources = next.sources.filter(indexed => indexed.id !== source.id)
      next.sources.push({ id: source.id, label: source.label, path: source.path, kind: source.kind, enabled: true })
      next.files = next.files.filter(file => file.id !== outcome.id)
      next.documents = next.documents.filter(doc => doc.id !== outcome.id)
      if (document) {
        const { content, ...metadata } = document
        const characters = next.documents.reduce((sum, doc) => sum + doc.indexedCharacters, 0) + content.length
        const passages = next.documents.reduce((sum, doc) => sum + (doc.sections?.length ?? 0), 0) + (document.sections?.length ?? 0)
        if (next.documents.length >= Math.max(1, Math.min(current.maxDocuments || 500, DOCUMENT_LIBRARY_BOUNDS.maxFiles)) || characters > DOCUMENT_LIBRARY_BOUNDS.maxCharacters || passages > DOCUMENT_LIBRARY_BOUNDS.maxPassages) throw new Error('Managed library content limit reached; narrow collections or adjust the document count before retrying.')
        next.documents.push({ ...metadata, indexedCharacters: content.length })
      }
      if (next.files.length >= DOCUMENT_LIBRARY_BOUNDS.maxFiles) throw new Error('Managed library file limit reached; clear unused collections before retrying.')
      next.files.push({ ...outcome, title: document?.title, size: document?.size, modifiedAt: document?.modifiedAt, indexedCharacters: document?.content.length, locationNote: document?.locationNote })
      const indexed = next.sources.find(item => item.id === source.id)!
      const files = next.files.filter(file => file.sourceId === source.id)
      const count = (status: DocumentFileOutcome['status']) => files.filter(file => file.status === status).length
      indexed.lastScan = { status: files.every(file => file.status === 'ready') ? 'ready' : files.some(file => file.status === 'ready') ? 'partial' : 'error', ready: count('ready'), failed: count('failed'), unsupported: count('unsupported'), skipped: count('skipped'), truncated: false, scannedAt: outcome.scannedAt }
      next.indexedAt = outcome.scannedAt
      await this.save(next); this.data = next
    })
  }

  private async verifyRevision(doc: StoredDocument, source: DocumentSourceConfig): Promise<{ status: 'current' | 'changed' | 'missing' | 'unavailable'; message?: string }> {
    try {
      await assertDocumentPath(source, doc.path)
      const bytes = await readFileBytesBounded(doc.path, documentByteLimit(this.options.getSettings()))
      if (createHash('sha256').update(bytes).digest('hex') !== doc.revision) return { status: 'changed', message: 'Original changed; sync this collection before using its evidence.' }
      return { status: 'current' }
    } catch (error) {
      const code = (error as NodeJS.ErrnoException)?.code
      if (code === 'ENOENT' || code === 'ENOTDIR') return { status: 'missing', message: 'Original file or collection folder is missing.' }
      return { status: 'unavailable', message: `Original cannot be safely verified: ${String((error as Error)?.message ?? error).slice(0, 160)}` }
    }
  }
  private async verifiedSnapshot(data: StoredLibrary): Promise<DocumentLibrarySnapshot> {
    const snapshot = this.snapshot(data)
    const active = this.activeSources(data)
    for (const file of snapshot.files) {
      const doc = data.documents.find(item => item.id === file.id)
      const source = active.get(file.sourceId)
      if (!doc || !source || file.status !== 'ready') continue
      const check = await this.verifyRevision(doc, source)
      file.verification = check.status
      if (check.status !== 'current') { file.status = 'failed'; file.error = check.message; file.retryable = true }
    }
    const latest = this.snapshot(data)
    snapshot.sources = latest.sources
    snapshot.files = snapshot.files.filter(file => latest.files.some(current => current.id === file.id)).map(file => {
      const current = latest.files.find(item => item.id === file.id)!
      return current.status === 'skipped' ? current : file
    })
    return snapshot
  }

  private activeSources(data: StoredLibrary): Map<string, DocumentSourceConfig> {
    const current = this.options.getSettings()
    return new Map(current.sources.filter(source => source.enabled && data.sources.some(indexed => compatible(source, indexed))).map(source => [source.id, source]))
  }
  private snapshot(data: StoredLibrary): DocumentLibrarySnapshot {
    const current = this.options.getSettings()
    const matching = new Set(current.sources.filter(source => data.sources.some(indexed => compatible(source, indexed))).map(source => source.id))
    return {
      sources: current.sources.map(source => { const indexed = data.sources.find(item => compatible(source, item)); return { id: source.id, label: source.label, path: source.path, enabled: source.enabled, lastScan: indexed?.lastScan, lastError: indexed?.lastError } }),
      files: data.files.filter(file => matching.has(file.sourceId)).map(file => current.sources.find(source => source.id === file.sourceId)?.enabled ? { ...file } : { ...file, status: 'skipped', error: 'Source disabled; its files are unavailable to search and evidence reads.', retryable: false, verification: 'disabled' }),
      indexedAt: data.indexedAt, bounds: { ...DOCUMENT_LIBRARY_BOUNDS },
    }
  }
  private passage(doc: StoredDocument, passageId: string, label: string): DocumentLibraryPassage {
    const section = doc.sections!.find(item => item.id === passageId)!
    return { documentId: doc.id, sourceId: doc.sourceId, sourceLabel: label, relativePath: doc.relativePath, title: doc.title, revision: doc.revision!, passageId, text: section.text, location: { ...section.location }, verification: 'unverified' }
  }
  private serial<T>(work: () => Promise<T>, whileQuiesced = false): Promise<T> { if (!this.accepting && !whileQuiesced) return Promise.reject(new Error('Document library is paused while workspace data is being changed.')); const result = this.queue.then(work, work); this.queue = result.catch(() => {}); return result }
  private async load(): Promise<StoredLibrary> {
    if (this.data) return this.data
    try {
      if ((await fs.lstat(this.options.directory)).isSymbolicLink()) throw new Error('The managed library directory cannot be a symbolic link.')
      const bytes = await readFileBytesBounded(path.join(this.options.directory, 'index.json'), MAX_CACHE_BYTES)
      const envelope = JSON.parse(bytes.toString('utf8')) as unknown
      if (!isProtectedJsonEnvelope(envelope)) throw new Error('Document cache must use protected profile storage.')
      const data = revealJsonPayload<StoredLibrary | null>(envelope, null)
      if (!data) throw new Error('Document cache cannot be decrypted with the active profile key.')
      if (data?.version !== 1 || !Array.isArray(data.sources) || !Array.isArray(data.files) || !Array.isArray(data.documents) || data.sources.length > 5_000 || data.files.length > DOCUMENT_LIBRARY_BOUNDS.maxFiles || data.documents.length > DOCUMENT_LIBRARY_BOUNDS.maxFiles) throw new Error('Invalid library schema or limits.')
      let characters = 0, passages = 0
      for (const source of data.sources) if (typeof source.id !== 'string' || typeof source.path !== 'string' || !['file', 'dir'].includes(source.kind)) throw new Error('Invalid collection metadata.')
      for (const file of data.files) if (typeof file.id !== 'string' || typeof file.sourceId !== 'string' || typeof file.relativePath !== 'string' || !['ready', 'failed', 'unsupported', 'skipped'].includes(file.status)) throw new Error('Invalid file metadata.')
      for (const doc of data.documents) {
        if (typeof doc.id !== 'string' || typeof doc.path !== 'string' || typeof doc.relativePath !== 'string' || !/^[a-f0-9]{64}$/.test(doc.revision ?? '') || !Array.isArray(doc.sections) || !Number.isSafeInteger(doc.indexedCharacters) || doc.indexedCharacters < 0) throw new Error('Invalid indexed document metadata.')
        characters += doc.indexedCharacters
        passages += doc.sections.length
        for (const section of doc.sections) if (typeof section.id !== 'string' || typeof section.text !== 'string' || section.text.length > 1_800 || !section.location || !['text', 'table', 'extracted'].includes(section.location.kind)) throw new Error('Invalid indexed passage.')
      }
      if (characters > DOCUMENT_LIBRARY_BOUNDS.maxCharacters || passages > DOCUMENT_LIBRARY_BOUNDS.maxPassages) throw new Error('Library content limit exceeded.')
      this.data = data
      return data
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') { this.data = empty(); return this.data }
      throw new Error(`The local document index could not be read; sync or clear it to rebuild: ${String((error as Error)?.message ?? error).slice(0, 180)}`)
    }
  }
  private async save(data: StoredLibrary): Promise<void> {
    const protectedData = protectJsonPayload(data)
    if (Buffer.byteLength(JSON.stringify(protectedData), 'utf8') > MAX_CACHE_BYTES) throw new Error('Managed document index exceeds its storage limit; narrow the configured collections.')
    await fs.mkdir(this.options.directory, { recursive: true, mode: 0o700 })
    if ((await fs.lstat(this.options.directory)).isSymbolicLink()) throw new Error('The managed library directory cannot be a symbolic link.')
    await atomicWriteJson(path.join(this.options.directory, 'index.json'), protectedData)
  }
}
