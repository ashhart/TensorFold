import type { DocumentProcessingJob, DocumentProcessingSnapshot } from '../../../shared/document-processing'
import type { Settings } from '../types'
import type { DocumentEvidenceResult, DocumentLibraryApi, DocumentLibraryFile, DocumentLibraryPassage, DocumentLibrarySnapshot } from '../../../shared/document-library'

interface SyntheticRecord { file: DocumentLibraryFile; passage?: DocumentLibraryPassage; state: 'ready' | 'failed' | 'changed' | 'missing' }
const BOUNDS = { maxFiles: 1_500, maxCharacters: 32_000_000, maxPassages: 50_000 }
/** Browser demo data only; this module never reads paths, originals, profiles or providers. */
export function createMockDocumentLibraryApi(settingsGetter: () => Pick<Settings, 'documents'>): DocumentLibraryApi {
  const purged = new Map<string, string>()
  const indexed = new Map<string, { path: string; kind: string; sync: number }>()
  const signature = (source: Settings['documents']['sources'][number]) => `${source.path}|${source.kind}|${source.lastSyncAt ?? 0}`
  const records = (): SyntheticRecord[] => settingsGetter().documents.sources.filter(source => {
    if (!source.lastSyncAt || purged.get(source.id) === signature(source)) return false
    const prior = indexed.get(source.id)
    if (prior && (prior.path !== source.path || prior.kind !== source.kind) && prior.sync === source.lastSyncAt) return false
    indexed.set(source.id, { path: source.path, kind: source.kind, sync: source.lastSyncAt })
    return true
  }).flatMap(source => {
    const now = source.lastSyncAt!
    const make = (name: string, state: SyntheticRecord['state'], text: string, revision: string): SyntheticRecord => {
      const id = `demo-${source.id}-${name}`
      const file: DocumentLibraryFile = { id, sourceId: source.id, relativePath: name, scannedAt: now, status: state === 'ready' ? 'ready' : 'failed', retryable: state !== 'ready', revision, indexedCharacters: text.length, verification: state === 'ready' ? 'current' : state === 'failed' ? undefined : state, error: state === 'failed' ? 'Synthetic file exceeds its configured byte limit; reduce its size and retry.' : state === 'changed' ? 'Synthetic original changed; sync before reading its evidence.' : state === 'missing' ? 'Synthetic original is missing; restore it and retry.' : undefined }
      const passage: DocumentLibraryPassage | undefined = text ? { documentId: id, sourceId: source.id, sourceLabel: source.label, relativePath: name, title: name, revision, passageId: 'demo-passage', text, location: name.endsWith('.csv') ? { kind: 'table', rowStart: 3, rowEnd: 3, columnStart: 1, columnEnd: 2, charStart: 24, charEnd: 24 + text.length } : { kind: 'text', section: 'Final decision', lineStart: 200, lineEnd: 201, charStart: 4_000, charEnd: 4_000 + text.length }, verification: 'unverified' } : undefined
      return { file, passage, state }
    }
    return [make('brief.md', 'ready', 'The violet heron marker is near the end of this synthetic document.', 'a'.repeat(64)), make('metrics.csv', 'ready', 'Second,crimson egret', 'b'.repeat(64)), make('oversized.txt', 'failed', '', 'c'.repeat(64)), make('changed.md', 'changed', 'An old synthetic revision that must never be shown as current evidence.', 'd'.repeat(64)), make('missing.md', 'missing', 'Synthetic content whose original is now missing.', 'e'.repeat(64))]
  })
  const snapshot = (): DocumentLibrarySnapshot => {
    const settings = settingsGetter().documents
    const all = records()
    return {
      sources: settings.sources.map(source => {
        const files = all.filter(record => record.file.sourceId === source.id)
        return { id: source.id, label: source.label, path: source.path, enabled: source.enabled, lastScan: files.length ? { status: source.enabled ? 'partial' : 'disabled', ready: 2, failed: 3, unsupported: 0, skipped: 0, truncated: false, scannedAt: source.lastSyncAt! } : undefined, lastError: files.length ? 'Synthetic collection: 2 ready; 3 failed or unavailable.' : null }
      }),
      files: all.map(({ file }) => settings.sources.find(source => source.id === file.sourceId)?.enabled ? { ...file } : { ...file, status: 'skipped', verification: 'disabled', retryable: false, error: 'Source disabled; files are unavailable to search and evidence reads.' }),
      indexedAt: all.length ? settings.lastSyncAt ?? Math.max(...all.map(record => record.file.scannedAt)) : null, bounds: { ...BOUNDS },
    }
  }
  let jobs: DocumentProcessingJob[] = [], paused = false
  const processing = (): DocumentProcessingSnapshot => ({ jobs: structuredClone(jobs), paused, running: !paused && jobs.some(job => job.state === 'queued'), updatedAt: jobs.length ? Math.max(...jobs.map(job => job.updatedAt)) : null, concurrency: 1, checkpoint: 'between-files', truncated: false })
  const client: DocumentLibraryApi = {
    documentLibraryProcessing: async () => {
      if (!paused) {
        const job = jobs.find(item => item.state === 'queued')
        if (job) { const record = records().find(item => item.file.id === job.id); job.state = record?.state === 'ready' ? 'ready' : 'failed'; job.error = record?.file.error; job.updatedAt = Date.now() }
      }
      return processing()
    },
    documentLibraryProcess: async request => {
      if (request.action === 'start') { jobs = records().filter(record => !request.sourceId || record.file.sourceId === request.sourceId).map(({ file }) => ({ id: file.id, sourceId: file.sourceId, relativePath: file.relativePath, state: 'queued', updatedAt: Date.now() })); paused = false }
      else {
        if (!request.jobId && !request.sourceId) paused = request.action === 'pause' ? true : request.action === 'resume' || request.action === 'retry' ? false : paused
        for (const job of jobs.filter(item => (!request.jobId || item.id === request.jobId) && (!request.sourceId || item.sourceId === request.sourceId))) {
          if (request.action === 'pause' && ['queued', 'processing'].includes(job.state)) job.state = 'paused'
          if (request.action === 'cancel' && ['queued', 'processing', 'paused'].includes(job.state)) job.state = 'cancelled'
          if (request.action === 'resume' && job.state === 'paused' || request.action === 'retry' && ['failed', 'cancelled', 'skipped'].includes(job.state)) { job.state = 'queued'; paused = false }
          job.updatedAt = Date.now()
        }
      }
      return processing()
    },
    documentLibraryReadSource: async request => {
      const evidence = await client.documentLibraryReadEvidence(request)
      if (evidence.status !== 'ready') return evidence
      return { ...evidence, original: { kind: 'text', text: evidence.passage.text, truncated: false }, extractedText: evidence.passage.text, extractedTruncated: false, parser: 'plain-text' }
    },
    documentLibraryList: async () => snapshot(),
    documentLibrarySearch: async options => {
      const query = typeof options?.query === 'string' ? options.query.trim().slice(0, 256) : ''
      const limit = Math.max(1, Math.min(Number(options?.limit) || 20, 100))
      const allowed = new Set(settingsGetter().documents.sources.filter(source => source.enabled && (!options?.sourceId || source.id === options.sourceId)).map(source => source.id))
      const candidates = query ? records().filter(record => record.state === 'ready' && allowed.has(record.file.sourceId) && record.passage?.text.normalize('NFKC').toLowerCase().includes(query.normalize('NFKC').toLowerCase())).map(record => record.passage!) : []
      return { query, passages: structuredClone(candidates.slice(0, limit)), indexedAt: snapshot().indexedAt, truncated: candidates.length > limit }
    },
    documentLibraryReadEvidence: async request => {
      if (!request || !/^[a-f0-9]{64}$/.test(request.revision ?? '')) return { status: 'refused', message: 'A valid indexed passage and revision are required.' }
      const record = records().find(item => item.file.id === request.documentId)
      const source = settingsGetter().documents.sources.find(item => item.id === record?.file.sourceId)
      if (!record || !source?.enabled) return { status: 'unavailable', message: 'This collection is disabled, removed or purged.' }
      if (record.file.revision !== request.revision) return { status: 'changed', message: 'The citation uses an older indexed revision; search again.' }
      if (request.passageId !== record.passage?.passageId) return { status: 'refused', message: 'This passage is not in the indexed revision.' }
      if (record.state !== 'ready') return { status: record.state === 'failed' ? 'unavailable' : record.state, message: record.file.error! } as DocumentEvidenceResult
      return { status: 'ready', passage: structuredClone(record.passage!), verifiedAt: Date.now() }
    },
    documentLibraryPurge: async sourceId => { for (const source of settingsGetter().documents.sources) if (!sourceId || source.id === sourceId) purged.set(source.id, signature(source)); return snapshot() },
  }
  return client
}
