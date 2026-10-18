import type { DocumentEvidenceResult, DocumentLibraryPassage } from './document-library'

export type DocumentJobState = 'queued' | 'processing' | 'paused' | 'ready' | 'failed' | 'unsupported' | 'skipped' | 'cancelled'
export interface DocumentProcessingJob {
  id: string
  sourceId: string
  relativePath: string
  state: DocumentJobState
  updatedAt: number
  error?: string
  revision?: string
}
export interface DocumentProcessingSnapshot {
  jobs: DocumentProcessingJob[]
  paused: boolean
  running: boolean
  updatedAt: number | null
  concurrency: 1
  checkpoint: 'between-files'
  truncated: boolean
}
export interface DocumentProcessingRequest { action: 'start' | 'pause' | 'resume' | 'retry' | 'cancel'; jobId?: string; sourceId?: string }
export type DocumentSourceResult = Exclude<DocumentEvidenceResult, { status: 'ready' }> | {
  status: 'ready'
  passage: DocumentLibraryPassage
  verifiedAt: number
  original: { kind: 'text'; text: string; truncated: boolean } | { kind: 'unavailable'; reason: string }
  extractedText: string
  extractedTruncated: boolean
  parser: 'plain-text' | 'workbook' | 'pdf-text' | 'office-text'
  locationNote?: string
}
