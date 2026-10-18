import type { DocumentProcessingRequest, DocumentProcessingSnapshot, DocumentSourceResult } from './document-processing'
/** Serializable document contracts; renderer imports types only. */
export type DocumentFileStatus = 'ready' | 'failed' | 'unsupported' | 'skipped'
export type DocumentScanStatus = 'ready' | 'partial' | 'error' | 'empty' | 'disabled'
export interface DocumentFileOutcome {
  id: string
  sourceId: string
  relativePath: string
  status: DocumentFileStatus
  error?: string
  retryable: boolean
  scannedAt: number
  revision?: string
}
export interface DocumentScanSummary {
  status: DocumentScanStatus
  ready: number
  failed: number
  unsupported: number
  skipped: number
  truncated: boolean
  scannedAt: number
}
export type DocumentLocation =
  | { kind: 'text'; section?: string; lineStart: number; lineEnd: number; charStart: number; charEnd: number }
  | { kind: 'table'; sheet?: string; sheetPath?: string; rowStart: number; rowEnd: number; columnStart: number; columnEnd: number; range?: string; charStart: number; charEnd: number }
  | { kind: 'extracted'; reason: string; charStart: number; charEnd: number }
export interface DocumentSection {
  id: string
  text: string
  location: DocumentLocation
}
export interface DocumentLibrarySource {
  id: string
  label: string
  path: string
  enabled: boolean
  lastScan?: DocumentScanSummary
  lastError?: string | null
}
export interface DocumentLibraryFile extends DocumentFileOutcome {
  title?: string
  size?: number
  modifiedAt?: number | null
  indexedCharacters?: number
  locationNote?: string
  verification?: 'current' | 'changed' | 'missing' | 'unavailable' | 'disabled'
}
export interface DocumentLibrarySnapshot {
  sources: DocumentLibrarySource[]
  files: DocumentLibraryFile[]
  indexedAt: number | null
  bounds: { maxFiles: number; maxCharacters: number; maxPassages: number }
}
export interface DocumentLibrarySearchOptions { query: string; sourceId?: string; limit?: number }
export interface DocumentLibraryPassage {
  documentId: string
  sourceId: string
  sourceLabel: string
  relativePath: string
  title: string
  revision: string
  passageId: string
  text: string
  location: DocumentLocation
  /** Originals are checked only when evidence is opened. */
  verification: 'unverified'
}
export interface DocumentLibrarySearchResult { query: string; passages: DocumentLibraryPassage[]; indexedAt: number | null; truncated: boolean; warnings?: string[] }
export interface DocumentEvidenceRequest { documentId: string; revision: string; passageId: string }
export type DocumentEvidenceResult =
  | { status: 'ready'; passage: DocumentLibraryPassage; verifiedAt: number }
  | { status: 'changed' | 'missing' | 'unavailable' | 'refused'; message: string }
export interface DocumentLibraryApi {
  documentLibraryList(): Promise<DocumentLibrarySnapshot>
  documentLibrarySearch(options: DocumentLibrarySearchOptions): Promise<DocumentLibrarySearchResult>
  documentLibraryReadEvidence(request: DocumentEvidenceRequest): Promise<DocumentEvidenceResult>
  documentLibraryPurge(sourceId?: string): Promise<DocumentLibrarySnapshot>
  documentLibraryProcessing(): Promise<DocumentProcessingSnapshot>
  documentLibraryProcess(request: DocumentProcessingRequest): Promise<DocumentProcessingSnapshot>
  documentLibraryReadSource(request: DocumentEvidenceRequest): Promise<DocumentSourceResult>
}
