import { ConnectorSourceRef } from './core'

export type DocumentSourceKind = 'file' | 'dir'

export interface DocumentSourceConfig {
  id: string
  label: string
  path: string
  kind: DocumentSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
  lastScan?: import("../../../../shared/document-library").DocumentScanSummary
  lastFileOutcomes?: import("../../../../shared/document-library").DocumentFileOutcome[]
}

export interface DocumentConnectorSettings {
  sources: DocumentSourceConfig[]
  maxDocuments: number
  maxDocumentBytes: number
  lastSyncAt: number | null
}

export interface DocumentItem {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  relativePath: string
  title: string
  excerpt: string
  size: number
  modifiedAt: number | null
  extension: string
}

export interface DocumentReadResult extends DocumentItem {
  content: string
  revision?: string
  sections?: import("../../../../shared/document-library").DocumentSection[]
  locationNote?: string
}

export interface DocumentListOptions {
  query?: string
  path?: string
  extension?: string
  limit?: number
}

export interface DocumentSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface DocumentSyncResult {
  documents: DocumentItem[]
  sources: DocumentSourceConfig[]
  syncedAt: number
}

export interface TableColumnSummary {
  name: string
  count: number
  min: number
  max: number
  average: number
}

export interface TableItem extends DocumentItem {
  rowCount: number
  columnCount: number
  columns: string[]
  delimiter: ',' | '\t'
  numericColumns: TableColumnSummary[]
}

export interface TableReadResult extends TableItem {
  rows: string[][]
}

export interface TableListOptions extends DocumentListOptions { }

export interface TableSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}
