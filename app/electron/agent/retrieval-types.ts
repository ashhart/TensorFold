import type {
  BrowserConnectorSettings, CalendarConnectorSettings, ConnectorEvent, ConnectorSourceRef,
  DocumentConnectorSettings, EmailConnectorSettings, Mission, ProjectMemoryEntry, Room,
  SlackConnectorSettings, Thread, Ticket,
} from './types'

export const RETRIEVAL_INDEX_SCHEMA = 'vontra.retrieval-index/1'
export const RETRIEVAL_KINDS = ['code', 'doc', 'chat', 'email', 'summary', 'artifact'] as const

export type RetrievalKind = typeof RETRIEVAL_KINDS[number]

export interface RetrievalRecord {
  id: string
  kind: RetrievalKind
  title: string
  text: string
  sourceIds: string[]
  path?: string | null
  createdAt?: number | null
  metadata?: Record<string, string | number | boolean | null>
}

export interface RetrievalIndexStats {
  total: number
  byKind: Record<RetrievalKind, number>
  sourceIds: number
  codeScan?: RetrievalCodeScan
}

export type RetrievalCodeScanLimit = 'files' | 'directories' | 'entries' | 'directory_entries' | 'depth'

export interface RetrievalCodeScan {
  scope: string
  readFiles: number
  discoveredFiles: number
  visitedDirectories: number
  visitedEntries: number
  skippedFiles: number
  unreadableDirectories: number
  maxFiles: number
  maxDirectories: number
  maxEntries: number
  maxEntriesPerDirectory: number
  maxDepth: number
  limits: RetrievalCodeScanLimit[]
  truncated: boolean
}

export interface RetrievalIndex {
  schema: typeof RETRIEVAL_INDEX_SCHEMA
  generatedAt: number
  records: RetrievalRecord[]
  stats: RetrievalIndexStats
}

export interface RetrievalSearchOptions {
  query?: string | null
  kinds?: RetrievalKind[] | string | null
  limit?: number | null
}

export interface RetrievalSearchHit {
  record: RetrievalRecord
  score: number
  matchedTerms: string[]
}

export interface RetrievalSearchResult {
  query: string
  kinds: RetrievalKind[]
  generatedAt: number
  total: number
  hits: RetrievalSearchHit[]
  stats: RetrievalIndexStats
}

export interface RetrievalCodeFileInput {
  path: string
  content: string
  modifiedAt?: number | null
}

export interface RetrievalArtifactInput {
  id: string
  title?: string | null
  text?: string | null
  sourceIds?: string[]
  path?: string | null
  createdAt?: number | null
}

export interface RetrievalIndexInput {
  savedMemoryAllowed?: boolean
  currentThreadId?: string
  records?: Partial<RetrievalRecord>[]
  codeFiles?: RetrievalCodeFileInput[]
  documents?: ConnectorSourceRef[]
  connectorEvents?: ConnectorEvent[]
  threads?: Thread[]
  missions?: Mission[]
  rooms?: Room[]
  tickets?: Ticket[]
  projectMemory?: ProjectMemoryEntry[]
  artifacts?: RetrievalArtifactInput[]
}

export interface WorkspaceRetrievalIndexInput extends RetrievalIndexInput {
  // Host-owned live sources, refreshed after asynchronous workspace collection.
  recallThreads?: () => Thread[]
  recallProjectMemory?: () => ProjectMemoryEntry[]
  cwd: string
  path?: string | null
  query?: string | null
  kinds?: RetrievalKind[] | string | null
  maxCodeFiles?: number | null
  maxConnectorItems?: number | null
  emailSettings?: EmailConnectorSettings
  slackSettings?: SlackConnectorSettings
  calendarSettings?: CalendarConnectorSettings
  documentSettings?: DocumentConnectorSettings
  browserSettings?: BrowserConnectorSettings
}
