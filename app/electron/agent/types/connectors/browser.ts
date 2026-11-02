import { ConnectorSourceRef, ConnectorEvent } from './core'

export type BrowserSourceKind = 'current_tab' | 'history_export'

export interface BrowserSourceConfig {
  id: string
  label: string
  kind: BrowserSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
  url?: string
  title?: string
  excerpt?: string
  capturedAt?: number | null
  path?: string
}

export interface BrowserConnectorSettings {
  sources: BrowserSourceConfig[]
  maxItems: number
  lastSyncAt: number | null
}

export interface BrowserItem {
  id: string
  sourceId: string
  sourceLabel: string
  kind: BrowserSourceKind
  url: string
  title: string
  excerpt: string
  capturedAt: number | null
  path?: string | null
}

export interface BrowserReadResult extends BrowserItem {
  content: string
}

export interface BrowserListOptions {
  query?: string
  limit?: number
}

export interface BrowserSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface BrowserSnapshotInput {
  url: string
  title?: string
  excerpt?: string
  capturedAt?: number | null
}

export interface BrowserSyncResult {
  events: ConnectorEvent[]
  items: BrowserItem[]
  sources: BrowserSourceConfig[]
  syncedAt: number
}
