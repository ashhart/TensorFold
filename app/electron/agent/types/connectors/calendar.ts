import { ConnectorSourceRef } from './core'

export type CalendarSourceKind = 'ics_file' | 'ics_dir'

export interface CalendarSourceConfig {
  id: string
  label: string
  path: string
  kind: CalendarSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
}

export interface CalendarConnectorSettings {
  sources: CalendarSourceConfig[]
  maxEvents: number
  lastSyncAt: number | null
}

export interface CalendarEvent {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  title: string
  startAt: number | null
  endAt: number | null
  allDay: boolean
  location: string
  description: string
  organizer: string
  attendees: string[]
  status: string
}

export interface CalendarReadResult extends CalendarEvent {
  uid: string
  raw: Record<string, string>
}

export interface CalendarListOptions {
  query?: string
  from?: number | string
  to?: number | string
  day?: string
  limit?: number
}

export interface CalendarSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface CalendarSyncResult {
  events: CalendarEvent[]
  sources: CalendarSourceConfig[]
  syncedAt: number
}

export interface CalendarMutationInput {
  mode?: 'create' | 'update'
  id?: string
  sourceId?: string
  title: string
  startAt?: number | string | null
  endAt?: number | string | null
  allDay?: boolean
  location?: string
  description?: string
  organizer?: string
  attendees?: string[]
}
