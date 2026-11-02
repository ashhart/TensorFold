import type { SecretRotationMetadata } from '../base'
import { SyncCursor, ConnectorSourceRef, ConnectorEvent } from './core'

export type SlackSourceKind = 'export_dir' | 'api_token'

export interface SlackSourceConfig {
  id: string
  label: string
  path: string
  kind: SlackSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null

  cursor?: SyncCursor

  token?: string
  channels?: string

  channelPrompt?: string
  secretRotations?: SecretRotationMetadata
}

export interface SlackConnectorSettings {
  sources: SlackSourceConfig[]
  maxMessages: number
  lastSyncAt: number | null
}

export interface SlackFileRef {
  urlPrivate: string
  name: string | null
  mimetype: string | null
}

export interface SlackMessage {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  channelId: string
  channelName: string
  senderId: string
  senderName: string
  ts: string
  threadTs?: string | null
  date: number | null
  text: string
  snippet: string

  files?: SlackFileRef[]
}

export interface SlackReadResult extends SlackMessage {
  raw: Record<string, unknown>
}

export interface SlackListOptions {
  query?: string
  channel?: string
  limit?: number
}

export interface SlackSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface SlackSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: SlackSourceConfig[]
  syncedAt: number
}
