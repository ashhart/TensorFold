import type { SecretRotationMetadata } from '../base'
import { SyncCursor, ConnectorSourceRef, ConnectorEvent } from './core'

export interface MattermostChannel {
  id: string
  name: string // url-safe slug (channel.name)
  displayName: string // human label (channel.display_name)
  type: 'O' | 'P' | 'D' | 'G' | string
  teamId: string
  teamName: string
}

export interface MattermostSource {
  id: string
  label: string
  baseUrl: string // server origin, e.g. https://mm.example.com (no /api/v4 suffix)
  token: string

  allowPrivateNetwork?: boolean
  allowedChannels: string // csv allowlist of channel ids and/or slugs; blank = none; '*' = explicit all visible
  enabled: boolean
  lastSyncAt: number | null
  lastError: string | null

  cursor?: SyncCursor
  // Optional per-source ephemeral system-prompt override (see ConnectorChannelPrompt).
  channelPrompt?: string
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}

export interface MattermostConnectorSettings {
  sources: MattermostSource[]
  lastSyncAt: number | null
}

export interface MattermostMessage {
  id: string // post id
  sourceId: string
  sourceLabel: string
  channelId: string
  channelName: string // display name when known, else slug/id
  teamName: string
  senderId: string
  senderName: string
  isBot: boolean
  date: number // create_at (ms)
  text: string
  snippet: string

  rootId: string | null
  replyTo: string | null
  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}

export type MattermostReadResult = MattermostMessage & { raw: Record<string, unknown> }

export interface MattermostListOptions {
  query?: string
  channel?: string
  limit?: number
}

export interface MattermostSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface MattermostSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: MattermostSource[]
  syncedAt: number
}
