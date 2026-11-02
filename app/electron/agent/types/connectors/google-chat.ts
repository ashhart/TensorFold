import type { SecretRotationMetadata } from '../base'
import { SyncCursor, ConnectorSourceRef, ConnectorEvent } from './core'

export interface GoogleChatSpace {
  // API resource name, e.g. "spaces/AAAAt3pNxxx". This is the allowlist + threadKey unit.
  name: string
  displayName: string
  // 'SPACE' (named room) | 'DIRECT_MESSAGE' (DM) | 'GROUP_CHAT' — best-effort from the API.
  spaceType: string | null
}

export interface GoogleChatSource {
  id: string
  label: string

  webhookUrl: string

  accessToken?: string

  baseUrl?: string

  allowedChannels: string
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

export interface GoogleChatConnectorSettings {
  sources: GoogleChatSource[]
  lastSyncAt: number | null
}

export interface GoogleChatMessage {
  // API message resource name, e.g. "spaces/AAAA/messages/MMMM.MMMM". Stable id.
  id: string
  sourceId: string
  sourceLabel: string
  // Space (the "channel"): resource name + best-effort display name.
  spaceName: string
  spaceDisplayName: string
  // Thread resource name within the space, e.g. "spaces/AAAA/threads/TTTT". null at top level.
  threadName: string | null
  senderId: string
  senderName: string
  senderEmail: string | null

  isBot: boolean
  date: number
  text: string
  snippet: string

  replyTo: string | null
  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}

export type GoogleChatReadResult = GoogleChatMessage & { raw: Record<string, unknown> }

export interface GoogleChatListOptions {
  query?: string
  // Restrict to a single space resource name (or bare id). Blank = all allowed spaces.
  space?: string
  limit?: number
}

export interface GoogleChatSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface GoogleChatSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: GoogleChatSource[]
  syncedAt: number
  // Advanced incremental cursor to persist back onto the originating source.cursor.
  cursor?: SyncCursor
}
