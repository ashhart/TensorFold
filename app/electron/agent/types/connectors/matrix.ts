import type { SecretRotationMetadata } from '../base'
import { SyncCursor, ConnectorSourceRef, ConnectorEvent } from './core'

export interface MatrixRoom {
  roomId: string
  alias: string | null
  name: string
}

export interface MatrixSource {
  id: string
  label: string

  baseUrl: string
  token: string

  allowPrivateNetwork?: boolean

  allowedRooms: string
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

export interface MatrixConnectorSettings {
  sources: MatrixSource[]
  lastSyncAt: number | null
}

export interface MatrixMessage {
  id: string // Matrix event_id ("$opaque")
  sourceId: string
  sourceLabel: string
  roomId: string
  roomName: string
  roomAlias: string | null
  senderId: string // Matrix user_id ("@user:server")
  senderName: string // resolved display name, falls back to the user_id
  isSelf: boolean // true when sender === the source account's own user_id
  date: number // origin_server_ts (ms)
  text: string
  snippet: string

  threadRootId: string | null

  replyTo: string | null

  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}

export type MatrixReadResult = MatrixMessage & { raw: Record<string, unknown> }

export interface MatrixListOptions {
  query?: string
  limit?: number
}

export interface MatrixSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface MatrixSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: MatrixSource[]
  syncedAt: number
}
