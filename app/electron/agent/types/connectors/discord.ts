import type { SecretRotationMetadata } from '../base'
import { SyncCursor } from './core'

export interface DiscordChannel {
  id: string
  name: string
  guildId: string
  guildName: string
  threadId: string | null
  parentId?: string | null
}

export interface DiscordSource {
  id: string
  label: string
  token: string // protected secret
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

export interface DiscordConnectorSettings {
  sources: DiscordSource[]
  lastSyncAt: number | null
}

export interface DiscordMessage {
  id: string
  sourceId: string
  sourceLabel: string
  channelId: string
  channelName: string
  threadId: string | null
  threadName: string | null
  guildName: string
  senderId: string
  senderName: string
  isBot: boolean
  date: number
  text: string
  snippet: string
  replyTo: string | null

  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}

export type DiscordReadResult = DiscordMessage & { raw: Record<string, unknown> }
