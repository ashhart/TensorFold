import type { SecretRotationMetadata } from '../base'
import { SyncCursor, ConnectorSourceRef, ConnectorEvent } from './core'

export type EmailSourceKind = 'eml_dir' | 'maildir' | 'imap' | 'outlook_local' | 'microsoft_graph'

export interface EmailSourceConfig {
  microsoftRevision?: number
  id: string
  label: string
  path: string
  kind: EmailSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null

  cursor?: SyncCursor

  host?: string
  port?: number
  tls?: boolean
  user?: string
  password?: string
  mailbox?: string

  fromAddress?: string
  // Optional per-source ephemeral system-prompt override (see ConnectorChannelPrompt).
  channelPrompt?: string
  secretRotations?: SecretRotationMetadata
}

export interface EmailConnectorSettings {
  sources: EmailSourceConfig[]
  maxMessages: number
  lastSyncAt: number | null

  detectOutlook?: boolean
}

export interface EmailMessage {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  subject: string
  from: string
  to: string
  date: number | null
  unread: boolean
  snippet: string
  size: number

  messageId?: string | null

  inReplyTo?: string | null
}

export interface EmailReadResult extends EmailMessage {
  headers: Record<string, string>
  body: string

  attachments?: EmailAttachment[]
}

export interface EmailAttachment {
  // BODYSTRUCTURE part identifier (e.g. "2", "1.2") the bytes were downloaded from.
  part: string
  filename: string | null
  contentType: string | null
  // Declared size from BODYSTRUCTURE (pre-decode estimate); the cache caps on actual bytes.
  size: number | null
  data: Uint8Array
}

export interface EmailThreadResult {
  root: EmailReadResult
  messages: EmailReadResult[]
  summary: string
}

export interface EmailSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface EmailListOptions {
  query?: string
  unreadOnly?: boolean
  limit?: number
}

export interface EmailDraftInput {
  accountId?: string
  mode?: 'draft' | 'send'
  to: string
  subject: string
  body: string
  threadKey?: string
  inReplyTo?: string
}

export interface EmailResolveInput {
  id: string
  note?: string
}

export interface EmailSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: EmailSourceConfig[]
  syncedAt: number

  attachmentsByEvent?: Record<string, EmailAttachment[]>
}
