import { SyncCursor, ConnectorEvent } from './core'

export interface IMessageItem {
  id: string
  handle: string
  isFromMe: boolean
  text: string
  date: number | null
  hasAttachments: boolean
  chatId: string
  chatName: string

  guid?: string | null

  replyToId?: string | null
}

export type IMessageReadResult = IMessageItem

export interface IMessageConnectorSettings {
  transport?: 'local' | 'bluebubbles'
  blueBubblesUrl?: string
  blueBubblesPassword?: string
  bridgeCursorIdentity?: string
  enabled: boolean
  allowedChatIds: string
  maxMessages: number
  lastSyncAt: number | null
  lastError: string | null

  cursor?: SyncCursor
}

export interface IMessageSyncResult {
  events: ConnectorEvent[]
  messages: number
  syncedAt: number
  // Advanced incremental cursor to persist back onto IMessageConnectorSettings.cursor.
  cursor?: SyncCursor
}
