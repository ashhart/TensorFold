import type { TelegramRemoteSettings } from '../../../../shared/telegram-remote'
import type { SecretRotationMetadata } from '../base'

// ---- Phone bot control (Telegram and signed WhatsApp Business Cloud ingress) ----
export type BotControlChannel = 'telegram' | 'whatsapp'

export interface TelegramBotControlSettings {
  remote?: TelegramRemoteSettings
  enabled: boolean
  botToken: string
  allowedChatIds: string
  // Telegram user IDs allowed to act, independently of the chat allowlist.
  allowedUserIds: string
  polling: boolean
  lastUpdateId: number | null
  lastPollAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface WhatsAppBotControlSettings {
  enabled: boolean
  accessToken: string
  phoneNumberId: string
  verifyToken: string
  // Meta App Secret used to verify X-Hub-Signature-256 over the raw webhook body.
  appSecret: string
  allowedChatIds: string
  lastEventAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface BotControlSettings {
  telegram: TelegramBotControlSettings
  whatsapp: WhatsAppBotControlSettings
}

export interface BotControlIngestInput {
  channel: BotControlChannel
  chatId: string
  // Authenticated channel principal (Telegram from.id or Meta messages[].from).
  senderId?: string
  // Stable, channel-issued delivery identifier used to reject replays.
  eventId?: string
  text: string
  senderName?: string
  payload?: Record<string, unknown>
}

export interface BotControlResult {
  ok: boolean
  channel: BotControlChannel
  chatId: string
  reply: string
  action: string
  roomId: string | null
}

export interface BotControlPollResult {
  ok: boolean
  processed: number
  skipped: number
  error: string | null
  lastUpdateId: number | null
}

export interface BotControlSessionMessage {
  route: 'central' | 'room'
  text: string
  roomId: string | null
  action: string
  createdAt: number
}

export interface BotControlSessionRoom {
  roomId: string | null
  action: string
  createdAt: number
}

export interface BotControlPendingConfirmation {
  code: string
  action: 'close' | 'remove' | 'cleanup'
  roomRef?: string | null
  // Confirmation is bound to the authenticated principal that requested it.
  principalId: string
  attemptsRemaining: number
  createdAt: number
  expiresAt: number
}

export interface BotControlSession {
  id: string
  channel: BotControlChannel
  chatId: string
  activeRoomId: string | null
  createdAt: number
  updatedAt: number
  pendingConfirmation?: BotControlPendingConfirmation | null
  // Recent authenticated channel delivery ids; persisted to make retries idempotent.
  processedEventIds?: string[]
  roomHistory: BotControlSessionRoom[]
  messageHistory: BotControlSessionMessage[]
}
