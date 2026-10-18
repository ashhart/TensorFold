import type { Room, Thread } from '../electron/agent/types'

export type TelegramRemoteTarget = { kind: 'chief' } | { kind: 'chat' | 'room'; id: string }

export interface TelegramRemoteSession {
  ownerKey: string
  authorizationRevision: string
  target: TelegramRemoteTarget
  lastUpdateId: number
  updatedAt: number
  recentMessageTimes: number[]
}

export interface TelegramRemoteSettings {
  enabled: boolean
  scope: 'selected' | 'all-eligible'
  threadIds: string[]
  roomIds: string[]
  /** Bump when changing access; an in-flight request must retain the same revision. */
  revision: number
  sessions: TelegramRemoteSession[]
}

export type TelegramRemoteConfigInput = Pick<TelegramRemoteSettings, 'enabled' | 'scope' | 'threadIds' | 'roomIds'>

export interface TelegramRemotePublicStatus extends TelegramRemoteConfigInput {
  revision: number
  eligibleThreads: Array<{ id: string; title: string }>
  eligibleRooms: Array<{ id: string; name: string; state: Room['state'] }>
}

export interface TelegramRemoteScope {
  threads: readonly Thread[]
  rooms: readonly Room[]
}

export interface TelegramRemoteTurnRequest {
  ownerKey: string
  target: TelegramRemoteTarget
  text: string
  scope: TelegramRemoteScope
  signal: AbortSignal
  /** Host must check before admitting a run and before exposing its result. */
  isAuthorized(): boolean
}

export interface TelegramRemoteTurnResult {
  status: 'completed' | 'queued' | 'busy' | 'incomplete' | 'error'
  text?: string
  runId?: string
}

export interface TelegramRemoteIngestResult {
  consumed: boolean
  reply: string
  skipped?: boolean
  /** Bound to the authenticated update, never an ID supplied by model output. */
  chatId?: string
  authorizationRevision?: string
  scopeRevision?: string
  target?: TelegramRemoteTarget
}

export interface TelegramVerificationResult {
  ok: boolean
  botId: string | null
  username: string | null
  error: string | null
}

export function defaultTelegramRemoteSettings(): TelegramRemoteSettings {
  return { enabled: false, scope: 'selected', threadIds: [], roomIds: [], revision: 0, sessions: [] }
}
