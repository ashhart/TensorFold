import { createHash } from 'node:crypto'
import type { Room, TelegramBotControlSettings, Thread } from '../types'
import type { TelegramRemoteScope, TelegramRemoteSettings, TelegramRemoteTarget } from '../../../shared/telegram-remote'
import { defaultTelegramRemoteSettings } from '../../../shared/telegram-remote'
import { parseAllowedBotChatIds } from '../bot-control'
import { canRecallThread } from '../session-recall'
import { parseThreadMemoryPolicy } from '../../../shared/thread-memory-policy'

export interface TelegramRemoteOwner {
  ownerKey: string
  chatId: string
  senderId: string
  updateId: number
  text: string
}

export type RemoteTelegramSettings = TelegramBotControlSettings & { remote?: TelegramRemoteSettings }

export function remoteSettings(settings: RemoteTelegramSettings): TelegramRemoteSettings {
  const remote = settings.remote
  if (!remote || typeof remote.enabled !== 'boolean' || !['selected', 'all-eligible'].includes(remote.scope)
    || !Number.isSafeInteger(remote.revision) || remote.revision < 0 || !Array.isArray(remote.sessions) || remote.sessions.length > 100
    || [remote.threadIds, remote.roomIds].some((ids) => !Array.isArray(ids) || ids.length > 500 || ids.some((id) => typeof id !== 'string' || !id || id.length > 160))) return defaultTelegramRemoteSettings()
  for (const session of remote.sessions) {
    if (!session || typeof session.ownerKey !== 'string' || typeof session.authorizationRevision !== 'string'
      || !session.target || !['chief', 'chat', 'room'].includes(session.target.kind)
      || session.target.kind !== 'chief' && (typeof session.target.id !== 'string' || !session.target.id || session.target.id.length > 160)
      || !Number.isSafeInteger(session.lastUpdateId) || session.lastUpdateId < 0 || !Number.isFinite(session.updatedAt)
      || !Array.isArray(session.recentMessageTimes) || session.recentMessageTimes.length > 12 || session.recentMessageTimes.some((time) => !Number.isFinite(time))) return defaultTelegramRemoteSettings()
  }
  return remote
}

export function telegramAuthorizationRevision(settings: RemoteTelegramSettings): string {
  const remote = remoteSettings(settings)
  return createHash('sha256').update(JSON.stringify([
    settings.enabled, settings.polling, settings.botToken.trim(),
    [...parseAllowedBotChatIds(settings.allowedChatIds)].sort(),
    [...parseAllowedBotChatIds(settings.allowedUserIds)].sort(),
    remote.enabled, remote.scope, [...remote.threadIds].sort(), [...remote.roomIds].sort(), remote.revision,
  ])).digest('hex')
}

/** Only fresh ordinary messages from an allowlisted human in their own private chat. */
export function authenticateTelegramOwner(update: unknown, settings: RemoteTelegramSettings): TelegramRemoteOwner | null {
  if (!settings.enabled || !settings.botToken.trim() || !remoteSettings(settings).enabled) return null
  const item = update as { update_id?: unknown; message?: { chat?: { id?: unknown; type?: unknown }; from?: { id?: unknown; is_bot?: unknown }; text?: unknown } }
  const message = item?.message
  if (!Number.isSafeInteger(item?.update_id) || (item.update_id as number) < 0 || !message || message.chat?.type !== 'private' || message.from?.is_bot !== false) return null
  const chatId = String(message.chat.id ?? '')
  const senderId = String(message.from.id ?? '')
  if (!/^[1-9]\d{0,19}$/.test(chatId) || chatId !== senderId) return null
  if (!parseAllowedBotChatIds(settings.allowedChatIds).has(chatId) || !parseAllowedBotChatIds(settings.allowedUserIds).has(senderId)) return null
  if (typeof message.text !== 'string' || !message.text.trim() || message.text.length > 12_000) return null
  return { ownerKey: `telegram:${chatId}:${senderId}`, chatId, senderId, updateId: item.update_id as number, text: message.text.trim() }
}

export function eligibleTelegramThread(thread: Thread, threads: readonly Thread[], seen = new Set<string>()): boolean {
  if (seen.has(thread.id) || seen.size > 8) return false
  if (thread.kind !== 'chat' || !canRecallThread(thread) || thread.archivedAt || thread.bot || thread.phoneLocalOnly || thread.googleLocalOnly || thread.orchestratorHostId) return false
  try { if (!parseThreadMemoryPolicy(thread.memoryPolicy).contribute) return false } catch { return false }
  seen.add(thread.id)
  for (const sourceId of [thread.parentId, thread.messageEditSource?.threadId]) {
    if (!sourceId) continue
    const source = threads.find((candidate) => candidate.id === sourceId)
    if (!source || !eligibleTelegramThread(source, threads, new Set(seen))) return false
  }
  return true
}

export function eligibleTelegramRoom(room: Room): boolean {
  return !room.archivedAt && room.state !== 'archived' && !room.orchestratorHostId && room.collaborationMode !== 'local_private'
}

export function telegramRemoteScope(settings: RemoteTelegramSettings, threads: readonly Thread[], rooms: readonly Room[]): TelegramRemoteScope {
  const remote = remoteSettings(settings)
  if (!remote.enabled) return { threads: [], rooms: [] }
  const all = remote.scope === 'all-eligible'
  return {
    threads: threads.filter((thread) => eligibleTelegramThread(thread, threads) && (all || remote.threadIds.includes(thread.id))),
    rooms: rooms.filter((room) => eligibleTelegramRoom(room) && (all || remote.roomIds.includes(room.id))),
  }
}

export function targetInTelegramScope(target: TelegramRemoteTarget, scope: TelegramRemoteScope): boolean {
  return target.kind === 'chief' || (target.kind === 'chat' ? scope.threads : scope.rooms).some((item) => item.id === target.id)
}

/** Privacy and membership changes revoke captured chief context as well as selected targets. */
export function telegramScopeRevision(scope: TelegramRemoteScope): string {
  return createHash('sha256').update(JSON.stringify([
    // Native admission durably reaccepts unchanged choices and advances its CAS revision.
    // Bind actual access choices here; settings/token scope has its own monotonic revision.
    scope.threads.map((thread) => [thread.id, parseThreadMemoryPolicy(thread.memoryPolicy)]).sort(),
    scope.rooms.map((room) => room.id).sort(),
  ])).digest('hex')
}
