import { botControlBridgePolicy, connectorEventFromBot, dispatchBotControlMessage, isBotChatAllowed } from '../../agent/bot-control'
import type { BotControlIngestInput, BotControlResult, ConnectorEvent, Room, RoomEventItem, RoomStreamEvent } from '../../agent/types'
import { store } from '../state'
import { persistBotSessions, persistRooms } from './persistence'

export interface BotControlRuntime {
  abortRoom: (roomId: string) => void
  hasActiveRoom: (roomId: string) => boolean
  emitRoom: (event: RoomStreamEvent) => void
  rememberConnectorEvents: (events: ConnectorEvent[]) => void
}

export function botChannelAllowed(input: BotControlIngestInput): boolean {
  const policy = botControlBridgePolicy(store.settings.botControl, input.channel)
  if (!policy.optedIn) return false
  const senderId = String(input.senderId ?? '').trim()
  if (!senderId) return false
  if (input.channel === 'telegram') {
    return isBotChatAllowed(store.settings.botControl.telegram.allowedChatIds, input.chatId)
      && isBotChatAllowed(store.settings.botControl.telegram.allowedUserIds, senderId)
  }
  // Meta Cloud API conversations are 1:1; the signed messages[].from value is both
  // the sender principal and conversation id. Never authorize a bridge-supplied alias.
  return senderId === String(input.chatId)
    && isBotChatAllowed(store.settings.botControl.whatsapp.allowedChatIds, senderId)
}

function appendBotRoomMessage(runtime: BotControlRuntime, room: Room, item: RoomEventItem) {
  room.events.push(item)
  room.updatedAt = Date.now()
  runtime.emitRoom({ roomId: room.id, type: 'event', item })
  persistRooms()
}

function closeBotRoom(runtime: BotControlRuntime, room: Room): 'closed' | 'not_running' {
  const active = runtime.hasActiveRoom(room.id)
  if (!active && !['queued', 'running', 'waiting_user', 'waiting_external'].includes(room.state)) return 'not_running'
  runtime.abortRoom(room.id)
  room.state = 'stopped'
  room.updatedAt = Date.now()
  runtime.emitRoom({ roomId: room.id, type: 'status', state: room.state })
  persistRooms()
  return 'closed'
}

function removeBotRoom(runtime: BotControlRuntime, room: Room): boolean {
  const before = store.rooms.length
  runtime.abortRoom(room.id)
  store.rooms = store.rooms.filter((item) => item.id !== room.id)
  persistRooms()
  return store.rooms.length < before
}

function cleanupBotRooms(runtime: BotControlRuntime, targets: Room[]): number {
  const ids = new Set(targets.map((room) => room.id))
  if (!ids.size) return 0
  for (const id of ids) runtime.abortRoom(id)
  const before = store.rooms.length
  store.rooms = store.rooms.filter((room) => !ids.has(room.id))
  persistRooms()
  return before - store.rooms.length
}

export function ingestBotControl(input: BotControlIngestInput, runtime: BotControlRuntime): BotControlResult {
  const channelSettings = input.channel === 'telegram' ? store.settings.botControl.telegram : store.settings.botControl.whatsapp
  if (!channelSettings.enabled) throw new Error(`${input.channel} bot control is disabled.`)
  const policy = botControlBridgePolicy(store.settings.botControl, input.channel)
  if (!policy.optedIn) throw new Error(policy.permissionDetail)
  if (!botChannelAllowed(input)) throw new Error(`${input.channel} chat or authenticated sender is not allowlisted.`)
  const result = dispatchBotControlMessage(input, {
    projects: store.projects,
    rooms: store.rooms,
    sessions: store.botSessions,
    appendRoomMessage: (room, item) => appendBotRoomMessage(runtime, room, item),
    closeRoom: (room) => closeBotRoom(runtime, room),
    removeRoom: (room) => removeBotRoom(runtime, room),
    cleanupRooms: (targets) => cleanupBotRooms(runtime, targets),
  })
  persistBotSessions()
  if (result.action !== 'replay_ignored') runtime.rememberConnectorEvents([connectorEventFromBot(input, result)])
  return result
}
