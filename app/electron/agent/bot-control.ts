import { randomUUID } from 'node:crypto'
import type {
  BotControlChannel,
  BotControlIngestInput,
  BotControlPendingConfirmation,
  BotControlResult,
  BotControlSession,
  BotControlSettings,
  ConnectorPermissionStatus,
  ConnectorStatus,
  ConnectorEvent,
  Project,
  Room,
  RoomEventItem,
  RoomState,
} from './types'

export type BotControlIntent =
  | { type: 'status' }
  | { type: 'help' }
  | { type: 'enter'; roomRef: string }
  | { type: 'leave' }
  | { type: 'close'; roomRef: string }
  | { type: 'remove'; roomRef: string }
  | { type: 'cleanup' }
  | { type: 'message'; text: string }

export interface BotControlDispatchDeps {
  projects: Project[]
  rooms: Room[]
  sessions: BotControlSession[]
  appendRoomMessage: (room: Room, item: RoomEventItem) => void
  closeRoom: (room: Room) => 'closed' | 'not_running'
  removeRoom: (room: Room) => boolean
  cleanupRooms: (rooms: Room[]) => number
  now?: number
}

export interface TelegramUpdateExtract {
  updateId: number | null
  callbackQueryId: string | null
  input: BotControlIngestInput | null
}

export type PersonalBridgeChannel = BotControlChannel | 'imessage'

export interface PersonalBridgePolicyInput {
  channel: PersonalBridgeChannel
  label: string
  enabled: boolean
  allowedTargets: string
  allowedTargetLabel: string
  requiredFields?: Array<{ label: string; value: string | number | boolean | null | undefined }>
  lastActivityAt?: number | null
  lastError?: string | null
  unsupportedReason?: string | null
}

export interface PersonalBridgePolicy {
  channel: PersonalBridgeChannel
  label: string
  supported: boolean
  enabled: boolean
  optedIn: boolean
  allowedCount: number
  missing: string[]
  status: ConnectorStatus
  permissionStatus: ConnectorPermissionStatus
  permissionDetail: string
  lastActivityAt: number | null
  lastError: string | null
  outboundApprovalRequired: true
}

const STATUS_PHRASES = new Set([
  'status',
  'rooms',
  'all rooms',
  'active rooms',
  'current rooms',
  'live rooms',
  'show rooms',
  'show active rooms',
  'show me active rooms',
  'room update',
  'what is running',
  "what's running",
  'whats running',
])

const LEAVE_PHRASES = new Set([
  'leave',
  'leave room',
  'leave the room',
  'exit',
  'exit room',
  'exit the room',
  'step out',
  'step out of room',
  'step out of the room',
  'back',
  'done',
  'central',
  'concierge',
])

const ACTIVE_ROOM_STATES = new Set<RoomState>(['queued', 'running', 'waiting_user', 'waiting_external'])
const TERMINAL_ROOM_STATES = new Set<RoomState>(['complete', 'stopped', 'failed', 'archived', 'error'])
const CONFIRMATION_WINDOW_MS = 5 * 60_000
const CONFIRMATION_MAX_ATTEMPTS = 3
const RATE_LIMIT_WINDOW_MS = 60_000
const RATE_LIMIT_MAX_MESSAGES = 12

export function parseAllowedBotChatIds(raw: string): Set<string> {
  return new Set(String(raw || '').split(/[\s,;]+/).map((item) => item.trim()).filter(Boolean))
}

export function isBotChatAllowed(raw: string, chatId: string): boolean {
  const allowed = parseAllowedBotChatIds(raw)
  return allowed.size > 0 && allowed.has(String(chatId))
}

export function personalBridgePolicy(input: PersonalBridgePolicyInput): PersonalBridgePolicy {
  const allowedCount = parseAllowedBotChatIds(input.allowedTargets).size
  const supported = !input.unsupportedReason
  const missing: string[] = []
  if (!input.enabled) missing.push('enabled bridge')
  for (const field of input.requiredFields ?? []) {
    const value = typeof field.value === 'string' ? field.value.trim() : field.value
    if (!value) missing.push(field.label)
  }
  if (allowedCount === 0) missing.push(input.allowedTargetLabel)
  const optedIn = supported && missing.length === 0
  const status: ConnectorStatus = !optedIn ? 'pending' : input.lastError ? 'error' : 'connected'
  const permissionStatus: ConnectorPermissionStatus = !optedIn ? 'not_configured' : input.lastError ? 'degraded' : 'granted'
  const targetWord = input.channel === 'telegram' ? 'chat' : input.channel === 'whatsapp' ? 'sender' : 'contact'
  const detail = !supported
    ? input.unsupportedReason!
    : missing.length
      ? `${input.label} requires explicit opt-in: add ${joinHuman(missing)}.`
      : input.lastError
        ? `${input.label} bridge is allowlisted for ${allowedCount} ${targetWord}${allowedCount === 1 ? '' : 's'}, but reports: ${input.lastError}`
        : `${input.label} bridge is opted in for ${allowedCount} allowed ${input.label} ${targetWord}${allowedCount === 1 ? '' : 's'}. Outbound replies require approval.`
  return {
    channel: input.channel,
    label: input.label,
    supported,
    enabled: input.enabled,
    optedIn,
    allowedCount,
    missing,
    status,
    permissionStatus,
    permissionDetail: detail,
    lastActivityAt: input.lastActivityAt ?? null,
    lastError: input.lastError ?? null,
    outboundApprovalRequired: true,
  }
}

export function botControlBridgePolicy(settings: BotControlSettings, channel: BotControlChannel): PersonalBridgePolicy {
  if (channel === 'telegram') {
    return personalBridgePolicy({
      channel,
      label: 'Telegram',
      enabled: settings.telegram.enabled,
      allowedTargets: settings.telegram.allowedChatIds,
      allowedTargetLabel: 'allowed Telegram chat IDs',
      requiredFields: [
        { label: 'bot token', value: settings.telegram.botToken },
        { label: 'allowed Telegram user IDs', value: parseAllowedBotChatIds(settings.telegram.allowedUserIds).size ? settings.telegram.allowedUserIds : '' },
      ],
      lastActivityAt: settings.telegram.lastPollAt,
      lastError: settings.telegram.lastError,
    })
  }
  return personalBridgePolicy({
    channel,
    label: 'WhatsApp',
    enabled: settings.whatsapp.enabled,
    allowedTargets: settings.whatsapp.allowedChatIds,
    allowedTargetLabel: 'allowed sender IDs',
    requiredFields: [
      { label: 'access token', value: settings.whatsapp.accessToken },
      { label: 'phone number ID', value: settings.whatsapp.phoneNumberId },
      { label: 'verify token', value: settings.whatsapp.verifyToken },
      { label: 'Meta app secret', value: settings.whatsapp.appSecret },
    ],
    lastActivityAt: settings.whatsapp.lastEventAt,
    lastError: settings.whatsapp.lastError,
  })
}

export function normalizeBotControlText(text: string): string {
  let out = String(text || '').trim()
  if (out.startsWith('/')) out = out.replace(/^\/([a-z_]+)(?:@[A-Za-z0-9_]+)?\b/i, (_, cmd) => `/${String(cmd).toLowerCase()}`)
  if (out.startsWith('llmc:')) {
    const [, action, ...rest] = out.split(':')
    const room = rest.join(':')
    if (action === 'enter') return `enter room ${room}`
    if (action === 'close') return `close room ${room}`
    if (action === 'remove' || action === 'delete') return `remove room ${room}`
    if (action === 'cleanup') return 'clean up old rooms'
    if (action === 'status') return 'status'
  }
  return out
}

export function parseBotControlIntent(text: string, activeRoomId: string | null = null): BotControlIntent {
  const message = normalizeBotControlText(text).replace(/^\s*(?:please|pls|can\s+you|could\s+you|can\s+u)\s+/i, '').trim()
  const lower = message.toLowerCase()
  if (!message) return { type: 'help' }
  if (message === '/status' || message === '/rooms') return { type: 'status' }
  if (message.startsWith('/enter_room')) return { type: 'enter', roomRef: message.replace(/^\/enter_room\s*/i, '').trim() || 'latest' }
  if (message.startsWith('/close_room')) return { type: 'close', roomRef: message.replace(/^\/close_room\s*/i, '').trim() || activeRoomId || '' }
  if (message.startsWith('/remove_room')) return { type: 'remove', roomRef: message.replace(/^\/remove_room\s*/i, '').trim() || activeRoomId || '' }
  if (message === '/cleanup') return { type: 'cleanup' }
  if (lower === 'help' || lower === '/help') return { type: 'help' }
  if (STATUS_PHRASES.has(lower) || /(?:what|which)\s+rooms?\b/.test(lower) || /rooms?\s+(?:are\s+)?open\b/.test(lower)) return { type: 'status' }
  if (LEAVE_PHRASES.has(lower)) return { type: 'leave' }
  if (/^(clean\s+up|cleanup|clear|delete|remove)\s+(the\s+)?(old|finished|archived)\s+rooms?\b/.test(lower)) return { type: 'cleanup' }
  if (/^(clean\s+up|cleanup)\s+rooms?\b/.test(lower)) return { type: 'cleanup' }
  if (/^(clear|delete|remove)\s+(all\s+)?rooms?$/.test(lower)) return { type: 'cleanup' }

  const enter = message.match(/^(?:step\s+into|join|enter|switch\s+to|use)\s+(?:the\s+)?(?:(?:rooms?|huddles?)\s*)?(.*)$/i)
  if (enter) return { type: 'enter', roomRef: normalizeRoomRef(enter[1]) || 'latest' }
  const direct = message.match(/^(?:room:|#)([A-Za-z0-9_.:-]+)/i)
  if (direct) return { type: 'enter', roomRef: direct[1] }

  const close = message.match(/^(close|stop|end|cancel)\s+(?:the\s+)?(?:(?:rooms?|huddles?)\s*)?(.*)$/i)
  if (close) return { type: 'close', roomRef: normalizeRoomRef(close[2]) || activeRoomId || '' }
  const remove = message.match(/^(remove|delete)\s+(?:the\s+)?(?:(?:rooms?|huddles?)\s*)?(.*)$/i)
  if (remove) return { type: 'remove', roomRef: normalizeRoomRef(remove[2]) || activeRoomId || '' }

  return { type: 'message', text: message }
}

export function dispatchBotControlMessage(input: BotControlIngestInput, deps: BotControlDispatchDeps): BotControlResult {
  const now = deps.now ?? Date.now()
  const session = getOrCreateSession(deps.sessions, input.channel, input.chatId, now)
  const eventId = String(input.eventId ?? '').trim()
  if (eventId && session.processedEventIds?.includes(eventId)) {
    session.updatedAt = now
    return {
      ok: false,
      channel: input.channel,
      chatId: input.chatId,
      reply: 'This channel delivery was already processed.',
      action: 'replay_ignored',
      roomId: null,
    }
  }
  const confirmation = confirmationCode(input.text)
  // Only an actually pending challenge may bypass the chat-level limiter, and that path has
  // its own strict three-attempt budget. Random `confirm <guess>` traffic with no live challenge
  // is ordinary input and cannot evade the general rate limiter.
  const liveConfirmationAttempt = !!(confirmation && session.pendingConfirmation && session.pendingConfirmation.expiresAt >= now)
  if (!liveConfirmationAttempt && isRateLimited(session, now)) {
    return recordBotResult(session, input, {
      ok: false,
      channel: input.channel,
      chatId: input.chatId,
      reply: 'Too many phone-control messages. Wait a minute, then try again.',
      action: 'rate_limited',
      roomId: null,
    }, now)
  }
  let intent = parseBotControlIntent(input.text, session.activeRoomId)
  let confirmed = false
  if (confirmation) {
    const pending = session.pendingConfirmation
    if (!pending || pending.expiresAt < now) {
      session.pendingConfirmation = null
      return recordBotResult(session, input, {
        ok: false,
        channel: input.channel,
        chatId: input.chatId,
        reply: 'No destructive action is waiting for confirmation.',
        action: 'confirm_failed',
        roomId: null,
      }, now)
    }
    const principalId = String(input.senderId ?? input.chatId)
    if (pending.principalId !== principalId || pending.code.toLowerCase() !== confirmation.toLowerCase()) {
      pending.attemptsRemaining = Math.max(0, (pending.attemptsRemaining ?? CONFIRMATION_MAX_ATTEMPTS) - 1)
      if (pending.attemptsRemaining === 0) session.pendingConfirmation = null
      return recordBotResult(session, input, {
        ok: false,
        channel: input.channel,
        chatId: input.chatId,
        reply: pending.attemptsRemaining === 0
          ? 'Confirmation failed too many times. Request the action again for a new code.'
          : `Confirmation did not match this sender or code. ${pending.attemptsRemaining} attempt${pending.attemptsRemaining === 1 ? '' : 's'} remain.`,
        action: 'confirm_failed',
        roomId: pending.roomRef ?? null,
      }, now)
    }
    session.pendingConfirmation = null
    intent = pending.action === 'cleanup'
      ? { type: 'cleanup' }
      : { type: pending.action, roomRef: pending.roomRef || session.activeRoomId || '' }
    confirmed = true
  }
  const channelName = input.channel === 'telegram' ? 'Telegram' : 'WhatsApp'
  let reply = ''
  let action: string = intent.type
  let roomId: string | null = null

  if (intent.type === 'status') {
    reply = formatBotRoomStatus(deps.rooms, deps.projects, session.activeRoomId)
  } else if (intent.type === 'help') {
    reply = helpText()
  } else if (intent.type === 'enter') {
    const room = resolveRoomRef(deps.rooms, intent.roomRef, session.activeRoomId)
    if (!room) {
      reply = `I could not find room "${intent.roomRef}". Try "rooms" or "enter latest".`
    } else {
      session.activeRoomId = room.id
      rememberRoom(session, room.id, 'enter', now)
      roomId = room.id
      reply = `Active room set to ${room.name} (${shortId(room.id)}). Send any normal message and I will relay it to the team.`
    }
  } else if (intent.type === 'leave') {
    const previous = session.activeRoomId
    session.activeRoomId = null
    rememberRoom(session, previous, 'leave', now)
    roomId = previous
    reply = 'Stepped out. You are talking to TensorFold central again.'
  } else if (intent.type === 'close') {
    const ref = intent.roomRef || session.activeRoomId || ''
    const room = resolveRoomRef(deps.rooms, ref, session.activeRoomId)
    if (!room) reply = ref ? `Could not close ${ref}: room not found.` : 'Which room should I close? Say "close room <id>".'
    else {
      if (!confirmed) {
        const pending = setPendingConfirmation(session, 'close', room.id, String(input.senderId ?? input.chatId), now)
        reply = `Reply "confirm ${pending.code}" within 5 minutes to close ${room.name} (${shortId(room.id)}).`
        action = 'close_confirm_required'
        roomId = room.id
      } else {
        const closed = deps.closeRoom(room)
        roomId = room.id
        rememberRoom(session, room.id, 'close', now)
        if (session.activeRoomId === room.id) session.activeRoomId = null
        reply = closed === 'closed'
          ? `Close requested for ${room.name} (${shortId(room.id)}).`
          : `${room.name} (${shortId(room.id)}) is not running. Use "remove room ${shortId(room.id)}" if you want it gone.`
      }
    }
  } else if (intent.type === 'remove') {
    const ref = intent.roomRef || session.activeRoomId || ''
    const room = resolveRoomRef(deps.rooms, ref, session.activeRoomId)
    if (!room) reply = ref ? `Could not remove ${ref}: room not found.` : 'Which room should I remove? Say "remove room <id>".'
    else if (ACTIVE_ROOM_STATES.has(room.state)) reply = `${room.name} (${shortId(room.id)}) is ${room.state}. Say "close room ${shortId(room.id)}" first, then remove it.`
    else {
      if (!confirmed) {
        const pending = setPendingConfirmation(session, 'remove', room.id, String(input.senderId ?? input.chatId), now)
        reply = `Reply "confirm ${pending.code}" within 5 minutes to remove ${room.name} (${shortId(room.id)}).`
        action = 'remove_confirm_required'
        roomId = room.id
      } else {
        const removed = deps.removeRoom(room)
        roomId = room.id
        rememberRoom(session, room.id, 'remove', now)
        if (session.activeRoomId === room.id) session.activeRoomId = null
        reply = removed ? `Removed room ${room.name} (${shortId(room.id)}).` : `Could not remove ${room.name} (${shortId(room.id)}).`
      }
    }
  } else if (intent.type === 'cleanup') {
    const targets = deps.rooms.filter((room) => TERMINAL_ROOM_STATES.has(room.state))
    if (!confirmed && targets.length) {
      const pending = setPendingConfirmation(session, 'cleanup', null, String(input.senderId ?? input.chatId), now)
      reply = `Reply "confirm ${pending.code}" within 5 minutes to remove ${targets.length} finished room${targets.length === 1 ? '' : 's'}.`
      action = 'cleanup_confirm_required'
    } else {
      const removed = deps.cleanupRooms(targets)
      if (session.activeRoomId && !deps.rooms.some((room) => room.id === session.activeRoomId)) session.activeRoomId = null
      reply = removed ? `Removed ${removed} finished room${removed === 1 ? '' : 's'}.` : 'Nothing to remove - no finished rooms on file.'
    }
  } else {
    const room = session.activeRoomId ? deps.rooms.find((item) => item.id === session.activeRoomId) : null
    if (!room) {
      reply = 'No active room selected. Say "rooms", then "enter room <id>" before sending instructions.'
    } else if (TERMINAL_ROOM_STATES.has(room.state) || room.state === 'created' || room.state === 'idle') {
      session.activeRoomId = null
      reply = `Room ${room.name} (${shortId(room.id)}) is ${room.state}, so I stepped you out. Ask for "rooms" to pick another.`
    } else {
      const event: RoomEventItem = {
        id: randomUUID(),
        tick: nextRoomTick(room),
        kind: 'user',
        seat: `${channelName}${input.senderName ? ` ${input.senderName}` : ''}`,
        target: 'all',
        text: intent.text,
        createdAt: now,
      }
      deps.appendRoomMessage(room, event)
      roomId = room.id
      reply = `Sent to ${room.name} (${shortId(room.id)}) - the room will pick it up on the next turn.`
    }
  }

  return recordBotResult(session, input, { ok: true, channel: input.channel, chatId: input.chatId, reply, action, roomId }, now)
}

function recordBotResult(session: BotControlSession, input: BotControlIngestInput, result: BotControlResult, now: number): BotControlResult {
  session.updatedAt = now
  const eventId = String(input.eventId ?? '').trim()
  if (eventId) session.processedEventIds = [eventId, ...(session.processedEventIds ?? []).filter((id) => id !== eventId)].slice(0, 120)
  const route: 'central' | 'room' = result.roomId ? 'room' : 'central'
  session.messageHistory = [
    { route, text: input.text, roomId: result.roomId, action: result.action, createdAt: now },
    ...session.messageHistory,
  ].slice(0, 80)
  return result
}

function isRateLimited(session: BotControlSession, now: number): boolean {
  const recent = session.messageHistory.filter((item) => now - item.createdAt <= RATE_LIMIT_WINDOW_MS)
  return recent.length >= RATE_LIMIT_MAX_MESSAGES
}

function confirmationCode(text: string): string | null {
  const match = normalizeBotControlText(text).match(/^confirm\s+([a-z0-9-]{4,16})$/i)
  return match?.[1] ?? null
}

function setPendingConfirmation(session: BotControlSession, action: BotControlPendingConfirmation['action'], roomRef: string | null, principalId: string, now: number): BotControlPendingConfirmation {
  const pending: BotControlPendingConfirmation = {
    code: randomUUID().replace(/-/g, '').slice(0, 6),
    action,
    roomRef,
    principalId,
    attemptsRemaining: CONFIRMATION_MAX_ATTEMPTS,
    createdAt: now,
    expiresAt: now + CONFIRMATION_WINDOW_MS,
  }
  session.pendingConfirmation = pending
  return pending
}

export function connectorEventFromBot(input: BotControlIngestInput, result: BotControlResult, now = Date.now()): ConnectorEvent {
  return {
    id: randomUUID(),
    connectorId: input.channel,
    pluginName: input.channel === 'telegram' ? 'Telegram bot control' : 'WhatsApp bot control',
    action: result.action,
    direction: 'inbound',
    classification: 'communication',
    priority: result.roomId ? 'high' : 'normal',
    trustLevel: 'notify',
    requiresApproval: false,
    approvalId: null,
    title: `${input.channel === 'telegram' ? 'Telegram' : 'WhatsApp'} room control`,
    summary: result.reply.slice(0, 220),
    actor: `${input.channel}:${input.chatId}`,
    threadKey: result.roomId ? `room:${result.roomId}` : `bot:${input.channel}:${input.chatId}`,
    payloadKeys: Object.keys(input.payload ?? {}),
    createdAt: now,
  }
}

export function telegramUpdateToIngest(update: any): TelegramUpdateExtract {
  const updateId = typeof update?.update_id === 'number' ? update.update_id : null
  const callback = update?.callback_query
  if (callback) {
    const data = typeof callback.data === 'string' ? callback.data : ''
    const chatId = callback.message?.chat?.id ?? callback.from?.id
    const senderId = callback.from?.id
    if (!data || chatId == null || senderId == null) return { updateId, callbackQueryId: typeof callback.id === 'string' ? callback.id : null, input: null }
    return {
      updateId,
      callbackQueryId: typeof callback.id === 'string' ? callback.id : null,
      input: {
        channel: 'telegram',
        chatId: String(chatId),
        senderId: String(senderId),
        eventId: updateId == null ? undefined : `telegram:${updateId}`,
        text: normalizeBotControlText(data),
        senderName: displayName(callback.from),
        payload: { update_id: updateId, callback_query: true },
      },
    }
  }

  const message = update?.message ?? update?.edited_message
  const text = typeof message?.text === 'string' ? message.text : ''
  const chatId = message?.chat?.id
  const senderId = message?.from?.id
  if (!text || chatId == null || senderId == null) return { updateId, callbackQueryId: null, input: null }
  return {
    updateId,
    callbackQueryId: null,
    input: {
      channel: 'telegram',
      chatId: String(chatId),
      senderId: String(senderId),
      eventId: updateId == null ? undefined : `telegram:${updateId}`,
      text,
      senderName: displayName(message.from),
      payload: { update_id: updateId },
    },
  }
}

export function formatBotRoomStatus(rooms: Room[], projects: Project[], activeRoomId: string | null): string {
  const sorted = [...rooms].sort((a, b) => {
    const liveDelta = Number(ACTIVE_ROOM_STATES.has(b.state)) - Number(ACTIVE_ROOM_STATES.has(a.state))
    return liveDelta || b.updatedAt - a.updatedAt
  }).slice(0, 8)
  if (!sorted.length) return 'No rooms yet.'
  const lines = sorted.map((room) => {
    const project = projects.find((item) => item.id === room.projectId)
    const active = room.id === activeRoomId ? ' *active*' : ''
    const seats = room.seats.length ? `${room.seats.length} seats` : 'no seats'
    return `- ${room.name} (${shortId(room.id)}) - ${room.state}, ${seats}${project ? `, ${project.name}` : ''}${active}`
  })
  return [
    'TensorFold rooms:',
    ...lines,
    '',
    'Commands: enter room <id>, leave, close room <id>, remove room <id>. Send a normal message while inside a room to guide it.',
  ].join('\n')
}

export function resolveRoomRef(rooms: Room[], ref: string, activeRoomId: string | null): Room | null {
  const clean = normalizeRoomRef(ref)
  if (!clean && activeRoomId) return rooms.find((room) => room.id === activeRoomId) ?? null
  const lower = clean.toLowerCase()
  if (['latest', 'newest', 'last', 'recent', 'this', 'that', 'it', 'active', 'current'].includes(lower)) {
    if (activeRoomId) {
      const active = rooms.find((room) => room.id === activeRoomId)
      if (active) return active
    }
    return [...rooms].sort((a, b) => {
      const liveDelta = Number(ACTIVE_ROOM_STATES.has(b.state)) - Number(ACTIVE_ROOM_STATES.has(a.state))
      return liveDelta || b.updatedAt - a.updatedAt
    })[0] ?? null
  }
  return rooms.find((room) => room.id === clean || room.id.startsWith(clean) || room.name.toLowerCase() === lower || room.name.toLowerCase().includes(lower)) ?? null
}

function getOrCreateSession(sessions: BotControlSession[], channel: BotControlChannel, chatId: string, now: number): BotControlSession {
  const id = `${channel}:${chatId}`
  let session = sessions.find((item) => item.id === id)
  if (session) return session
  session = { id, channel, chatId, activeRoomId: null, createdAt: now, updatedAt: now, roomHistory: [], messageHistory: [] }
  sessions.unshift(session)
  return session
}

function rememberRoom(session: BotControlSession, roomId: string | null, action: string, now: number): void {
  session.roomHistory = [{ roomId, action, createdAt: now }, ...session.roomHistory].slice(0, 60)
}

function normalizeRoomRef(ref: string): string {
  const clean = String(ref || '').trim().replace(/^(?:room:|#)/i, '').replace(/^room\s+/i, '').replace(/[.,]+$/g, '')
  const lower = clean.toLowerCase()
  const activePhrases = new Map([
    ['current room', 'current'],
    ['this room', 'this'],
    ['that room', 'that'],
    ['active room', 'active'],
    ['latest room', 'latest'],
    ['newest room', 'newest'],
    ['last room', 'last'],
    ['recent room', 'recent'],
  ])
  if (activePhrases.has(lower)) return activePhrases.get(lower)!
  return clean || ''
}

function nextRoomTick(room: Room): number {
  return Math.max(0, ...room.events.map((event) => event.tick || 0)) + 1
}

function shortId(id: string): string {
  return id.slice(0, 8)
}

function helpText(): string {
  return [
    'TensorFold phone control:',
    '- rooms / status',
    '- enter room <id> or enter latest',
    '- send any normal message while inside a room',
    '- leave',
    '- close room <id>',
    '- remove room <id>',
  ].join('\n')
}

function joinHuman(items: string[]): string {
  if (items.length <= 1) return items[0] ?? ''
  if (items.length === 2) return `${items[0]} and ${items[1]}`
  return `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`
}

function displayName(from: any): string | undefined {
  const parts = [from?.first_name, from?.last_name].filter((item) => typeof item === 'string' && item.trim())
  return parts.length ? parts.join(' ') : typeof from?.username === 'string' ? from.username : undefined
}
