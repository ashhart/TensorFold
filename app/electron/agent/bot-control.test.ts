// Run: node electron/agent/bot-control.test.ts
import assert from 'node:assert'
import {
  botControlBridgePolicy,
  dispatchBotControlMessage,
  isBotChatAllowed,
  personalBridgePolicy,
  parseBotControlIntent,
  resolveRoomRef,
  telegramUpdateToIngest,
} from './bot-control.ts'
import type { BotControlSession, BotControlSettings, Project, Room, RoomEventItem } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const project: Project = {
  id: 'proj-1',
  name: 'TensorFold Studio',
  cwd: '/tmp/vontra',
  memory: '',
  team: '',
  createdAt: 1,
  updatedAt: 1,
}

function room(id: string, state: Room['state'], updatedAt: number): Room {
  return {
    id,
    projectId: project.id,
    name: id === 'room-active-1234' ? 'Active Room' : 'Finished Room',
    seats: [{ id: 'seat-1', name: 'Planner', role: 'plan', providerId: 'openai', model: 'gpt-4o' }],
    problem: 'Build a thing',
    state,
    events: [],
    finalText: '',
    createdAt: updatedAt - 10,
    updatedAt,
  }
}

console.log('bot control tests:')

{
  assert.deepStrictEqual(parseBotControlIntent('/status@vontra_bot'), { type: 'status' })
  assert.deepStrictEqual(parseBotControlIntent('llmc:enter:room-active-1234'), { type: 'enter', roomRef: 'room-active-1234' })
  assert.deepStrictEqual(parseBotControlIntent('please close this room', 'room-active-1234'), { type: 'close', roomRef: 'this' })
  assert.deepStrictEqual(parseBotControlIntent('hello team'), { type: 'message', text: 'hello team' })
  ok('parses slash, callback, lifecycle, and free-text commands')
}

{
  assert.strictEqual(isBotChatAllowed('123, 456\n789', '456'), true)
  assert.strictEqual(isBotChatAllowed('', '456'), false)
  assert.strictEqual(isBotChatAllowed('123', '456'), false)
  ok('requires an explicit chat allowlist')
}

{
  const settings: BotControlSettings = {
    telegram: {
      enabled: true,
      botToken: 'token',
      allowedChatIds: '42',
      allowedUserIds: '7',
      polling: true,
      lastUpdateId: null,
      lastPollAt: 123,
      lastError: null,
    },
    whatsapp: {
      enabled: true,
      accessToken: 'token',
      phoneNumberId: 'phone',
      verifyToken: 'verify',
      appSecret: 'app-secret',
      allowedChatIds: '',
      lastEventAt: null,
      lastError: null,
    },
  }
  const telegram = botControlBridgePolicy(settings, 'telegram')
  assert.equal(telegram.optedIn, true)
  assert.equal(telegram.permissionStatus, 'granted')
  assert.equal(telegram.outboundApprovalRequired, true)
  assert.match(telegram.permissionDetail, /1 allowed Telegram chat/)

  const whatsapp = botControlBridgePolicy(settings, 'whatsapp')
  assert.equal(whatsapp.optedIn, false)
  assert.equal(whatsapp.permissionStatus, 'not_configured')
  assert.ok(whatsapp.missing.includes('allowed sender IDs'))

  const imessage = personalBridgePolicy({
    channel: 'imessage',
    label: 'iMessage',
    enabled: false,
    allowedTargets: '',
    allowedTargetLabel: 'allowed contacts',
    unsupportedReason: 'iMessage requires an explicit local bridge configuration before messages can enter or leave TensorFold Studio.',
  })
  assert.equal(imessage.supported, false)
  assert.equal(imessage.optedIn, false)
  assert.equal(imessage.permissionStatus, 'not_configured')
  assert.match(imessage.permissionDetail, /explicit local bridge/)
  ok('builds opt-in policy for Telegram, WhatsApp, and iMessage bridges')
}

{
  const rooms = [room('room-finished-9999', 'complete', 20), room('room-active-1234', 'running', 10)]
  assert.strictEqual(resolveRoomRef(rooms, 'latest', null)?.id, 'room-active-1234')
  assert.strictEqual(resolveRoomRef(rooms, 'active', 'room-finished-9999')?.id, 'room-finished-9999')
  assert.strictEqual(resolveRoomRef(rooms, 'room-active', null)?.id, 'room-active-1234')
  ok('resolves latest, active, and short room references')
}

{
  const rooms = [room('room-active-1234', 'running', 100)]
  const sessions: BotControlSession[] = []
  const appended: RoomEventItem[] = []
  const result1 = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', text: 'enter latest', senderName: 'Maya' }, {
    projects: [project],
    rooms,
    sessions,
    appendRoomMessage: (_room, item) => appended.push(item),
    closeRoom: () => 'closed',
    removeRoom: () => true,
    cleanupRooms: () => 0,
    now: 123,
  })
  assert.strictEqual(result1.roomId, 'room-active-1234')
  assert.strictEqual(sessions[0].activeRoomId, 'room-active-1234')

  const result2 = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', text: 'focus on token telemetry', senderName: 'Maya' }, {
    projects: [project],
    rooms,
    sessions,
    appendRoomMessage: (_room, item) => appended.push(item),
    closeRoom: () => 'closed',
    removeRoom: () => true,
    cleanupRooms: () => 0,
    now: 124,
  })
  assert.strictEqual(result2.action, 'message')
  assert.strictEqual(appended.length, 1)
  assert.strictEqual(appended[0].kind, 'user')
  assert.strictEqual(appended[0].seat, 'Telegram Maya')
  assert.match(appended[0].text, /token telemetry/)
  ok('enters a room and relays free text into the room')
}

{
  const rooms = [room('room-active-1234', 'running', 100)]
  const sessions: BotControlSession[] = [{ id: 'telegram:42', channel: 'telegram', chatId: '42', activeRoomId: 'room-active-1234', createdAt: 1, updatedAt: 1, roomHistory: [], messageHistory: [] }]
  let closed = false
  const close = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', text: 'close room active' }, {
    projects: [project],
    rooms,
    sessions,
    appendRoomMessage: () => {},
    closeRoom: () => { closed = true; return 'closed' },
    removeRoom: () => true,
    cleanupRooms: () => 0,
    now: 200,
  })
  assert.strictEqual(closed, false)
  assert.strictEqual(close.action, 'close_confirm_required')
  assert.strictEqual(close.roomId, 'room-active-1234')
  assert.ok(sessions[0].pendingConfirmation?.code)

  const confirm = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', text: `confirm ${sessions[0].pendingConfirmation?.code}` }, {
    projects: [project],
    rooms,
    sessions,
    appendRoomMessage: () => {},
    closeRoom: () => { closed = true; return 'closed' },
    removeRoom: () => true,
    cleanupRooms: () => 0,
    now: 210,
  })
  assert.strictEqual(closed, true)
  assert.strictEqual(confirm.action, 'close')
  assert.strictEqual(confirm.roomId, 'room-active-1234')
  assert.strictEqual(sessions[0].activeRoomId, null)
  ok('requires confirmation before closing active rooms')
}

{
  const rooms = [room('room-active-1234', 'running', 100)]
  const sessions: BotControlSession[] = [{
    id: 'telegram:42',
    channel: 'telegram',
    chatId: '42',
    activeRoomId: 'room-active-1234',
    createdAt: 1,
    updatedAt: 1,
    roomHistory: [],
    messageHistory: Array.from({ length: 12 }, (_, i) => ({ route: 'central' as const, text: `msg ${i}`, roomId: null, action: 'status', createdAt: 1_000 + i })),
  }]
  const result = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', text: 'status' }, {
    projects: [project],
    rooms,
    sessions,
    appendRoomMessage: () => {},
    closeRoom: () => 'closed',
    removeRoom: () => true,
    cleanupRooms: () => 0,
    now: 1_030,
  })
  assert.strictEqual(result.ok, false)
  assert.strictEqual(result.action, 'rate_limited')
  ok('rate limits noisy phone control chats')
}

{
  const sessions: BotControlSession[] = []
  const deps = { projects: [project], rooms: [], sessions, appendRoomMessage: () => {}, closeRoom: () => 'closed' as const, removeRoom: () => true, cleanupRooms: () => 0 }
  for (let i = 0; i < 12; i++) {
    const result = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', senderId: '7', eventId: `guess:${i}`, text: `confirm bad${String(i).padStart(2, '0')}` }, { ...deps, now: 2_000 + i })
    assert.equal(result.action, 'confirm_failed')
  }
  const limited = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', senderId: '7', eventId: 'guess:12', text: 'confirm bad12' }, { ...deps, now: 2_020 })
  assert.equal(limited.action, 'rate_limited', 'confirmation syntax without a live challenge cannot bypass the general limiter')
  ok('rate limits confirmation guesses when no live challenge exists')
}

{
  const update = telegramUpdateToIngest({
    update_id: 9,
    callback_query: {
      id: 'cb-1',
      data: 'llmc:close:room-active-1234',
      from: { id: 7, first_name: 'Maya' },
      message: { chat: { id: 42 } },
    },
  })
  assert.strictEqual(update.updateId, 9)
  assert.strictEqual(update.callbackQueryId, 'cb-1')
  assert.strictEqual(update.input?.chatId, '42')
  assert.strictEqual(update.input?.senderId, '7')
  assert.strictEqual(update.input?.eventId, 'telegram:9')
  assert.strictEqual(update.input?.text, 'close room room-active-1234')
  ok('extracts Telegram callback updates without exposing tokens')
}

{
  const rooms = [room('room-active-1234', 'running', 100)]
  const sessions: BotControlSession[] = []
  let closed = 0
  const deps = {
    projects: [project], rooms, sessions,
    appendRoomMessage: () => {},
    closeRoom: () => { closed++; return 'closed' as const },
    removeRoom: () => true,
    cleanupRooms: () => 0,
  }
  dispatchBotControlMessage({ channel: 'telegram', chatId: '-100', senderId: '7', eventId: 'telegram:20', text: 'close room active' }, { ...deps, now: 300 })
  const code = sessions[0].pendingConfirmation?.code
  assert.ok(code)
  for (let i = 0; i < 3; i++) {
    const result = dispatchBotControlMessage({ channel: 'telegram', chatId: '-100', senderId: i === 0 ? '8' : '7', eventId: `telegram:${21 + i}`, text: 'confirm wrong0' }, { ...deps, now: 301 + i })
    assert.equal(result.action, 'confirm_failed')
  }
  assert.equal(sessions[0].pendingConfirmation, null, 'three wrong sender/code attempts invalidate the pending action')
  const exhausted = dispatchBotControlMessage({ channel: 'telegram', chatId: '-100', senderId: '7', eventId: 'telegram:24', text: `confirm ${code}` }, { ...deps, now: 305 })
  assert.equal(exhausted.action, 'confirm_failed')
  assert.equal(closed, 0)
  ok('caps confirmation attempts and binds codes to the authenticated sender')
}

{
  const rooms = [room('room-active-1234', 'running', 100)]
  const sessions: BotControlSession[] = []
  let closed = 0
  const deps = {
    projects: [project], rooms, sessions,
    appendRoomMessage: () => {},
    closeRoom: () => { closed++; return 'closed' as const },
    removeRoom: () => true,
    cleanupRooms: () => 0,
  }
  dispatchBotControlMessage({ channel: 'telegram', chatId: '42', senderId: '7', eventId: 'telegram:30', text: 'close room active' }, { ...deps, now: 400 })
  const code = sessions[0].pendingConfirmation?.code
  assert.ok(code)
  const accepted = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', senderId: '7', eventId: 'telegram:31', text: `confirm ${code}` }, { ...deps, now: 401 })
  assert.equal(accepted.action, 'close')
  const replay = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', senderId: '7', eventId: 'telegram:31', text: `confirm ${code}` }, { ...deps, now: 402 })
  assert.equal(replay.action, 'replay_ignored')
  const newDeliverySameCode = dispatchBotControlMessage({ channel: 'telegram', chatId: '42', senderId: '7', eventId: 'telegram:32', text: `confirm ${code}` }, { ...deps, now: 403 })
  assert.equal(newDeliverySameCode.action, 'confirm_failed')
  assert.equal(closed, 1, 'a confirmation code can execute its destructive action only once')
  ok('rejects channel-delivery and confirmation-code replays')
}

console.log(`bot control tests passed (${passed})`)
