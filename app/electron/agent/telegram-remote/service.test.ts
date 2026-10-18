import assert from 'node:assert/strict'
import type { Room, Thread } from '../types'
import type { TelegramRemoteTurnRequest, TelegramRemoteTurnResult } from '../../../shared/telegram-remote'
import { createTelegramRemoteService } from './service'
import { authenticateTelegramOwner, eligibleTelegramThread, telegramRemoteScope, type RemoteTelegramSettings } from './policy'

function thread(id: string, extra: Partial<Thread> = {}): Thread {
  return { id, title: `Chat ${id}`, kind: 'chat', status: 'idle', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1, cwd: '/synthetic', model: 'local-test', providerId: 'local', ...extra }
}
function room(id: string, extra: Partial<Room> = {}): Room {
  return { id, name: `Room ${id}`, state: 'running', projectId: 'synthetic', problem: 'Synthetic task', seats: [], events: [], finalText: '', createdAt: 1, updatedAt: 1, ...extra }
}
function update(id: number, text: string, principal = 7) {
  return { update_id: id, message: { chat: { id: principal, type: 'private' }, from: { id: principal, is_bot: false }, text } }
}
function fixture(dispatch?: (request: TelegramRemoteTurnRequest) => Promise<TelegramRemoteTurnResult>) {
  const settings: RemoteTelegramSettings = { enabled: true, polling: false, botToken: 'synthetic-token', allowedChatIds: '7 8', allowedUserIds: '7 8', lastUpdateId: null, lastPollAt: null, lastError: null,
    remote: { enabled: true, scope: 'selected', threadIds: ['one', 'two'], roomIds: ['allowed'], revision: 1, sessions: [] } }
  const threads = [thread('one'), thread('two'), thread('hidden'), thread('private', { incognito: true }), thread('phone', { phoneLocalOnly: true })]
  const rooms = [room('allowed'), room('hidden'), room('private', { collaborationMode: 'local_private' })]
  const calls: TelegramRemoteTurnRequest[] = []
  let persisted = 0
  let failSave = false
  let time = 1_000
  const deps = { settings: () => settings, threads: () => threads, rooms: () => rooms,
    persist: async () => { if (failSave) throw new Error('Synthetic protected save failure'); persisted++ },
    dispatchTurn: async (request: TelegramRemoteTurnRequest): Promise<TelegramRemoteTurnResult> => { calls.push(request); assert.ok(persisted > 0, 'checkpoint must be durable before dispatch'); return dispatch ? dispatch(request) : { status: 'completed', text: `Answer for ${request.target.kind === 'chief' ? 'chief' : request.target.id}` } },
    now: () => time, timeoutMs: 1_000 }
  return { settings, threads, rooms, calls, deps, service: createTelegramRemoteService(deps), failSave: () => { failSave = true }, advance: () => { time += 61_000 } }
}

{
  const f = fixture()
  assert.equal(authenticateTelegramOwner(update(1, 'hello'), f.settings)?.senderId, '7', 'manual polling works with background polling off')
  assert.equal(authenticateTelegramOwner({ ...update(2, 'hi'), message: { ...update(2, 'hi').message, chat: { id: 7, type: 'group' } } }, f.settings), null)
  assert.equal(authenticateTelegramOwner({ ...update(2, 'hi'), message: { ...update(2, 'hi').message, from: { id: 8, is_bot: false } } }, f.settings), null)
  assert.equal(authenticateTelegramOwner({ ...update(2, 'hi'), message: { ...update(2, 'hi').message, from: { id: 7, is_bot: true } } }, f.settings), null)
  assert.equal(authenticateTelegramOwner({ ...update(2, 'hi'), callback_query: { data: 'hi' }, message: undefined }, f.settings), null)
  assert.equal(authenticateTelegramOwner(update(2, 'hi', 9), f.settings), null)
  const source = thread('source', { incognito: true })
  assert.equal(eligibleTelegramThread(thread('fork', { parentId: 'source' }), [source]), false)
  assert.equal(eligibleTelegramThread(thread('missing', { messageEditSource: { threadId: 'deleted', itemId: 'item', editedAt: 1 } }), []), false)
  assert.equal(eligibleTelegramThread(thread('memory-private', { memoryPolicy: { useExisting: true, contribute: false } }), []), false)
  assert.equal(eligibleTelegramThread(thread('google-private', { googleLocalOnly: true }), []), false, 'Google mail history requires separate consent and is not ordinary Telegram scope')
  f.settings.remote!.scope = 'all-eligible'
  assert.deepEqual(telegramRemoteScope(f.settings, f.threads, f.rooms).threads.map((item) => item.id), ['one', 'two', 'hidden'])
  assert.deepEqual(telegramRemoteScope(f.settings, f.threads, f.rooms).rooms.map((item) => item.id), ['allowed', 'hidden'])
}

{
  const f = fixture()
  assert.match((await f.service.ingest(update(1, '/chats'))).reply, /Chat one/)
  assert.doesNotMatch((await f.service.ingest(update(2, '/rooms'))).reply, /Room hidden|Room private/)
  assert.match((await f.service.ingest(update(3, '/chat one'))).reply, /selected/)
  assert.equal((await f.service.ingest(update(4, 'continue the task'))).reply, 'Answer for one')
  assert.equal(f.calls[0].target.kind, 'chat')
  assert.deepEqual(f.calls[0].scope.threads.map((item) => item.id), ['one', 'two'])
  assert.equal(f.calls[0].scope.rooms.length, 1)
  assert.equal((await f.service.ingest(update(4, 'replay'))).skipped, true)
  const restarted = createTelegramRemoteService(f.deps)
  assert.equal((await restarted.ingest(update(5, 'after restart'))).reply, 'Answer for one')
  assert.match((await restarted.ingest(update(6, '/chat hidden'))).reply, /not allowed/)
  assert.equal((await restarted.ingest(update(7, 'still there'))).reply, 'Answer for one', 'invalid selection must keep the current target')
  assert.match((await restarted.ingest(update(8, '/back'))).reply, /Chief of staff/)
  assert.equal((await restarted.ingest(update(9, 'overview'))).reply, 'Answer for chief')
  assert.match((await restarted.ingest(update(10, '/approve'))).reply, /Approvals/)
  assert.match((await restarted.ingest(update(11, '/remove_room allowed'))).reply, /destructive/)
  assert.equal(f.calls.length, 4, 'commands never dispatch tools')
}

{
  const f = fixture()
  await f.service.ingest(update(1, '/chat one'))
  const other = await f.service.ingest(update(2, 'owner two overview', 8))
  assert.equal(other.reply, 'Answer for chief', 'owners do not inherit each other’s selections')
  f.threads.splice(f.threads.findIndex((item) => item.id === 'one'), 1)
  assert.match((await f.service.ingest(update(3, 'no fallback'))).reply, /no longer available/)
  assert.equal(f.calls.length, 1)
  assert.match((await f.service.ingest(update(4, '/chats'))).reply, /Chat two/, 'navigation works after selected chat deletion')
}

{
  const f = fixture()
  await f.service.ingest(update(1, '/room allowed'))
  assert.equal((await f.service.ingest(update(2, 'guide the room'))).reply, 'Answer for allowed')
  f.rooms[0].state = 'complete'
  assert.match((await f.service.ingest(update(3, 'start again'))).reply, /Start it in TensorFold first/)
  assert.equal(f.calls.length, 1, 'remote input never restarts inactive rooms')
}

{
  const f = fixture()
  f.failSave()
  await assert.rejects(() => f.service.ingest(update(1, 'must not run')), /protected save/)
  assert.equal(f.calls.length, 0)
}

{
  let complete!: (result: TelegramRemoteTurnResult) => void
  const f = fixture(() => new Promise((resolve) => { complete = resolve }))
  const inFlight = f.service.ingest(update(1, 'long answer'))
  await new Promise((resolve) => setTimeout(resolve, 5))
  f.settings.allowedUserIds = '8'
  complete({ status: 'completed', text: 'Sensitive captured answer' })
  const result = await inFlight
  assert.equal(result.reply, '', 'midflight sender revocation discards captured output')
  assert.equal(result.skipped, true)
  assert.equal(f.calls[0].isAuthorized(), false)
}

{
  let complete!: (result: TelegramRemoteTurnResult) => void
  const f = fixture(() => new Promise((resolve) => { complete = resolve }))
  const inFlight = f.service.ingest(update(1, 'overview'))
  await new Promise((resolve) => setTimeout(resolve, 5))
  f.threads[0].incognito = true
  complete({ status: 'completed', text: 'Captured private context' })
  const result = await inFlight
  assert.match(result.reply, /changed or became private/, 'privacy change revokes chief context with a harmless interruption notice')
  assert.ok(!result.reply.includes('Captured private context'))
  assert.equal(f.service.canDeliver(result), true)
}

{
  const f = fixture()
  const result = await f.service.ingest(update(1, 'hello'))
  assert.equal(f.service.canDeliver(result), true)
  f.settings.botToken = 'rotated-synthetic-token'
  assert.equal(f.service.canDeliver(result), false, 'old answers are never delivered using a newly configured bot')
  assert.match((await f.service.ingest(update(2, 'do not silently fall back'))).reply, /previous destination was cleared/)
  assert.equal(f.calls.length, 1, 'identity changes require visible reselection before another ordinary task')
}

{
  let finish!: (result: TelegramRemoteTurnResult) => void
  const f = fixture(() => new Promise((resolve) => { finish = resolve }))
  const pending = f.service.ingest(update(1, 'long-running reply'))
  await new Promise((resolve) => setTimeout(resolve, 5))
  await f.service.configureRemote({ enabled: false, scope: 'selected', threadIds: [], roomIds: [] })
  finish({ status: 'completed', text: 'Do not disclose this result' })
  assert.equal((await pending).reply, '')
  assert.equal(f.calls[0].signal.aborted, true)
}

{
  const f = fixture(async () => ({ status: 'completed', text: 'x'.repeat(5000) }))
  assert.match((await f.service.ingest(update(1, 'long output'))).reply, /Reply shortened/)
}

{
  const f = fixture()
  await f.service.ingest(update(1, '/chat one'))
  const status = await f.service.configureRemote({ enabled: true, scope: 'selected', threadIds: ['two'], roomIds: [] })
  assert.equal(status.revision, 2)
  assert.equal(f.settings.remote!.sessions.length, 0)
  assert.equal(f.settings.lastUpdateId, 1, 'clearing selection retains the durable replay barrier')
  assert.equal((await f.service.ingest(update(1, 'old delivery'))).skipped, true)
  assert.equal((await f.service.ingest(update(2, 'new scope'))).reply, 'Answer for chief')
  assert.deepEqual(f.calls[0].scope.threads.map((item) => item.id), ['two'])
  assert.ok(!JSON.stringify(status).includes('sessions'))
  await assert.rejects(() => f.service.configureRemote({ enabled: true, scope: 'selected', threadIds: ['phone'], roomIds: [] }), /private/)
}

{
  const f = fixture()
  for (let id = 1; id <= 12; id++) await f.service.ingest(update(id, '/help'))
  assert.match((await f.service.ingest(update(13, 'rate limited'))).reply, /Too many/)
  assert.equal(f.calls.length, 0)
  f.advance()
  assert.equal((await f.service.ingest(update(14, 'after limit'))).reply, 'Answer for chief')
}

{
  const f = fixture(async () => ({ status: 'queued', runId: 'synthetic-run' }))
  assert.match((await f.service.ingest(update(1, 'task'))).reply, /Queued.*synthetic-run.*approvals/s)
  f.settings.remote!.enabled = false
  assert.equal((await f.service.ingest(update(2, 'legacy'))).consumed, false, 'disabled remote mode preserves existing bot behavior')
}

console.log('Telegram remote routing, isolation, replay, durable configuration and revocation tests passed')
