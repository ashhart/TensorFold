import assert from 'node:assert/strict'
import type { Room, Settings, Thread } from '../../agent/types'
import type { TelegramRemoteTurnRequest } from '../../../shared/telegram-remote'
import { createTelegramRemoteDispatch } from './telegram-remote-dispatch'
import { telegramChiefMessages } from './telegram-remote-chief'
import { steerExistingRoom } from './room-steering-runtime'
import { defaultSettings } from '../../store'

const local = { id: 'fixture-local', name: 'Synthetic local model', kind: 'local' as const, baseUrl: 'http://127.0.0.1:9099/v1', apiKey: '', models: ['synthetic'] }
function thread(id: string, extra: Partial<Thread> = {}): Thread {
  return { id, title: `Chat ${id}`, kind: 'chat', status: 'idle', cwd: '/synthetic', providerId: local.id, providerAccountId: null, model: 'synthetic', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1, ...extra }
}
function fixture(mode: 'final' | 'queued' | 'error' = 'final') {
  const settings: Settings = { ...defaultSettings(), providers: [local], activeProviderId: local.id, activeModel: 'synthetic' }
  const threads = [thread('one'), thread('hidden'), thread('private', { incognito: true })]
  const rooms = [{ id: 'allowed', name: 'Synthetic active room', state: 'running', seats: [], events: [], updatedAt: 1 } as unknown as Room]
  const aborters = new Map<string, AbortController>()
  const parent = new AbortController()
  let authorized = true
  const calls: any[] = []
  let streamed: any
  const request: TelegramRemoteTurnRequest = { ownerKey: 'telegram:7:7', target: { kind: 'chat', id: 'one' }, text: 'Synthetic task', scope: { threads: [threads[0]], rooms }, signal: parent.signal, isAuthorized: () => authorized }
  const dispatch = createTelegramRemoteDispatch({ settings: () => settings, threads: () => threads, rooms: () => rooms, aborters, requireOperator() {}, waitMs: 15, pollMs: 1,
    startThreadRun: async (payload, admission) => {
      calls.push(payload); admission.assertAllowed()
      const controller = new AbortController(); aborters.set(payload.threadId, controller); admission.onController(controller)
      const selected = threads.find((item) => item.id === payload.threadId)!
      selected.status = 'running'
      if (mode !== 'queued') setTimeout(() => {
        selected.timeline.push({ id: 'answer', kind: mode === 'error' ? 'error' : 'assistant', text: 'Actual selected answer', trace: { traceId: 'synthetic', runId: 'owned-run', surface: 'main' } })
        selected.status = mode === 'error' ? 'error' : 'idle'; aborters.delete(selected.id)
      }, 1)
      return { runId: 'owned-run' }
    },
    steerRoom: (id, text) => { calls.push({ roomId: id, text }) },
    streamChat: async function* (params) { streamed = params; yield { type: 'content', text: 'Scoped chief answer' }; yield { type: 'finish', reason: 'stop' } },
  })
  return { settings, threads, rooms, aborters, parent, request, calls, dispatch, stream: () => streamed, revoke: () => { authorized = false } }
}

{
  const f = fixture()
  const original = { providerId: f.threads[0].providerId, model: f.threads[0].model, cwd: f.threads[0].cwd, engine: f.threads[0].engine, approvalMode: f.settings.approvalMode }
  assert.deepEqual(await f.dispatch(f.request), { status: 'completed', runId: 'owned-run', text: 'Actual selected answer' })
  assert.deepEqual(f.calls, [{ threadId: 'one', userText: 'Synthetic task' }], 'only canonical host start receives the chosen target and text')
  assert.deepEqual({ providerId: f.threads[0].providerId, model: f.threads[0].model, cwd: f.threads[0].cwd, engine: f.threads[0].engine, approvalMode: f.settings.approvalMode }, original)
  const busy = new AbortController(); f.aborters.set('one', busy)
  assert.equal((await f.dispatch(f.request)).status, 'busy')
  f.parent.abort()
  assert.equal(busy.signal.aborted, false, 'a remote request never cancels an existing unrelated run')
  assert.equal(f.calls.length, 1)
}
{
  const f = fixture('queued')
  const result = await f.dispatch(f.request)
  assert.equal(result.status, 'queued')
  assert.equal(result.runId, 'owned-run')
  const owned = f.aborters.get('one')!
  const replacement = new AbortController(); f.aborters.set('one', replacement)
  f.parent.abort()
  assert.equal(replacement.signal.aborted, false)
  assert.equal(owned.signal.aborted, false, 'replaced ownership is not guessed from thread identity')
  await new Promise((resolve) => setTimeout(resolve, 110))
}
{
  const f = fixture('queued')
  assert.equal((await f.dispatch(f.request)).status, 'queued')
  const owned = f.aborters.get('one')!
  f.revoke()
  await new Promise((resolve) => setTimeout(resolve, 110))
  assert.equal(owned.signal.aborted, true, 'queued remote-owned work retains a revocation watcher after Telegram returns status')
  f.aborters.delete('one')
}
{
  const f = fixture('error')
  assert.equal((await f.dispatch(f.request)).status, 'error')
  assert.equal((await f.dispatch({ ...f.request, target: { kind: 'chat', id: 'hidden' } })).status, 'error')
  assert.equal((await f.dispatch({ ...f.request, target: { kind: 'chat', id: 'private' } })).status, 'error')
  assert.equal(f.calls.length, 1)
}
{
  const f = fixture()
  f.threads[0].timeline = [
    { id: 'visible', kind: 'user', text: 'Primary request', attachments: [{ id: 'hidden-file', kind: 'document', name: 'hidden.txt', mime: 'text/plain', text: 'HIDDEN-ATTACHMENT' }] },
    { id: 'tool', kind: 'tool', output: 'HIDDEN-TOOL', args: { secret: 'HIDDEN-ARGS' } },
    { id: 'answer', kind: 'assistant', text: 'Primary answer' },
  ]
  f.threads[0].apiMessages = [{ role: 'system', content: 'HIDDEN-SYSTEM' }]
  f.threads[0].messageEditRevisions = [{ text: 'SUPERSEDED-PRIVATE' } as any]
  f.threads[0].notepad = 'HIDDEN-NOTEPAD'
  f.threads[1].timeline = [{ id: 'secret', kind: 'user', text: 'UNSELECTED-CHAT' }]
  f.threads.push(thread('google', { googleLocalOnly: true, timeline: [{ id: 'mail', kind: 'assistant', text: 'PRIVATE-GOOGLE-MAIL' }] }))
  f.request.scope = { threads: [...f.request.scope.threads, f.threads.at(-1)!], rooms: f.rooms }
  const chief = { ...f.request, target: { kind: 'chief' as const } }
  const context = JSON.stringify(telegramChiefMessages(chief, f.threads, f.rooms))
  assert.match(context, /Primary request|Primary answer/)
  assert.doesNotMatch(context, /HIDDEN-|SUPERSEDED|UNSELECTED|PRIVATE-GOOGLE/)
  assert.equal((await f.dispatch(chief)).text, 'Scoped chief answer')
  assert.deepEqual(f.stream().tools, [])
  assert.equal(f.stream().approvalMode, 'readonly')
  assert.equal(f.stream().threadId, undefined, 'chief does not open or resume a persistent model session')
  assert.equal(f.calls.length, 0)
  f.settings.providers = [{ ...local, kind: 'cloud', baseUrl: 'https://model.example.com/v1' }]
  assert.match((await f.dispatch(chief)).text ?? '', /No conversation data was sent to a cloud/)
}
{
  const f = fixture()
  assert.match((await f.dispatch({ ...f.request, target: { kind: 'chief' }, text: 'dispatch chat one Synthetic delegated task' })).text ?? '', /Chat one: Actual selected answer/)
  assert.deepEqual(f.calls, [{ threadId: 'one', userText: 'Synthetic delegated task' }])
  assert.equal((await f.dispatch({ ...f.request, target: { kind: 'chief' }, text: 'dispatch chat hidden outside scope' })).status, 'error')
  assert.equal(f.calls.length, 1)
  assert.equal((await f.dispatch({ ...f.request, target: { kind: 'room', id: 'allowed' } })).status, 'queued')
  f.rooms[0].state = 'complete'
  assert.equal((await f.dispatch({ ...f.request, target: { kind: 'room', id: 'allowed' } })).status, 'error')
  assert.equal(f.calls.length, 2)
}
{
  const f = fixture()
  let persisted = 0, emitted = 0
  const deps = { rooms: () => f.rooms, requireOperator() {}, createId: () => 'synthetic-steer', persistRooms: () => { persisted++ }, emitRoom: () => { emitted++ }, hasActiveRoom: () => true, now: () => 42 }
  const event = steerExistingRoom(deps, 'allowed', 'Synthetic room guidance')
  assert.equal(event.text, 'Synthetic room guidance')
  assert.equal(f.rooms[0].events.length, 1)
  assert.equal(persisted, 1); assert.equal(emitted, 1)
  assert.throws(() => steerExistingRoom({ ...deps, hasActiveRoom: () => false }, 'allowed', 'Must not start'), /Start the room/)
}

console.log('Telegram host dispatch preserves canonical chat policy, exact controller ownership, queued status, local scoped chief and canonical active room steering')
