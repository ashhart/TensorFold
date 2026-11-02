import assert from 'node:assert/strict'
import path from 'node:path'
import { load } from '../components/reading/reading-test-harness'

const existing = { id: 'existing', kind: 'chat', providerId: 'fixture/cloud', model: 'assistant-default',
  timeline: [{ kind: 'user', text: 'Existing conversation' }], apiMessages: [{ role: 'user', content: 'Existing conversation' }] }
const settings = { providers: [{ id: 'tensorfold' }, { id: 'fixture/cloud' }],
  assistantProviderId: 'fixture/cloud', assistantModel: 'assistant-default', activeProviderId: 'tensorfold', activeModel: 'fixture/local' }
let state: any
let calls: string[] = []
let refuse = false
let holdCreate = false
let holdConfigure = false
let resume: () => void = () => {}
let supplied: any
function reset() {
  state = { settings: structuredClone(settings), threads: [structuredClone(existing)], view: { name: 'thread', id: 'existing' } }
  calls = []; refuse = false; holdCreate = false; holdConfigure = false
  supplied = { id: 'created', kind: 'chat', providerId: 'fixture/cloud', model: 'assistant-default', timeline: [], apiMessages: [] }
}
const api = {
  async createThread(cwd: string, kind: string, projectId: unknown, commandId: string) {
    assert.equal(cwd, ''); assert.equal(kind, 'chat'); assert.equal(projectId, null); assert.ok(commandId)
    calls.push('create')
    if (holdCreate) await new Promise<void>(resolve => { resume = resolve })
    return supplied
  },
  async configureThread(id: string, providerId: string, model: string, accountId: string | null) {
    assert.deepEqual([id, providerId, model, accountId], ['created', 'tensorfold', 'fixture/local', null])
    assert.equal(state.view.id, 'existing', 'the chat cannot open with the saved cloud default while configuration waits')
    calls.push('configure')
    if (holdConfigure) await new Promise<void>(resolve => { resume = resolve })
    if (refuse) throw new Error('Synthetic model selection refused')
  },
  startRun() { throw new Error('Opening an explicit model chat must never send a request') },
}
const { openModelChat } = load(path.join(__dirname, 'model-chat.ts'), {
  './ipc': { api }, './store': { useStore: {
    getState: () => state,
    setState(update: any) { calls.push('open'); state = { ...state, ...update(state) } },
  } }, './provider-accounts': { resolveProviderModel: (_provider: unknown, _account: unknown, model: string) => model },
})
const selection = { providerId: 'tensorfold', model: 'fixture/local' }
const settle = async () => { for (let i = 0; i < 10; i++) await Promise.resolve() }

reset()
await openModelChat(selection)
assert.deepEqual(calls, ['create', 'configure', 'open'])
assert.equal(state.threads[0].providerId, 'tensorfold')
assert.equal(state.threads[0].model, 'fixture/local')
assert.equal(state.threads[0].timeline.length, 0)
assert.deepEqual(state.threads[1], existing)
assert.deepEqual(state.settings, settings, 'explicit selection preserves saved Assistant and global model defaults')

reset(); refuse = true
await assert.rejects(openModelChat(selection), /Synthetic model selection refused/)
assert.deepEqual(calls, ['create', 'configure'])
assert.equal(state.threads.length, 1)
assert.equal(state.view.id, 'existing')

reset(); holdCreate = true
let present = true
const creating = openModelChat(selection, () => present)
present = false; resume(); await creating
assert.deepEqual(calls, ['create'], 'leaving during creation cannot configure or reopen an abandoned model chat')

reset(); holdConfigure = true; present = true
const configuring = openModelChat(selection, () => present)
await settle(); present = false; resume(); await configuring
assert.deepEqual(calls, ['create', 'configure'])
assert.equal(state.view.id, 'existing', 'late configuration completion cannot steal navigation')

reset(); supplied.timeline.push({ kind: 'user', text: 'Unexpected content' })
await assert.rejects(openModelChat(selection), /new local chat changed/)
assert.deepEqual(calls, ['create'])
reset()
await openModelChat(selection, () => false)
assert.deepEqual(calls, [])
console.log('Explicit model chats preserve saved surface defaults, configure before opening, never send and fence late navigation.')
