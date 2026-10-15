import assert from 'node:assert/strict'
import { createThreadTitleGenerator } from './thread-title'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

const guard = new WorkspaceMutationGuard()
let calls = 0
let persistCalls = 0
let response: Promise<string> = Promise.resolve('A Short Title')
let currentThread = true
let reenter = false
let emits = 0
const title = createThreadTitleGenerator({
  completeTitle: async () => { calls++; return response },
  reserveOperation: () => guard.reserveStart({ workspaceKeys: ['title:background'] }),
  isCurrentThread: () => currentThread,
  persistThreads: () => { persistCalls++ },
  emitToRenderer: () => { emits++; if (reenter) void title(thread, provider, 'synthetic', 'request') },
})
const thread = { id: 'synthetic', title: 'New task' } as any
const provider = { baseUrl: 'https://model.example.test', apiKey: '', transport: 'chat-completions' } as any
const lease = await guard.acquireGlobal('App update installation')
await title(thread, provider, 'synthetic', 'request')
assert.equal(calls, 0, 'closed update admission refuses a background title request')
assert.equal(persistCalls, 0, 'closed admission cannot mutate the thread either')
await title(thread, { ...provider, transport: 'codex-cli' }, 'synthetic', 'local title words')
assert.equal(persistCalls, 0, 'local CLI title updates obey the same admission barrier')
lease.release()

let finish!: (title: string) => void
response = new Promise((resolve) => { finish = resolve })
const pending = title(thread, provider, 'synthetic', 'request')
assert.equal(calls, 1)
let acquired = false
const updating = guard.acquireGlobal('App update installation').then((value) => { acquired = true; return value })
await Promise.resolve()
assert.equal(acquired, false, 'the complete background request holds update admission until its response and persistence finish')
finish('A Short Title')
await pending
const updateLease = await updating
assert.equal(thread.title, 'A Short Title')
assert.equal(persistCalls, 1)
updateLease.release()

response = Promise.reject(new Error('synthetic title rejection'))
thread.title = 'New task'
await title(thread, provider, 'synthetic', 'request')
assert.equal(calls, 2, 'error cleanup covers an actual admitted completion')
const afterFailure = await guard.acquireGlobal('App update installation')
afterFailure.release()

for (const action of ['rename', 'remove'] as const) {
  thread.title = 'New task'
  response = new Promise((resolve) => { finish = resolve })
  const pendingOwnership = title(thread, provider, 'synthetic', 'request')
  if (action === 'rename') thread.title = 'My Chosen Name'
  else currentThread = false
  finish('Late Generated Title')
  await pendingOwnership
  assert.equal(persistCalls, 1, 'late title cannot persist a renamed or removed thread')
  assert.equal(emits, 1, 'late title cannot publish stale ownership')
  if (action === 'rename') assert.equal(thread.title, 'My Chosen Name')
  currentThread = true
}
thread.title = 'New task'
response = Promise.resolve('A Final Title')
reenter = true
await title(thread, provider, 'synthetic', 'request')
assert.equal(calls, 5, 'a reentrant callback cannot regenerate an already titled conversation')
assert.equal(persistCalls, 2)
console.log('background cloud/built-in/CLI title admission, full response lifetime and failure cleanup passed')
