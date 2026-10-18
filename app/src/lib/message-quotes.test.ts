import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import type { Thread } from './types'
const globals = globalThis as any, target = new EventTarget()
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = target.addEventListener.bind(target); globals.removeEventListener = target.removeEventListener.bind(target); globals.dispatchEvent = target.dispatchEvent.bind(target); globals.window = globals
await (await import('./browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('./ipc'), { useStore } = await import('./store')
const { requestMessageQuote, MESSAGE_QUOTE_EVENT } = await import('./message-quotes')
const settings = await api.getSettings(), old = api.prepareMessageQuote
const quote = { id: 'quote', sourceItemId: 'source', sourceRevision: createHash('sha256').update('Original').digest('hex'), sourceText: 'Original', text: 'Original' }
const a = { id: 'a', timeline: [{ id: 'source', kind: 'assistant', text: 'Original' }], apiMessages: [], kind: 'chat', status: 'idle' } as unknown as Thread
let resolve!: (value: typeof quote) => void, calls = 0, events: any[] = []
window.addEventListener(MESSAGE_QUOTE_EVENT, event => events.push((event as CustomEvent).detail))
try {
  api.prepareMessageQuote = async () => { calls++; return new Promise(done => { resolve = done }) }
  useStore.setState({ settings, threads: [a], view: { name: 'thread', id: 'a' } })
  const pending = requestMessageQuote({ threadId: 'a', itemId: 'source' })
  useStore.setState({ view: { name: 'thread', id: 'b' } }); resolve(quote); await pending
  assert.deepEqual(events, [], 'a delayed quotation must not leak into a newly visible chat')
  await requestMessageQuote({ threadId: 'a', itemId: 'source' }); assert.equal(calls, 1)
  useStore.setState({ view: { name: 'thread', id: 'a' } })
  const accepted = requestMessageQuote({ threadId: 'a', itemId: 'source' }); resolve(quote); await accepted
  assert.deepEqual(events, [{ threadId: 'a', quote }])
  const acceptedCount = events.length
  for (const changed of [
    { ...a, timeline: [{ id: 'source', kind: 'assistant', text: 'SYNTHETIC_NEW_OWNER_TEXT' }] },
    { ...a, conversationRevision: 1 }, { ...a, providerId: 'different-provider' }, { ...a, providerAccountId: 'different-account' },
    { ...a, workspace: { kind: 'ssh', hostId: 'different-host', path: '/synthetic/work' } }, { ...a, cwd: '/synthetic/different' },
    { ...a, incognito: true }, { ...a, googleLocalOnly: true },
  ]) {
    useStore.setState({ threads: [a] })
    const delayed = requestMessageQuote({ threadId: 'a', itemId: 'source' })
    useStore.setState({ threads: [changed as Thread] }); resolve(quote); await delayed
    assert.equal(events.length, acceptedCount, 'Same-ID source, revision or owner changes cannot receive a pending quote')
  }
  useStore.setState({ threads: [a] })
  for (const bad of [{ ...quote, sourceItemId: 'different-source' }, { ...quote, sourceText: 'Not the selected source', text: 'Not the selected source' }, { ...quote, sourceRevision: 'a'.repeat(64) }, { ...quote, text: 'Unexpected edited context' }]) {
    api.prepareMessageQuote = async () => bad
    await requestMessageQuote({ threadId: 'a', itemId: 'source' })
    assert.equal(events.length, acceptedCount, 'Prepared quotes must match the requested source, SHA-256 revision and unedited selection')
    assert.match(useStore.getState().notice!.message, /could not be quoted/)
  }
  api.prepareMessageQuote = async () => { calls++; return new Promise(done => { resolve = done }) }
  const callsBeforeRemote = calls
  useStore.setState({ threads: [{ ...a, orchestratorHostId: 'remote' }] }); await requestMessageQuote({ threadId: 'a', itemId: 'source' }); assert.equal(calls, callsBeforeRemote)
  useStore.setState({ threads: [a] }); api.prepareMessageQuote = async () => { throw new Error('Synthetic changed source') }
  await requestMessageQuote({ threadId: 'a', itemId: 'source' }); assert.match(useStore.getState().notice!.message, /changed source/)
} finally { api.prepareMessageQuote = old }
console.log('Actual quotation request routing fences late responses, remote owners and errors without sending or mutating another draft')
