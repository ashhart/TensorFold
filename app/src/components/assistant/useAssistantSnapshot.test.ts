import assert from 'node:assert/strict'
import * as refreshModule from './snapshot-refresh'
import { harness, load, settle } from '../reading/reading-test-harness'

const hooks = harness()
const fakeDocument = new EventTarget() as EventTarget & { visibilityState: string }
fakeDocument.visibilityState = 'visible'
const fakeWindow = new EventTarget()
const oldDocument = Object.getOwnPropertyDescriptor(globalThis, 'document')
const oldWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
const originalTimeout = globalThis.setTimeout
const originalClear = globalThis.clearTimeout
const timers = new Map<number, { callback: () => void; ms: number }>()
let timerId = 0
;(globalThis as any).setTimeout = (callback: () => void, ms: number) => { const id = ++timerId; timers.set(id, { callback, ms }); return id }
;(globalThis as any).clearTimeout = (id: number) => { timers.delete(id) }
Object.defineProperty(globalThis, 'document', { configurable: true, value: fakeDocument })
Object.defineProperty(globalThis, 'window', { configurable: true, value: fakeWindow })
let reads = 0
let response: () => Promise<unknown> = async () => ({ generatedAt: ++reads })
const api = { getAssistantSnapshot: () => response() }
const { useAssistantSnapshot, ASSISTANT_REFRESH_MS } = load('src/components/assistant/useAssistantSnapshot.ts', {
  react: { ...hooks.hooks, useCallback: (callback: unknown) => callback }, '../../lib/ipc': { api }, './snapshot-refresh': refreshModule,
})
function render() { const value = hooks.render(() => useAssistantSnapshot()); hooks.flush(); return value }
function restore(name: string, descriptor: PropertyDescriptor | undefined) {
  if (descriptor) Object.defineProperty(globalThis, name, descriptor)
  else Reflect.deleteProperty(globalThis, name)
}
function runPoll() {
  const entry = [...timers].find(([, timer]) => timer.ms === ASSISTANT_REFRESH_MS)
  assert.ok(entry, 'a visible view schedules reconciliation')
  timers.delete(entry[0])
  entry[1].callback()
}

try {
  render()
  await settle()
  let hook = render()
  assert.equal(reads, 1, 'mount performs a snapshot read only')
  assert.equal(hook.snapshot.generatedAt, 1)
  assert.equal(timers.size, 1)
  runPoll()
  await settle()
  hook = render()
  assert.equal(reads, 2, 'visible reconciliation updates without manual interaction')
  assert.equal(timers.size, 1, 'each settled read schedules only one next poll')

  fakeDocument.visibilityState = 'hidden'
  fakeDocument.dispatchEvent(new Event('visibilitychange'))
  assert.equal(render().paused, true)
  assert.equal(timers.size, 0, 'background views do not keep a poll timer')
  fakeWindow.dispatchEvent(new Event('focus'))
  fakeWindow.dispatchEvent(new Event('online'))
  await settle()
  assert.equal(reads, 2, 'focus and network events never poll a hidden view')

  fakeDocument.visibilityState = 'visible'
  fakeDocument.dispatchEvent(new Event('visibilitychange'))
  fakeWindow.dispatchEvent(new Event('focus'))
  fakeWindow.dispatchEvent(new Event('online'))
  await settle()
  hook = render()
  assert.equal(reads, 3, 'returning to the view reconciles; simultaneous focus and online events coalesce')
  assert.equal(hook.paused, false)
  assert.equal(timers.size, 1)

  let finish!: (snapshot: unknown) => void
  response = () => { reads++; return new Promise((resolve) => { finish = resolve }) }
  runPoll()
  await settle()
  assert.equal(reads, 4)
  const manual = hook.refresh()
  response = async () => ({ generatedAt: ++reads })
  finish({ generatedAt: 4 })
  await manual
  hook = render()
  assert.equal(reads, 5, 'an explicit refresh during polling waits for a new post-action snapshot')
  assert.equal(hook.snapshot.generatedAt, 5)
  assert.equal(timers.size, 1)

  response = async () => { reads++; throw new Error('Disconnected') }
  await hook.refresh()
  hook = render()
  assert.equal(hook.snapshot.generatedAt, 5, 'a connection failure keeps the last usable inbox')
  assert.equal(hook.error, 'Disconnected')
  assert.equal(hook.loading, false)
  assert.equal(timers.size, 1, 'failed reads remain recoverable by the next visible reconciliation')

  hooks.cleanup()
  assert.equal(timers.size, 0)
  const readsBeforeCleanup = reads
  fakeWindow.dispatchEvent(new Event('focus'))
  fakeDocument.dispatchEvent(new Event('visibilitychange'))
  await hook.refresh()
  await settle()
  assert.equal(reads, readsBeforeCleanup, 'cleanup removes subscriptions and invalidates the refresh handle')
  console.log('Assistant visible-only polling, focus/network reconciliation, post-action refresh and cleanup passed')
} finally {
  hooks.cleanup()
  globalThis.setTimeout = originalTimeout
  globalThis.clearTimeout = originalClear
  restore('document', oldDocument)
  restore('window', oldWindow)
}
