import assert from 'node:assert/strict'
import { createSnapshotRefresh, type SnapshotRefreshState } from './snapshot-refresh'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
async function settle() { for (let i = 0; i < 16; i++) await Promise.resolve() }

const originalTimeout = globalThis.setTimeout
const originalClear = globalThis.clearTimeout
const timers = new Map<number, () => void>()
let timerId = 0
;(globalThis as any).setTimeout = (callback: () => void) => { const id = ++timerId; timers.set(id, callback); return id }
;(globalThis as any).clearTimeout = (id: number) => { timers.delete(id) }

try {
  const requests: ReturnType<typeof deferred<{ revision: number }>>[] = []
  const published: SnapshotRefreshState<{ revision: number }>[] = []
  let now = 40
  const controller = createSnapshotRefresh(() => {
    const request = deferred<{ revision: number }>()
    requests.push(request)
    return request.promise
  }, (state) => published.push(state), { now: () => now, timeoutMs: 100 })

  const initial = controller.refresh()
  assert.equal(controller.refresh(), initial, 'background invalidations share the in-flight read')
  await settle()
  assert.equal(requests.length, 1)
  assert.equal(controller.refresh(true), initial, 'an action can request a fresh read after an in-flight snapshot')
  controller.refresh(true)
  requests[0].resolve({ revision: 1 })
  await settle()
  assert.equal(requests.length, 2, 'multiple action refreshes coalesce into one trailing read')
  assert.equal(published.some((state) => state.snapshot?.revision === 1), false, 'pre-action response is never published')
  now = 50
  requests[1].resolve({ revision: 2 })
  await initial
  assert.deepEqual(controller.snapshot(), { snapshot: { revision: 2 }, loading: false, error: null, refreshedAt: 50 })
  assert.equal(timers.size, 0)

  const failed = controller.refresh()
  await settle()
  requests[2].reject(new Error('Host unavailable'))
  await failed
  assert.deepEqual(controller.snapshot(), { snapshot: { revision: 2 }, loading: false, error: 'Host unavailable', refreshedAt: 50 }, 'read errors preserve the last good snapshot and its freshness')
  const retry = controller.refresh()
  await settle()
  now = 70
  requests[3].resolve({ revision: 3 })
  await retry
  assert.equal(controller.snapshot().error, null, 'a successful retry clears the failed refresh state')
  assert.equal(controller.snapshot().refreshedAt, 70)

  const timedOut = controller.refresh()
  await settle()
  assert.equal(timers.size, 1)
  const expire = timers.values().next().value!
  timers.clear()
  expire()
  await timedOut
  assert.match(controller.snapshot().error!, /timed out/)
  assert.equal(controller.snapshot().loading, false, 'a hung bridge call cannot lock the refresh control forever')
  const afterTimeout = controller.refresh()
  await settle()
  requests[5].resolve({ revision: 5 })
  await afterTimeout
  const updatesBeforeLateResponse = published.length
  requests[4].resolve({ revision: 4 })
  await settle()
  assert.equal(controller.snapshot().snapshot?.revision, 5, 'an expired response cannot overwrite a newer successful retry')
  assert.equal(published.length, updatesBeforeLateResponse)

  const unmounting = controller.refresh()
  await settle()
  controller.refresh(true)
  const beforeUnmount = published.length
  controller.dispose()
  await unmounting
  assert.equal(timers.size, 0, 'dispose clears the deadline without waiting for the host')
  assert.equal(published.length, beforeUnmount, 'unmounted consumers receive no completion or error updates')
  requests[6].reject(new Error('Late bridge error'))
  await settle()
  await controller.refresh()
  assert.equal(requests.length, 7, 'dispose discards the queued read and future refreshes')

  const throws = createSnapshotRefresh(() => { throw new Error('Bridge missing') }, () => {})
  await throws.refresh()
  assert.equal(throws.snapshot().error, 'Bridge missing', 'synchronous bridge failures follow the same recoverable state path')
  throws.dispose()
  console.log('Assistant snapshot coalescing, post-action freshness, last-good retention, timeout fencing and cleanup passed')
} finally {
  globalThis.setTimeout = originalTimeout
  globalThis.clearTimeout = originalClear
}
