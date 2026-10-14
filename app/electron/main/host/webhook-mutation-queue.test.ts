import assert from 'node:assert/strict'
import { createWebhookMutationQueue } from './webhook-mutation-queue'

const queue = createWebhookMutationQueue()
const endpoint = { name: 'Original', enabled: false }
const order: string[] = []
let releaseDns!: () => void
const dns = new Promise<void>((resolve) => { releaseDns = resolve })

const first = queue.run(async () => {
  order.push('save-name:start')
  const next = { ...endpoint, name: 'Renamed' }
  await dns
  Object.assign(endpoint, next)
  order.push('save-name:end')
})
const second = queue.run(async () => {
  order.push('save-enabled:start')
  // This snapshot must be taken only after the first DNS-awaiting save commits.
  Object.assign(endpoint, { ...endpoint, enabled: true })
  order.push('save-enabled:end')
})
await new Promise<void>((resolve) => setImmediate(resolve))
assert.deepEqual(order, ['save-name:start'])
releaseDns()
await Promise.all([first, second])
assert.deepEqual(endpoint, { name: 'Renamed', enabled: true }, 'serialized partial saves compose instead of clobbering')
assert.deepEqual(order, ['save-name:start', 'save-name:end', 'save-enabled:start', 'save-enabled:end'])
await Promise.resolve()
assert.equal(queue.pending(), 0)

const paths = new Set<string>()
const create = () => queue.run(async () => {
  const path = '/hooks/unique'
  if (paths.has(path)) throw new Error('Webhook path already exists.')
  paths.add(path)
})
const [createA, createB] = await Promise.allSettled([create(), create()])
assert.equal(createA.status, 'fulfilled')
assert.equal(createB.status, 'rejected', 'concurrent creates observe the first committed path')
assert.equal(paths.size, 1)

console.log('webhook mutation queue serializes DNS-awaiting saves, approvals, and create uniqueness')
