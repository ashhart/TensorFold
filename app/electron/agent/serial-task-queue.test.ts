import assert from 'node:assert/strict'
import { createSerialTaskQueue } from './serial-task-queue.ts'

const queue = createSerialTaskQueue()
const order: string[] = []
let releaseFirst!: () => void
const firstBlocked = new Promise<void>((resolve) => { releaseFirst = resolve })

const first = queue.run('email', async () => {
  order.push('first:start')
  await firstBlocked
  order.push('first:end')
  return 1
})
const second = queue.run('email', async () => {
  order.push('second:start')
  return 2
})
const independent = queue.run('slack', async () => {
  order.push('slack')
  return 3
})

await new Promise<void>((resolve) => setImmediate(resolve))
assert.deepEqual(order, ['first:start', 'slack'], 'same-key work waits while a different key remains independent')
releaseFirst()
assert.deepEqual(await Promise.all([first, second, independent]), [1, 2, 3])
assert.deepEqual(order, ['first:start', 'slack', 'first:end', 'second:start'])

await queue.run('recover', async () => { throw new Error('expected') }).catch(() => {})
assert.equal(await queue.run('recover', async () => 4), 4, 'a rejected task does not poison the key')
await Promise.resolve()
assert.equal(queue.pendingKeys(), 0, 'settled keys are evicted')

console.log('serial task queue tests passed')
