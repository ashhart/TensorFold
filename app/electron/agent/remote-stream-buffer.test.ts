import assert from 'node:assert/strict'
import { KeepTailRemoteEventBuffer, remoteEventWireBytes } from './remote-stream-buffer'
import type { RemoteControlEvent } from './types'

const delta = (id: number, chunk = 'x'.repeat(80)): RemoteControlEvent => ({
  id: String(id), kind: 'agent', createdAt: id,
  data: { runId: 'run', threadId: 'thread', type: 'item_delta', id: 'item', field: 'text', chunk },
})
const update = (id: number): RemoteControlEvent => ({
  id: String(id), kind: 'agent', createdAt: id,
  data: { runId: 'run', threadId: 'thread', type: 'item_update', id: 'item', patch: { state: 'ok' } },
})

{
  const one = remoteEventWireBytes(delta(1))
  const buffer = new KeepTailRemoteEventBuffer(one * 2 + 20)
  for (let id = 1; id <= 6; id++) assert.equal(buffer.enqueue(delta(id)).overflow, false)
  assert.ok(buffer.length <= 2, 'a blocked delta flood stays bounded')
  const gap = buffer.takeGapEvent(100)
  assert.equal(gap?.kind, 'stream_gap')
  assert.equal((gap?.data as any).fromEventId, '1')
  assert.ok((gap?.data as any).droppedEvents >= 4)
  assert.deepEqual([buffer.shift()?.id, buffer.shift()?.id], ['5', '6'], 'the newest streamed deltas are retained')
}

{
  const cap = remoteEventWireBytes(update(1)) + 10
  const buffer = new KeepTailRemoteEventBuffer(cap)
  assert.equal(buffer.enqueue(update(1)).overflow, false)
  assert.equal(buffer.enqueue(delta(2)).overflow, true, 'ordered non-delta state is never dropped or overtaken')
}

{
  const huge = delta(1, 'x'.repeat(4_000))
  const buffer = new KeepTailRemoteEventBuffer(200)
  assert.equal(buffer.enqueue(huge).overflow, true, 'one oversized event forces a clean reconnect instead of unbounded buffering')
}

console.log('remote stream keep-tail buffer tests passed')
