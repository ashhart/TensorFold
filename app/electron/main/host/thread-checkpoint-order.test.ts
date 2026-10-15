import assert from 'node:assert/strict'
import { recordPreTurnCheckpoint } from './thread-runtime'

{
  const calls: string[] = []
  let finishCapture!: (value: { ok: boolean; sha?: string }) => void
  const captureDone = new Promise<{ ok: boolean; sha?: string }>((resolve) => { finishCapture = resolve })
  const pending = recordPreTurnCheckpoint({
    threadId: 'thread-1',
    turn: 2,
    target: { cwd: '/repo', kind: 'task' },
    conversation: { apiMessages: [], timeline: [] },
  }, {
    captureCheckpoint: async () => { calls.push('capture:start'); return captureDone },
    recordTurnCheckpoint: (_threadId, _turn, _conversation, sha) => { calls.push(`record:${sha ?? 'none'}`) },
  })
  await Promise.resolve()
  assert.deepEqual(calls, ['capture:start'], 'conversation recording/model startup cannot overtake the file snapshot')
  finishCapture({ ok: true, sha: 'checkpoint-sha' })
  await pending
  assert.deepEqual(calls, ['capture:start', 'record:checkpoint-sha'], 'the successful file SHA is correlated into the turn record')
}

{
  let captured = false
  let recorded: string | null | undefined
  await recordPreTurnCheckpoint({
    threadId: 'chat-1', turn: 1, target: { kind: 'chat' }, conversation: { apiMessages: [], timeline: [] },
  }, {
    captureCheckpoint: async () => { captured = true; return { ok: true, sha: 'wrong' } },
    recordTurnCheckpoint: (_threadId, _turn, _conversation, sha) => { recorded = sha },
  })
  assert.equal(captured, false, 'chat turns never touch git')
  assert.equal(recorded, null, 'conversation-only checkpoints have no fabricated file ref')
}

{
  let recorded: string | null | undefined
  await recordPreTurnCheckpoint({
    threadId: 'task-1', turn: 1, target: { cwd: '/repo', kind: 'task' }, conversation: { apiMessages: [], timeline: [] },
  }, {
    captureCheckpoint: async () => { throw new Error('git unavailable') },
    recordTurnCheckpoint: (_threadId, _turn, _conversation, sha) => { recorded = sha },
  })
  assert.equal(recorded, null, 'a failed file capture still records an honest conversation-only checkpoint')
}

console.log('thread checkpoint ordering tests passed')
