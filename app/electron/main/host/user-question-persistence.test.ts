import assert from 'node:assert/strict'
import { createUserQuestionPersistence } from './user-question-persistence'
import type { Thread, UserQuestion } from '../../agent/types'

const item: any = { id: 'item-1', kind: 'tool', tool: 'ask_user', state: 'pending' }
const thread = { id: 'thread-1', timeline: [item] } as Thread
const run = { threadId: thread.id, runId: 'run-1', itemId: item.id }
const question: UserQuestion = { id: 'question-1', ...run, question: 'Which colour?', choices: ['Amber', 'Teal'], state: 'pending', answer: null, createdAt: 1, resolvedAt: null }
let release!: () => void
const events: string[] = []
const persistence = createUserQuestionPersistence({ getThread: (id) => id === thread.id ? thread : null,
  save: async (_run, patch) => { events.push('save'); await new Promise<void>((resolve) => { release = resolve }); events.push('durable'); Object.assign(item, patch) },
  emit: () => { events.push('emit') },
})
const first = persistence.commit(run, question)
await new Promise((resolve) => setImmediate(resolve))
assert.deepEqual(events, ['save'], 'question is not announced before encrypted persistence completes')
assert.equal(item.question, undefined, 'live IPC snapshots cannot see a staged question before persistence commits')
release(); await first
assert.deepEqual(events, ['save', 'durable', 'emit'])
const rejected = createUserQuestionPersistence({ getThread: () => thread, save: async () => { throw new Error('Example disk failure') }, emit: () => { throw new Error('must not announce failed save') } })
await assert.rejects(() => rejected.commit(run, { ...question, state: 'answered', answer: 'Amber', resolvedAt: 2 }), /disk failure/)
assert.equal(item.question.state, 'pending', 'a failed answer write restores the prior durable question')
assert.equal(item.state, 'awaiting')
assert.equal(item.question.answer, null)
await assert.rejects(() => rejected.commit({ ...run, itemId: 'another-item' }, question), /saved chat turn/)
const silent = createUserQuestionPersistence({ getThread: () => thread, save: async (_run, patch) => { Object.assign(item, patch) }, emit: () => { throw new Error('Example disposed window') } })
await silent.commit(run, { ...question, state: 'answered', answer: 'Amber', resolvedAt: 2 })
assert.equal(item.question.answer, 'Amber', 'renderer failure preserves an already durable answer')
await silent.drain()
console.log('user question durable host commit: ok')
