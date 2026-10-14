import assert from 'node:assert/strict'
import { UserQuestionService } from './service'
import { createUserQuestionPersistence } from '../../main/host/user-question-persistence'
import type { Thread, UserQuestion } from '../types'

function gate() { let resolve!: () => void, reject!: (error: Error) => void; const promise = new Promise<void>((yes, no) => { resolve = yes; reject = no }); return { promise, resolve, reject } }
function fixture(options: { initial?: boolean; answer?: boolean; interruption?: boolean } = {}) {
  const item: any = { id: 'item', kind: 'tool', tool: 'ask_user', state: 'running' }
  const thread = { id: 'thread', timeline: [item] } as Thread
  const controller = new AbortController(), deadline = new AbortController()
  const initial = gate(), answer = gate(), interruption = gate()
  const writing = new Set<string>(), durable: UserQuestion[] = [], emitted: UserQuestion[] = []
  let holdAnswer = !!options.answer
  const persistence = createUserQuestionPersistence({ getThread: () => thread,
    async save(_run, patch) {
      const state = patch.question!.state; writing.add(state)
      if (state === 'pending' && options.initial) await initial.promise
      if (state === 'answered' && holdAnswer) await answer.promise
      if (state === 'interrupted' && options.interruption) await interruption.promise
      // This is the domain writer's synchronous onCommitted publication seam.
      Object.assign(item, patch); durable.push(structuredClone(patch.question!))
    }, emit(_run, patch) { emitted.push(structuredClone(patch.question!)) },
  })
  const service = new UserQuestionService({ getThread: () => thread, isRunActive: () => !controller.signal.aborted, nextId: () => 'question', commitQuestion: persistence.commit })
  const asked = service.ask({ question: 'Synthetic colour?', choices: ['Amber', 'Blue'] }, { threadId: thread.id, runId: 'run', itemId: item.id, signal: AbortSignal.any([controller.signal, deadline.signal]) })
  void asked.catch(() => {})
  const wait = async (state: string) => { for (let n = 0; n < 100 && !writing.has(state); n++) await new Promise((resolve) => setImmediate(resolve)); assert.ok(writing.has(state)); await new Promise((resolve) => setImmediate(resolve)) }
  const reply = () => service.answer({ questionId: 'question', threadId: thread.id, runId: 'run', action: 'answer', answer: 'Amber' })
  return { service, persistence, item, controller, deadline, initial, answer, interruption, writing, durable, emitted, asked, wait, reply, releaseAnswerHold: () => { holdAnswer = false } }
}

{
  const f = fixture({ answer: true }); await f.wait('pending')
  const resolving = f.reply(); await f.wait('answered')
  assert.equal(f.item.question.state, 'pending', 'threads:get cannot see an answer staged by a blocked save')
  assert.equal((await f.service.state('thread', 'question'))?.state, 'pending', 'direct question reads remain pending until durable acceptance')
  assert.equal((await f.service.state('thread', 'question'))?.answer, null)
  assert.ok(!f.emitted.some((question) => question.state === 'answered'))
  f.answer.resolve(); await resolving
  assert.deepEqual(await f.asked, { state: 'answered', answer: 'Amber' })
  assert.equal(f.item.question.state, 'answered'); await f.service.shutdown()
}
{
  const f = fixture({ answer: true }); await f.wait('pending')
  const resolving = f.reply(); void resolving.catch(() => {}); await f.wait('answered')
  f.answer.reject(new Error('Synthetic failed encrypted answer save'))
  await assert.rejects(resolving, /failed encrypted answer save/)
  assert.equal(f.item.question.state, 'pending')
  assert.equal((await f.service.state('thread', 'question'))?.state, 'pending')
  assert.equal(f.service.isWaiting('question'), true, 'failed persistence permits a real retry rather than accepting the answer')
  f.releaseAnswerHold(); await f.reply(); await f.asked; await f.service.shutdown()
}
for (const cancellation of ['stop', 'deadline'] as const) {
  const f = fixture({ initial: true, interruption: true }); await f.wait('pending')
  assert.equal(f.item.question, undefined, 'initial question remains detached while its save is blocked')
  if (cancellation === 'stop') f.controller.abort(); else f.deadline.abort()
  const settled = await Promise.race([f.asked.then(() => 'answered', () => 'interrupted'), new Promise<string>((resolve) => setTimeout(() => resolve('blocked-on-disk'), 100))])
  assert.equal(settled, 'interrupted', `${cancellation} must release the actual question await without waiting for disk readiness`)
  assert.equal(f.service.isWaiting('question'), false)
  assert.equal((await f.service.state('thread', 'question'))?.state, 'interrupted')
  f.initial.resolve(); await f.wait('interrupted')
  assert.equal(f.item.question.state, 'pending', 'the intentionally late initial save exposes the exact recovery race')
  assert.equal((await f.service.state('thread', 'question'))?.state, 'interrupted', 'authoritative terminal snapshot wins over late durable pending publication')
  let drained = false; const shuttingDown = f.service.shutdown().then(() => { drained = true })
  await new Promise((resolve) => setImmediate(resolve)); assert.equal(drained, false, 'Quit and purge still drain tracked late writes')
  const writes = f.writing.size; assert.equal((await f.service.state('thread', 'question'))?.state, 'interrupted'); assert.equal(f.writing.size, writes, 'closing reads never create more writes')
  f.interruption.resolve(); await shuttingDown; await f.persistence.drain()
  assert.equal(f.item.question.state, 'interrupted'); assert.equal(f.item.question.answer, null)
  await assert.rejects(f.reply(), /stale/)
}
console.log('question durability integration: no optimistic acceptance, retry on failed save, prompt Stop/deadline and drained late interrupted publication')
