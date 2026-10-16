import assert from 'node:assert/strict'
import { UserQuestionService } from './service'
import { normalizeUserQuestion, normalizeQuestionAnswer } from './validation'
import { executeTool, isReadOnlyTool } from '../tools'
import type { Thread, UserQuestion } from '../types'

const thread: Thread = { id: 't1', title: 'Synthetic question', kind: 'chat', cwd: '', providerId: 'fixture', model: 'fixture', status: 'idle', createdAt: 1, updatedAt: 1, apiMessages: [], timeline: [{ id: 'i1', kind: 'tool', tool: 'ask_user', state: 'running' }] }
let active = true, id = 0, failSave = false, blockAnswer = false, releaseAnswer: (() => void) | null = null
const commits: UserQuestion[] = []
const service = new UserQuestionService({ getThread: (value) => value === thread.id ? thread : null, isRunActive: (t, r) => active && t === 't1' && r === 'r1', nextId: () => `q${++id}`, now: () => 100 + id,
  async commitQuestion(run, question) {
    if (failSave) throw new Error('Synthetic encrypted store write failed')
    if (blockAnswer && question.state === 'answered') await new Promise<void>((resolve) => { releaseAnswer = resolve })
    commits.push(structuredClone(question)); Object.assign(thread.timeline.find((item) => item.id === run.itemId)!, { question: structuredClone(question), state: question.state === 'pending' ? 'awaiting' : question.state === 'interrupted' ? 'error' : 'ok' })
  },
})
const waitPending = async () => { for (let n = 0; n < 100 && thread.timeline[0].question?.state !== 'pending'; n++) await new Promise((resolve) => setTimeout(resolve, 1)); assert.equal(thread.timeline[0].question?.state, 'pending'); await Promise.resolve(); return thread.timeline[0].question! }
const ask = (signal: AbortSignal, choices = ['Amber', 'Blue']) => { delete thread.timeline[0].question; return service.ask({ question: 'Which colour should the fixture use?', choices }, { threadId: 't1', runId: 'r1', itemId: 'i1', signal }) }
const answer = (question: UserQuestion, value = 'Amber') => ({ questionId: question.id, threadId: 't1', runId: 'r1', action: 'answer' as const, answer: value })
assert.deepEqual(normalizeUserQuestion({ question: '  Which colour?  ', choices: [] }), { question: 'Which colour?', choices: [] })
for (const bad of [{ question: '' }, { question: 'x'.repeat(401) }, { question: 'Colour?', choices: ['a', 'b', 'c', 'd'] }, { question: 'Colour?', choices: ['Amber', 'amber'] }, { question: 'Colour?', choices: [''] }]) assert.throws(() => normalizeUserQuestion(bad))
assert.throws(() => normalizeQuestionAnswer({ questionId: 'q', threadId: 't', runId: 'r', action: 'answer', answer: ' ' }))
assert.ok(isReadOnlyTool('ask_user'), 'asking does not request permission to mutate')
{
  const controller = new AbortController(), pending = ask(controller.signal), question = await waitPending()
  let settled = false; void pending.then(() => { settled = true })
  await Promise.resolve(); assert.equal(settled, false, 'there is no default or automatic answer')
  await assert.rejects(service.ask({ question: 'Duplicate?' }, { threadId: 't1', runId: 'r1', itemId: 'i1', signal: controller.signal }), /already has a question/)
  await assert.rejects(service.answer({ ...answer(question), threadId: 't2' }), /another chat/)
  await assert.rejects(service.answer({ ...answer(question), runId: 'r2' }), /another chat/)
  const saved = await service.answer(answer(question, 'A custom typed colour'))
  assert.equal(saved.state, 'answered'); assert.equal(commits.at(-1)?.answer, 'A custom typed colour')
  assert.deepEqual(await pending, { state: 'answered', answer: 'A custom typed colour' })
  await assert.rejects(service.answer(answer(question)), /stale/)
}
{
  const controller = new AbortController(), pending = ask(controller.signal, []), question = await waitPending()
  assert.equal((await service.answer({ questionId: question.id, threadId: 't1', runId: 'r1', action: 'skip' })).answer, null)
  assert.deepEqual(await pending, { state: 'skipped', answer: null })
}
{
  const controller = new AbortController(), pending = ask(controller.signal), question = await waitPending()
  blockAnswer = true
  const first = service.answer(answer(question)); await Promise.resolve()
  await assert.rejects(service.answer(answer(question)), /already settling/)
  while (!releaseAnswer) await new Promise((resolve) => setTimeout(resolve, 1))
  ;(releaseAnswer as unknown as () => void)(); releaseAnswer = null; blockAnswer = false
  await first; await pending
}
{
  const controller = new AbortController(), pending = ask(controller.signal), question = await waitPending()
  blockAnswer = true
  const resolving = service.answer(answer(question))
  while (!releaseAnswer) await new Promise((resolve) => setTimeout(resolve, 1))
  const rejection = assert.rejects(pending, /interrupted/); controller.abort(); await rejection
  ;(releaseAnswer as unknown as () => void)(); releaseAnswer = null; blockAnswer = false
  await assert.rejects(resolving, /run stopped/)
  await service.quiesce()
  assert.equal(thread.timeline[0].question?.state, 'interrupted', 'cancellation wins over an answer whose store write was still pending')
  service.resume()
}
{
  const controller = new AbortController(); failSave = true
  await assert.rejects(ask(controller.signal), /store write failed/); failSave = false
  const pending = ask(controller.signal), question = await waitPending()
  active = false
  await assert.rejects(service.answer(answer(question)), /no longer active/)
  controller.abort(); await assert.rejects(pending, /interrupted/); active = true
}
{
  const previous: UserQuestion = { id: 'old', threadId: 't1', runId: 'old-run', itemId: 'i1', question: 'Previous question?', choices: [], state: 'pending', answer: null, createdAt: 1, resolvedAt: null }
  thread.timeline[0].question = previous
  assert.equal(await service.recoverInterrupted([thread]), 1)
  assert.equal(thread.timeline[0].question?.state, 'interrupted')
  await assert.rejects(service.answer({ questionId: 'old', threadId: 't1', runId: 'old-run', action: 'answer', answer: 'Never replay' }), /stale/)
}
{
  let guestCalls = 0
  const result = await executeTool('ask_user', { question: 'Which colour?', choices: ['Amber'] }, { cwd: '/workspace', currentThread: thread, signal: new AbortController().signal, allowedToolNames: ['ask_user'], userQuestionRunId: 'r1', userQuestionItemId: 'i1', botVm: { executeTool: async () => { guestCalls++; throw new Error('A question is not a guest or host command') } }, askUser: async () => ({ state: 'answered', answer: 'Do not elevate permissions' }) })
  assert.equal(result.ok, true); assert.equal(guestCalls, 0); assert.match(result.output, /not a permission change/)
}
{
  const controller = new AbortController(), pending = ask(controller.signal); await waitPending()
  const rejected = assert.rejects(pending, /interrupted/); await service.shutdown(); await rejected
  await assert.rejects(ask(new AbortController().signal), /interrupted/)
  thread.timeline[0].question = { ...thread.timeline[0].question!, state: 'pending' }
  const before = commits.length
  assert.equal((await service.state('t1', thread.timeline[0].question.id))?.state, 'interrupted')
  assert.equal(commits.length, before, 'a late state read during shutdown never schedules a new store write')
}
console.log('user questions: explicit answer/skip, duplicate/cross-run/stale refusal, durable resolution, cancellation, recovery and VM non-host dispatch passed')
