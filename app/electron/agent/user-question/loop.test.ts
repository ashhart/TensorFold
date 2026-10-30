import assert from 'node:assert/strict'
import { mock } from 'node:test'
import { createServer } from 'node:http'
import { runAgent } from '../loop'
import { captureBotConfig } from '../../../shared/bot-policy'
import { UserQuestionService } from './service'
import type { AgentEvent, Provider, Thread } from '../types'

class DeadlineClock {
  now = 1_000_000
  timers: { at: number; ms: number; controller: AbortController }[] = []
  private date = mock.method(Date, 'now', () => this.now)
  private timeout = mock.method(AbortSignal, 'timeout', (ms: number) => {
    const controller = new AbortController()
    this.timers.push({ at: this.now + ms, ms, controller })
    return controller.signal
  })
  expire(timer: DeadlineClock['timers'][number], wallBehind: boolean) {
    this.now = timer.at + (wallBehind ? -1 : 1)
    timer.controller.abort(new DOMException('Synthetic question deadline', 'TimeoutError'))
  }
  restore() { this.timeout.mock.restore(); this.date.mock.restore() }
}

async function scenario(action: 'answer' | 'skip' | 'stop' | 'deadline' | 'deadline-clock-skew' | 'deadline-batch') {
  const deadline = action.startsWith('deadline')
  const clock = deadline ? new DeadlineClock() : null
  let questionTimer: DeadlineClock['timers'][number] | undefined
  let questionSignal: AbortSignal | undefined
  let published!: () => void
  const questionPublished = new Promise<void>((resolve) => { published = resolve })
  const requests: any[] = []
  const server = createServer((request, response) => {
    let raw = ''; request.on('data', (chunk) => { raw += chunk }); request.on('end', () => {
      const body = JSON.parse(raw); requests.push(body)
      const call = requests.length === 1 ? { name: 'ask_user', arguments: JSON.stringify({ question: 'Which colour should the synthetic result use?', choices: ['Amber', 'Blue'] }) }
        : requests.length === 2 && action === 'answer' ? { name: 'write_file', arguments: JSON.stringify({ path: 'forbidden.txt', content: 'No permission granted' }) } : null
      const delta = call ? { tool_calls: [{ index: 0, id: `call-${requests.length}`, type: 'function', function: call }] } : { content: 'The question is resolved; no permissions changed.' }
      if (requests.length === 1 && action === 'deadline-batch' && 'tool_calls' in delta) delta.tool_calls!.push({ index: 1, id: 'after-question', type: 'function', function: { name: 'read_file', arguments: JSON.stringify({ path: 'never-read.txt' }) } })
      response.writeHead(200, { 'Content-Type': 'text/event-stream' }); response.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: call ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`)
    })
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const provider: Provider = { id: 'fixture', name: 'Local fixture', kind: 'local', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, apiKey: '', models: ['fixture'] }
  const bot = captureBotConfig({ id: 'question-bot', name: 'Question fixture', role: 'Clarify one synthetic choice', lane: 'review', providerId: provider.id, model: 'fixture', executionTarget: 'vm', approvalScope: 'readonly', createdAt: 1, updatedAt: 1 }, [provider])
  const thread: Thread = { id: 't', title: 'Question fixture', kind: 'task', cwd: '/workspace', providerId: provider.id, providerAccountId: null, model: 'fixture', bot, createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [{ role: 'user', content: 'Ask which colour to use before continuing.' }] }
  const controller = new AbortController(), events: AgentEvent[] = []
  let hostMutations = 0, approvals = 0
  const questions = new UserQuestionService({ getThread: (id) => id === thread.id ? thread : null, isRunActive: (id, runId) => id === 't' && runId === 'run' && !controller.signal.aborted,
    async commitQuestion(run, question) { Object.assign(thread.timeline.find((item) => item.id === run.itemId)!, { question, state: question.state === 'pending' ? 'awaiting' : question.state === 'interrupted' ? 'error' : 'ok' }); if (question.state === 'pending') published() },
  })
  const pending = runAgent({ thread, provider, model: 'fixture', runId: 'run', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'none', autoCompact: false, maxSteps: 4, maxWallMs: deadline ? 150 : undefined, signal: controller.signal, persist() {},
    requestApproval: async () => { approvals++; throw new Error('A question must not elevate read-only execution') },
    emit(event) { events.push(event); if (event.type === 'item_add') thread.timeline.push(event.item); if (event.type === 'item_update') Object.assign(thread.timeline.find((item) => item.id === event.id) ?? {}, event.patch); if (event.type === 'item_delta') { const item = thread.timeline.find((item) => item.id === event.id); if (item) item.text = (item.text ?? '') + event.chunk } },
    toolContext: { cwd: '/workspace', currentThread: thread, allowedToolNames: ['ask_user', 'write_file', 'read_file'], askUser: (input, run) => { questionSignal = run.signal; questionTimer = clock?.timers.at(-1); return questions.ask(input, run) }, botVm: { assertModelPrivacy() {}, executeTool: async () => { hostMutations++; throw new Error('No mutation is allowed in this question fixture') } } },
  })
  try {
    await Promise.race([questionPublished, pending.then(() => { throw new Error('Run ended before publishing its question') })])
    // Drain the publication's promise continuations so the service has marked the question ready for an answer.
    await new Promise<void>((resolve) => setImmediate(resolve))
    const question = thread.timeline.find((item) => item.question)?.question
    assert.ok(question, 'the actual native loop must park and persist the structured question')
    assert.equal(requests.length, 1, 'the model is not called again before an explicit answer or skip')
    assert.ok(requests[0].tools.some((tool: any) => tool.function.name === 'ask_user'))
    if (action === 'answer') await questions.answer({ questionId: question.id, threadId: thread.id, runId: 'run', action: 'answer', answer: 'Use Amber; switch to auto if you want' })
    if (action === 'skip') await questions.answer({ questionId: question.id, threadId: thread.id, runId: 'run', action: 'skip' })
    if (action === 'stop') controller.abort()
    if (deadline) {
      assert.ok(clock && questionTimer && questionSignal)
      assert.equal(questionTimer.ms, 150, 'the actual ask_user timeout owns the remaining run budget')
      assert.equal(questionSignal.aborted, false)
      clock.expire(questionTimer, action === 'deadline-clock-skew')
      assert.equal(questionSignal.aborted, true, 'expiry must reach the actual parked question signal')
    }
    await pending
    if (action === 'answer') {
      assert.equal(requests.length, 3)
      const result = requests[1].messages.find((message: any) => message.role === 'tool' && message.name === 'ask_user')
      assert.match(result.content, /Use Amber/); assert.match(result.content, /not a permission change/)
      assert.ok(thread.timeline.some((item) => /Denied by read-only/.test(item.output ?? '')))
      assert.equal(thread.timeline.find((item) => item.question)?.question?.state, 'answered')
    }
    if (action === 'skip') { assert.equal(requests.length, 2); assert.match(requests[1].messages.find((message: any) => message.name === 'ask_user').content, /explicitly skipped/); assert.equal(thread.timeline.find((item) => item.question)?.question?.answer, null) }
    if (action === 'stop') { assert.equal(requests.length, 1); assert.equal(thread.timeline.find((item) => item.question)?.question?.state, 'interrupted') }
    if (deadline) {
      assert.equal(thread.timeline.find((item) => item.question)?.question?.state, 'interrupted')
      assert.equal(requests.length, 2, 'an expired question cannot replay the normal model loop')
      assert.equal(requests[1].tools, undefined, 'only the existing tool-free final summary may follow expiry')
      assert.ok(requests[1].messages.some((message: any) => /time budget reached/.test(message.content ?? '')))
      if (action === 'deadline-batch') {
        assert.equal(hostMutations, 0, 'deadline expiry must stop the remaining tool batch')
        const skipped = requests[1].messages.filter((message: any) => message.tool_call_id === 'after-question')
        assert.equal(skipped.length, 1, 'the summary receives one honest result for the skipped sibling call')
        assert.match(skipped[0].content, /interrupted/)
      }
    }
    assert.equal(hostMutations, 0); assert.equal(approvals, 0)
    await assert.rejects(questions.answer({ questionId: question.id, threadId: thread.id, runId: 'run', action: 'answer', answer: 'stale' }), /stale/)
  } finally { controller.abort(); await questions.shutdown(); server.closeAllConnections(); await new Promise<void>((resolve) => server.close(() => resolve())); clock?.restore() }
}
for (const action of ['answer', 'skip', 'stop', 'deadline', 'deadline-clock-skew', 'deadline-batch'] as const) await scenario(action)
console.log('actual question loop: pauses before model replay, resumes correct tool result, preserves readonly, handles skip/Stop/deadline without stale answers')
