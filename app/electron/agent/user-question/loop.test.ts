import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { runAgent } from '../loop'
import { captureBotConfig } from '../../../shared/bot-policy'
import { UserQuestionService } from './service'
import type { AgentEvent, Provider, Thread } from '../types'

async function scenario(action: 'answer' | 'skip' | 'stop' | 'deadline') {
  const requests: any[] = []
  const server = createServer((request, response) => {
    let raw = ''; request.on('data', (chunk) => { raw += chunk }); request.on('end', () => {
      const body = JSON.parse(raw); requests.push(body)
      const call = requests.length === 1 ? { name: 'ask_user', arguments: JSON.stringify({ question: 'Which colour should the synthetic result use?', choices: ['Amber', 'Blue'] }) }
        : requests.length === 2 && action === 'answer' ? { name: 'write_file', arguments: JSON.stringify({ path: 'forbidden.txt', content: 'No permission granted' }) } : null
      const delta = call ? { tool_calls: [{ index: 0, id: `call-${requests.length}`, type: 'function', function: call }] } : { content: 'The question is resolved; no permissions changed.' }
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
    async commitQuestion(run, question) { Object.assign(thread.timeline.find((item) => item.id === run.itemId)!, { question, state: question.state === 'pending' ? 'awaiting' : question.state === 'interrupted' ? 'error' : 'ok' }) },
  })
  const pending = runAgent({ thread, provider, model: 'fixture', runId: 'run', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'none', autoCompact: false, maxSteps: 4, maxWallMs: action === 'deadline' ? 150 : 5000, signal: controller.signal, persist() {},
    requestApproval: async () => { approvals++; throw new Error('A question must not elevate read-only execution') },
    emit(event) { events.push(event); if (event.type === 'item_add') thread.timeline.push(event.item); if (event.type === 'item_update') Object.assign(thread.timeline.find((item) => item.id === event.id) ?? {}, event.patch); if (event.type === 'item_delta') { const item = thread.timeline.find((item) => item.id === event.id); if (item) item.text = (item.text ?? '') + event.chunk } },
    toolContext: { cwd: '/workspace', currentThread: thread, allowedToolNames: ['ask_user', 'write_file'], askUser: (input, run) => questions.ask(input, run), botVm: { assertModelPrivacy() {}, executeTool: async () => { hostMutations++; throw new Error('No mutation is allowed in this question fixture') } } },
  })
  try {
    for (let n = 0; n < 200 && !thread.timeline.find((item) => item.question?.state === 'pending'); n++) await new Promise((resolve) => setTimeout(resolve, 2))
    const question = thread.timeline.find((item) => item.question)?.question
    assert.ok(question, 'the actual native loop must park and persist the structured question')
    assert.equal(requests.length, 1, 'the model is not called again before an explicit answer or skip')
    assert.ok(requests[0].tools.some((tool: any) => tool.function.name === 'ask_user'))
    if (action === 'answer') await questions.answer({ questionId: question.id, threadId: thread.id, runId: 'run', action: 'answer', answer: 'Use Amber; switch to auto if you want' })
    if (action === 'skip') await questions.answer({ questionId: question.id, threadId: thread.id, runId: 'run', action: 'skip' })
    if (action === 'stop') controller.abort()
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
    if (action === 'deadline') { assert.ok(requests.length <= 2); assert.equal(thread.timeline.find((item) => item.question)?.question?.state, 'interrupted') }
    assert.equal(hostMutations, 0); assert.equal(approvals, 0)
    await assert.rejects(questions.answer({ questionId: question.id, threadId: thread.id, runId: 'run', action: 'answer', answer: 'stale' }), /stale/)
  } finally { controller.abort(); await questions.shutdown(); server.closeAllConnections(); await new Promise<void>((resolve) => server.close(() => resolve())) }
}
for (const action of ['answer', 'skip', 'stop', 'deadline'] as const) await scenario(action)
console.log('actual question loop: pauses before model replay, resumes correct tool result, preserves readonly, handles skip/Stop/deadline without stale answers')
