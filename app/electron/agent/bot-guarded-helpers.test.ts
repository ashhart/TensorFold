import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { compactThread, runAgent } from './loop'
import { captureBotConfig } from '../../shared/bot-policy'
import type { ApiMessage, Provider, Thread } from './types'

let calls = 0
const server = createServer((request, response) => {
  request.resume(); request.on('end', () => {
    calls++
    const content = calls % 2 ? 'The user prefers clear source evidence and direct explanations.' : '{"valid":true,"note":"Synthetic verification passed."}'
    response.writeHead(200, { 'content-type': 'text/event-stream' })
    response.end(`data: ${JSON.stringify({ choices: [{ delta: { content }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
const provider: Provider = { id: 'guarded-fixture', name: 'Synthetic local API alias', kind: 'local', baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, transport: 'chat-completions', apiKey: '', models: ['fixture'] }
const preset = { id: 'bot_guarded_fixture', name: 'Guarded bot', role: 'Use selected tools', lane: 'review' as const, providerId: provider.id, model: 'fixture', approvalScope: 'readonly' as const, createdAt: 1, updatedAt: 1 }
const history = (): ApiMessage[] => [{ role: 'system', content: 'Answer using synthetic facts only.' }, ...Array.from({ length: 80 }, (_, index) => [{ role: 'user' as const, content: `I always prefer explicit source evidence for synthetic item ${index}. ` + 'Synthetic context. '.repeat(60) }, { role: 'assistant' as const, content: `Synthetic response ${index}. ` + 'Recorded explanation. '.repeat(60) }]).flat()]
const make = (patch: Partial<Thread> = {}): Thread => {
  const apiMessages = history()
  return { id: 'guarded-chat', title: 'Synthetic chat', kind: 'chat', cwd: '', providerId: provider.id, model: 'fixture', createdAt: 1, updatedAt: 1, status: 'idle', timeline: [{ id: 'synthetic-user-origin', kind: 'user', memorySource: 'user', text: String(apiMessages[1].content) }], apiMessages, ...patch }
}
try {
  const selected = captureBotConfig({ ...preset, toolScope: { mode: 'selected', names: [] } }, [provider])
  const legacyVm = captureBotConfig({ ...preset, executionTarget: 'vm' }, [provider])
  for (const patch of [{ bot: selected }, { bot: legacyVm, cwd: '/workspace' }, { phoneLocalOnly: true }]) for (const trigger of ['auto', 'manual'] as const) {
    const before = calls, thread = make(patch)
    const entry = await compactThread(thread, provider, 'fixture', new AbortController().signal, trigger)
    assert.ok(entry)
    assert.equal(entry!.summaryMethod, 'deterministic-fallback')
    assert.equal(calls, before, 'guarded compaction never sends summarizer or verifier requests, including manual paths and legacy VM records')
    assert.ok(entry!.tokensAfter < entry!.tokensBefore)
  }
  const normal = make(), before = calls
  const verified = await compactThread(normal, provider, 'fixture', new AbortController().signal)
  assert.equal(calls - before, 2, 'ordinary chats retain their model summary and verification requests')
  assert.equal(verified?.summaryMethod, 'verified-model')
  for (const patch of [{ bot: selected }, { bot: legacyVm, cwd: '/workspace' }, { phoneLocalOnly: true }]) {
    let privatePolicy = false
    const thread = make(patch), events: any[] = [], start = calls
    const guard = () => { if (privatePolicy) throw new Error('Synthetic Cortheon privacy policy changed.') }
    await runAgent({ thread, provider, model: 'fixture', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', contextLimit: 1024, maxSteps: 2, maxWallMs: 3000, autoCompact: true, signal: new AbortController().signal, persist() { if (thread.compactions?.length) privatePolicy = true }, requestApproval: async () => 'deny', emit: (event) => { events.push(event) }, toolContext: { cwd: thread.cwd, currentThread: thread, botToolScopeGuard: guard, phonePrivacyGuard: () => privatePolicy ? 'Synthetic Cortheon privacy policy changed.' : null, ...(thread.bot?.executionTarget === 'vm' ? { allowedToolNames: ['read_file'], botVm: { assertModelPrivacy: guard, executeTool: async () => { throw new Error('No guest tool is requested.') } } } : {}) } })
    assert.equal(calls, start, 'a policy flip during compaction blocks the next primary request without any auxiliary request escaping first')
    assert.ok(events.some((event) => event.type === 'error' && /privacy policy changed/.test(event.message)))
  }
  console.log('Actual guarded auxiliary coverage passed: selected scopes, legacy VM and phone manual/automatic compaction make zero extra model requests; ordinary verified compaction stays active; policy-flip primary requests fail closed.')
} finally { server.closeAllConnections(); await new Promise<void>((resolve) => server.close(() => resolve())) }
