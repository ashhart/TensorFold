import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { runAgent } from '../loop'
import type { AgentEvent, Thread } from '../types'
import type { RunRecoverySnapshot } from '../../../shared/run-recovery'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-run-recovery-red-'))
  const saved = globalThis.fetch
  const events: AgentEvent[] = []
  const recovery: RunRecoverySnapshot[] = []
  let calls = 0
  const partial = 'A visible partial answer must survive a lost connection. '.repeat(6)
  globalThis.fetch = (async () => {
    calls++
    if (calls > 1) return new Response('data: {"choices":[{"delta":{"content":"Silently replayed answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } })
    return new Response(new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ choices: [{ delta: { content: partial }, finish_reason: null }] })}\n\n`))
      setTimeout(() => controller.error(Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' })), 5)
    } }), { headers: { 'content-type': 'text/event-stream' } })
  }) as typeof fetch
  const thread: Thread = { id: 'synthetic-partial', title: 'Synthetic', kind: 'chat', cwd: root, providerId: 'synthetic', model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Answer once.' }], timeline: [] }
  try {
    await runAgent({ thread, provider: { id: 'synthetic', name: 'Synthetic', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: [] }, model: 'synthetic', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', signal: new AbortController().signal, persist() {}, onRecovery: snapshot => recovery.push(snapshot), maxSteps: 1, autoCompact: false, requestApproval: async () => 'deny', emit(event) {
      events.push(event)
      if (event.type === 'item_add') thread.timeline.push(event.item)
      if (event.type === 'item_update') Object.assign(thread.timeline.find(item => item.id === event.id) ?? {}, event.patch)
      if (event.type === 'item_delta') { const item = thread.timeline.find(item => item.id === event.id); if (item) item.text = (item.text ?? '') + event.chunk }
    } })
    assert.ok(events.some(event => event.type === 'item_delta'), 'The fixture must produce visible partial output')
    assert.equal(calls, 1, 'A lost connection after partial output must never replay the billable request invisibly')
    assert.ok(thread.timeline.some(item => item.kind === 'assistant' && item.text?.includes('visible partial answer')), 'Partial text must remain visible')
    assert.equal(recovery.at(-1)?.state, 'needs_review')
    assert.equal(recovery.at(-1)?.reason, 'partial_response')
    assert.equal(recovery.at(-1)?.retry, 'requires_review')
    assert.equal(recovery.at(-1)?.pendingOperations, 0)
    console.log('PASS actual loop preserves partial output and refuses an invisible second request after connection loss')
  } finally { globalThis.fetch = saved; await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
