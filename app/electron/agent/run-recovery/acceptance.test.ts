import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { runAgent } from '../loop'
import type { Thread } from '../types'
import type { RunRecoverySnapshot } from '../../../shared/run-recovery'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-acceptance-'))
  const saved = globalThis.fetch
  const provider = { id: 'synthetic', name: 'Synthetic', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: [] }
  const success = () => new Response('data: {"choices":[{"delta":{"content":"The response completed once."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } })
  try {
    for (const scenario of ['unknown', 'refused', 'predispatch'] as const) {
      let calls = 0
      const recovery: RunRecoverySnapshot[] = []
      globalThis.fetch = (async () => {
        calls++
        if (calls > 1) return success()
        if (scenario === 'refused') return new Response('{"error":{"message":"rate limited"}}', { status: 429 })
        throw Object.assign(new Error(scenario === 'unknown' ? 'socket hang up' : 'connection refused'), { code: scenario === 'unknown' ? 'ECONNRESET' : 'ECONNREFUSED' })
      }) as typeof fetch
      const thread: Thread = { id: scenario, title: 'Synthetic', kind: 'chat', cwd: root, providerId: 'synthetic', model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Answer once.' }], timeline: [] }
      await runAgent({ thread, provider, model: 'synthetic', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', signal: new AbortController().signal, persist() {}, autoCompact: false, maxSteps: 1, onRecovery: snapshot => recovery.push(snapshot), requestApproval: async () => 'deny', emit(event) { if (event.type === 'item_add') thread.timeline.push(event.item) } })
      if (scenario === 'unknown') { assert.equal(calls, 1); assert.equal(recovery.at(-1)?.delivery, 'unknown'); assert.equal(recovery.at(-1)?.retry, 'requires_review'); assert.equal(recovery.at(-1)?.state, 'needs_review') }
      else { assert.equal(calls, 2, 'Known refusal and pre-dispatch failure can retry within the existing cap'); assert.equal(recovery.at(-1)?.state, 'completed') }
    }
    console.log('PASS actual loop separates unknown delivery from safe pre-dispatch and explicit refusal recovery without dispatching tools')
  } finally { globalThis.fetch = saved; await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
