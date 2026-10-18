import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { runAgent } from '../loop'
import { PluginRuntime } from '../plugin-runtime'
import type { Thread } from '../types'
import type { RunRecoverySnapshot } from '../../../shared/run-recovery'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-uncertain-tool-'))
  const saved = globalThis.fetch
  let executions = 0, requests = 0
  const states: RunRecoverySnapshot[] = []
  const schema = { type: 'function', function: { name: 'synthetic_delivery', description: 'Synthetic outbound delivery.', parameters: { type: 'object', properties: {} } } }
  const plugin = new PluginRuntime()
  plugin.listToolSchemaEntries = () => [{ pluginId: 'fixture', pluginName: 'Fixture', schema }]
  plugin.getRegisteredTool = name => name === 'synthetic_delivery' ? { pluginId: 'fixture', schema, handler: async () => { executions++; return { ok: false, output: 'The synthetic destination accepted the action but its reply was lost.' } } } : undefined
  globalThis.fetch = (async () => { requests++; return new Response(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: `action-${requests}`, type: 'function', function: { name: 'synthetic_delivery', arguments: '{}' } }] }, finish_reason: 'tool_calls' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } }) }) as typeof fetch
  const thread: Thread = { id: 'synthetic-uncertain', title: 'Synthetic', kind: 'chat', cwd: root, providerId: 'synthetic', model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Perform the synthetic delivery once.' }], timeline: [] }
  try {
    await runAgent({ thread, provider: { id: 'synthetic', name: 'Synthetic', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: [] }, model: 'synthetic', approvalMode: 'auto', temperature: 0, reasoningEffort: 'low', signal: new AbortController().signal, persist() {}, maxSteps: 4, autoCompact: false, pluginRuntime: plugin, onRecovery: state => states.push(state), requestApproval: async () => 'allow', emit(event) { if (event.type === 'item_add') thread.timeline.push(event.item); if (event.type === 'item_update') Object.assign(thread.timeline.find(item => item.id === event.id) ?? {}, event.patch) } })
    assert.equal(executions, 1, 'An opaque action with uncertain delivery must pause before the model can repeat it')
    assert.equal(requests, 1, 'No automatic model continuation may replay the uncertain action')
    assert.equal(states.at(-1)?.state, 'needs_review')
    assert.equal(states.at(-1)?.reason, 'tool_uncertain')
    assert.equal(states.at(-1)?.retry, 'requires_review')
    console.log('PASS actual uncertain action delivery pauses after one dispatch with explicit reconciliation before replay')
  } finally { globalThis.fetch = saved; await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
