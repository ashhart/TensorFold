import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { runAgent } from '../loop'
import { PluginRuntime } from '../plugin-runtime'
import type { AgentEvent, Thread } from '../types'
import type { RunRecoverySnapshot } from '../../../shared/run-recovery'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-stop-drain-'))
  const saved = globalThis.fetch
  const controller = new AbortController()
  let begin!: () => void, release!: () => void
  const began = new Promise<void>(resolve => { begin = resolve })
  const pending = new Promise<void>(resolve => { release = resolve })
  let mutations = 0
  const snapshots: RunRecoverySnapshot[] = []
  const events: AgentEvent[] = []
  const schema = { type: 'function', function: { name: 'synthetic_mutation', description: 'A controlled in-memory mutation.', parameters: { type: 'object', properties: {} } } }
  const plugin = new PluginRuntime()
  plugin.listToolSchemaEntries = () => [{ pluginId: 'fixture', pluginName: 'Fixture', schema }]
  plugin.getRegisteredTool = name => name === 'synthetic_mutation' ? { pluginId: 'fixture', schema, handler: async () => { begin(); await pending; mutations++; return { ok: true, output: 'Confirmed one in-memory mutation.', changedPaths: ['synthetic.txt'] } } } : undefined
  globalThis.fetch = (async () => new Response(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'controlled', type: 'function', function: { name: 'synthetic_mutation', arguments: '{}' } }] }, finish_reason: 'tool_calls' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })) as typeof fetch
  const thread: Thread = { id: 'synthetic-stop', title: 'Synthetic', kind: 'chat', cwd: root, providerId: 'synthetic', model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Run the synthetic tool once.' }], timeline: [] }
  let settled = false
  const run = runAgent({ thread, provider: { id: 'synthetic', name: 'Synthetic', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: [] }, model: 'synthetic', approvalMode: 'auto', temperature: 0, reasoningEffort: 'low', signal: controller.signal, persist() {}, maxSteps: 2, autoCompact: false, pluginRuntime: plugin, onRecovery: snapshot => snapshots.push(snapshot), requestApproval: async () => 'allow', emit(event) {
    events.push(event)
    if (event.type === 'item_add') thread.timeline.push(event.item)
    if (event.type === 'item_update') Object.assign(thread.timeline.find(item => item.id === event.id) ?? {}, event.patch)
  } }).finally(() => { settled = true })
  try {
    await Promise.race([began, new Promise((_, reject) => setTimeout(() => reject(new Error('Synthetic tool did not start')), 2_000))])
    controller.abort()
    await new Promise<void>(resolve => setImmediate(resolve))
    assert.equal(settled, false)
    assert.equal(events.some(event => event.type === 'done'), false, 'No completion can precede the uncooperative mutation settling')
    assert.ok(snapshots.some(snapshot => snapshot.state === 'stop_requested'))
    assert.equal(snapshots.at(-1)?.state, 'stopping')
    assert.equal(snapshots.at(-1)?.pendingMutations, 1)
    assert.equal(snapshots.at(-1)?.canStartNewRun, false)
    assert.equal(mutations, 0)
    release()
    await run
    assert.equal(mutations, 1)
    assert.equal(snapshots.at(-1)?.state, 'stopped')
    assert.equal(snapshots.at(-1)?.pendingOperations, 0)
    assert.equal(snapshots.at(-1)?.completedMutations, 1)
    assert.equal(snapshots.at(-1)?.canStartNewRun, true)
    assert.ok(thread.timeline.some(item => item.changedPaths?.includes('synthetic.txt')))
    assert.match(String(thread.apiMessages.find(message => message.role === 'tool')?.content), /settled after Stop.*Confirmed result: completed/)
    console.log('PASS actual run Stop requests remain stopping and block admission until an uncooperative mutation settles; known effects and transcript evidence are retained')
  } finally { release(); await run; globalThis.fetch = saved; await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
