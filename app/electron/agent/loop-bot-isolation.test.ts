import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { botInstructions, captureBotConfig } from '../../shared/bot-policy'
import { resolveRunInstructionGraph, runAgent } from './loop'
import { PluginRuntime } from './plugin-runtime'
import type { AgentEvent, Provider, Thread } from './types'

const requests: any[] = []
let instructionReads = 0
assert.equal(await resolveRunInstructionGraph('/workspace', 'fixture', undefined, (async () => { instructionReads++; throw new Error('Host instruction resolver was reached.'); }) as any, 'vm'), null)
assert.equal(instructionReads, 0, 'a VM cwd can never be used by the host instruction resolver')
let guestReads = 0
let hostAccesses = 0
let hooks = 0
const server = createServer((request, response) => {
  let raw = ''
  request.on('data', (chunk) => { raw += chunk })
  request.on('end', () => {
    const body = JSON.parse(raw)
    requests.push(body)
    const calls = [
      { name: 'fixture_host_plugin', arguments: '{}' },
      { name: 'write_file', arguments: '{"path":"result.txt","content":"not allowed"}' },
      { name: 'read_file', arguments: '{"path":"notes.txt"}' },
    ]
    const selected = calls[requests.length - 1]
    const delta = selected ? { tool_calls: [{ index: 0, id: `call-${requests.length}`, type: 'function', function: selected }] } : { content: 'The guest notes are available; the host plugin and write were refused.' }
    response.writeHead(200, { 'Content-Type': 'text/event-stream' })
    response.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: selected ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
const provider: Provider = { id: 'fixture', name: 'Local fixture', kind: 'local', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, apiKey: '', models: ['fixture'] }
const bot = captureBotConfig({ id: 'bot_fixture', name: 'Guest reader', role: 'Read guest notes', lane: 'review', providerId: provider.id, model: 'fixture', executionTarget: 'vm', approvalScope: 'readonly', instructions: 'Read the guest file once.', createdAt: 1, updatedAt: 1 }, [provider])
const thread: Thread = { id: 'fixture-chat', title: 'Guest reader chat', kind: 'chat', cwd: '/workspace', providerId: provider.id, providerAccountId: null, model: 'fixture', bot, createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [{ role: 'user', content: 'Read notes.txt.' }], goal: 'HOST_GOAL_MUST_NOT_BE_IMPORTED', goalStatus: 'active' }
const plugin = new PluginRuntime()
plugin.listToolSchemaEntries = () => [{ pluginId: 'fixture', pluginName: 'Fixture', schema: { type: 'function', function: { name: 'fixture_host_plugin', parameters: { type: 'object', properties: {} } } } }]
plugin.getRegisteredTool = () => ({ pluginId: 'fixture', schema: {}, handler: async () => { hostAccesses++; return 'Unexpected host access.' } })
plugin.emitAgentEvent = async () => { hooks++ }
plugin.runPreToolCall = async (_name, args) => { hooks++; return { args, blocked: false } }
const events: AgentEvent[] = []
const controller = new AbortController()
const deps = { thread, provider, model: 'fixture', approvalMode: 'readonly' as const, temperature: 0, reasoningEffort: 'low', signal: controller.signal, persist() {}, autoCompact: false, maxSteps: 5, pluginRuntime: plugin, personaBlock: botInstructions(bot), projectMemory: 'HOST_MEMORY_MUST_NOT_BE_IMPORTED', requestApproval: async () => { throw new Error('Read-only bot must never ask to elevate.'); }, emit(event: AgentEvent) {
  events.push(event)
  if (event.type === 'item_add') thread.timeline.push(event.item)
  if (event.type === 'item_update') Object.assign(thread.timeline.find((item) => item.id === event.id) ?? {}, event.patch)
  if (event.type === 'item_delta') { const item = thread.timeline.find((candidate) => candidate.id === event.id); if (item) item.text = (item.text ?? '') + event.chunk }
}, toolContext: { cwd: '/workspace', currentThread: thread, allowedToolNames: ['read_file', 'write_file'], backend: { kind: 'local', readFile: async () => { hostAccesses++; throw new Error('Host filesystem must not be used.'); } } as any, botVm: { assertModelPrivacy() {}, executeTool: async (name: string) => { assert.equal(name, 'read_file'); guestReads++; return { ok: true, output: 'Synthetic guest notes.' } } } } }
try {
  await assert.rejects(() => runAgent({ ...deps, toolContext: { cwd: '/workspace' } }), /host fallback is forbidden/)
  await assert.rejects(() => runAgent({ ...deps, toolContext: { ...deps.toolContext, botVm: { executeTool: deps.toolContext.botVm.executeTool } } }), /model privacy guard is missing/)
  await assert.rejects(() => runAgent({ ...deps, approvalMode: 'auto' }), /cannot be elevated/)
  assert.equal(requests.length, 0, 'invalid bot context is refused before calling any model')
  await runAgent(deps)
  assert.equal(guestReads, 1)
  assert.equal(hostAccesses, 0)
  assert.equal(hooks, 0, 'VM conversations cannot reach host plugin hooks or event streams')
  assert.equal(requests.length, 4)
  for (const request of requests) {
    assert.ok((request.tools ?? []).every((tool: any) => ['read_file', 'write_file'].includes(tool.function.name)))
    const messages = JSON.stringify(request.messages)
    assert.ok(!messages.includes('HOST_MEMORY_MUST_NOT_BE_IMPORTED'))
    assert.ok(!messages.includes('HOST_GOAL_MUST_NOT_BE_IMPORTED'))
    assert.match(messages, /Guest reader/)
  }
  assert.ok(thread.timeline.some((item) => /not available.*executable capability/.test(item.output ?? '')))
  assert.ok(thread.timeline.some((item) => /Denied by read-only approval mode/.test(item.output ?? '')))
  assert.ok(events.some((event) => event.type === 'done'))
  requests.length = 0
  events.length = 0
  thread.timeline = []
  thread.apiMessages = [{ role: 'user', content: 'Read the guest notes again.' }]
  let privateModel = true
  deps.toolContext.botVm.assertModelPrivacy = () => { if (!privateModel) throw new Error('Cortheon must be off for this private VM run.') }
  deps.toolContext.botVm.executeTool = async (name: string) => { assert.equal(name, 'read_file'); privateModel = false; return { ok: true, output: 'Private guest notes must not leave this Mac.' } }
  await runAgent(deps)
  assert.equal(requests.length, 3, 'a changed research policy blocks the next model request before guest output can leave')
  assert.ok(events.some((event) => event.type === 'error' && /Cortheon/.test(event.message)))
  console.log('Actual bot loop isolation passed: VM guest-only dispatch, read-only refusal, captured identity, no host plugins/memory/goals and fail-closed startup.')
} finally {
  controller.abort()
  server.closeAllConnections()
  await new Promise<void>((resolve) => server.close(() => resolve()))
}
