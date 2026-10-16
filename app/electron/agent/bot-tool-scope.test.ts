import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { captureBotConfig, normalizeAgentPresets } from '../../shared/bot-policy'
import { botAllowedToolNames, botToolBlockReason, normalizeBotToolScope } from '../../shared/bot-tool-policy'
import { scopeBotToolContext } from './bot-tool-scope'
import { executeTool, type ToolContext } from './tools'
import { executeRuntimeTool, runAgent } from './loop'
import { PluginRuntime } from './plugin-runtime'
import { setEngineServeHook } from './engine/provider'
import { planningGuard } from './planning-guard'
import { mostRecentCompatibleBotChat } from '../../src/lib/bot-chats'
import type { AgentPreset, McpToolDefinition, Provider, Thread } from './types'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-bot-tool-scope-'))
const priorHome = process.env.VONTRA_HOME
process.env.VONTRA_HOME = profile
const provider: Provider = { id: 'scope-fixture', name: 'Synthetic local API', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['fixture'] }
const preset: AgentPreset = { id: 'bot_scope_fixture', name: 'Selected reader', role: 'Read selected synthetic material', instructions: 'Use only synthetic selected material.', lane: 'review', providerId: provider.id, model: 'fixture', approvalScope: 'review', toolScope: { mode: 'selected', names: ['read_file'] }, createdAt: 1, updatedAt: 1 }
const makeThread = (bot = captureBotConfig(preset, [provider])): Thread => ({ id: 'bot-scope-chat', title: 'Selected reader chat', kind: 'chat', cwd: '', providerId: bot.providerId, providerAccountId: null, model: bot.model, bot, createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [{ role: 'user', content: 'Use only the selected tools.' }] })
const mcp = (name: string): McpToolDefinition => ({ serverId: 'fixture', serverName: 'Synthetic source', approvalMode: 'ask', name, callName: `mcp__fixture__${name}`, inputSchema: { type: 'object', properties: { value: { type: 'string' } } }, description: name === 'hidden' ? 'UNSELECTED_MCP_METADATA_SENTINEL' : 'Selected synthetic tool.' })
const selectedMcp = Array.from({ length: 25 }, (_, index) => mcp(`selected_${index}`))
const hidden = mcp('hidden')
let mcpCalls = 0, reads = 0, writes = 0, hooks = 0, childCalls = 0, approvals = 0
let afterRead = () => {}
const context = (thread: Thread): ToolContext => ({ cwd: profile, currentThread: thread, mcpTools: [...selectedMcp, hidden],
  botToolScopeGuard() {},
  callMcpTool: async (name, args) => { assert.ok(selectedMcp.some((tool) => tool.callName === name)); assert.deepEqual(args, { value: 'synthetic' }); mcpCalls++; return { ok: true, output: 'Selected MCP result.' } },
  launchSubAgent: async () => { childCalls++; throw new Error('An unrestricted child must never launch.') },
  backend: { kind: 'local', readFile: async () => { reads++; afterRead(); return 'Selected synthetic notes.' }, writeFile: async () => { writes++ } } as any,
})
const plugin = new PluginRuntime()
plugin.listToolSchemaEntries = () => [{ pluginId: 'fixture', pluginName: 'Synthetic plugin', schema: { type: 'function', function: { name: 'fixture_plugin', parameters: { type: 'object', properties: {} } } } }]
plugin.getRegisteredTool = () => ({ pluginId: 'fixture', schema: {}, handler: async () => { hooks++; return 'Must not execute.' } })
plugin.emitAgentEvent = async () => { hooks++ }
plugin.emitSessionStart = () => { hooks++ }
plugin.queueSessionEnd = async () => { hooks++ }
plugin.runPreToolCall = async (_name, args) => { hooks++; return { args, blocked: false } }
plugin.runPostToolCall = async () => { hooks++ }
let mode = 'scoped', count = 0
const requests: any[] = []
const server = createServer((request, response) => {
  let raw = ''; request.on('data', (chunk) => { raw += chunk }); request.on('end', () => {
    requests.push(JSON.parse(raw))
    const sequence = mode === 'readonlyWrite' ? [{ name: 'write_file', arguments: '{"path":"not-written.txt","content":"synthetic"}' }] : mode === 'race' || mode === 'cloud' ? [{ name: 'read_file', arguments: '{"path":"notes.txt"}' }] : [
      { name: 'read_file', arguments: '{"path":"notes.txt"}' },
      { name: 'write_file', arguments: '{"path":"blocked.txt","content":"blocked"}' },
      { name: 'fixture_plugin', arguments: '{}' },
      { name: 'launch_subagent', arguments: '{"task":"Bypass the selected scope"}' },
      { name: 'tool_call', arguments: JSON.stringify({ name: hidden.callName, arguments: { value: 'blocked' } }) },
      { name: 'tool_call', arguments: JSON.stringify({ name: selectedMcp[0].callName, arguments: { value: 'synthetic' } }) },
    ]
    const selected = sequence[count++]
    const delta = selected ? { tool_calls: [{ index: 0, id: `scope-call-${count}`, type: 'function', function: selected }] } : { content: 'The selected read and MCP call completed; other tools were refused.' }
    response.writeHead(200, { 'content-type': 'text/event-stream' })
    response.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: selected ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
provider.baseUrl = `http://127.0.0.1:${(server.address() as any).port}/v1`
try {
  assert.equal(normalizeBotToolScope(undefined), undefined)
  assert.deepEqual(normalizeBotToolScope({ mode: 'selected', names: ['read_file', 'glob', 'read_file'] }), { mode: 'selected', names: ['glob', 'read_file'] })
  const damaged = [null, [], { mode: 'all', names: [] }, { mode: 'selected', names: 'read_file' }, { mode: 'selected', names: [7] }, { mode: 'selected', names: ['read_file'], extra: true }, { mode: 'selected', names: ['read_file\nwrite_file'] }, { mode: 'selected', names: Array(201).fill('read_file') }, { mode: 'selected', names: ['delegate_task'] }]
  for (const bad of damaged) {
    assert.throws(() => normalizeBotToolScope(bad), /Invalid|unavailable/)
    assert.throws(() => normalizeAgentPresets([{ ...preset, toolScope: bad }], true))
    const retained = normalizeAgentPresets([{ ...preset, toolScope: bad }])
    assert.equal(retained.length, 1, 'damaged saved selection never erases a bot')
    assert.deepEqual(retained[0].toolScope, { mode: 'selected', names: [] })
    assert.deepEqual(retained[0].toolScopeProblem?.value, bad)
    assert.throws(() => captureBotConfig(retained[0], [provider]), /Repair/)
    assert.deepEqual(normalizeAgentPresets(retained, true)[0].toolScopeProblem?.value, bad, 'quarantined selections survive unrelated strict settings transactions')
  }
  const captured = captureBotConfig(preset, [provider])
  preset.toolScope!.names.push('glob')
  assert.deepEqual(captured.toolScope?.names, ['read_file'], 'preset edits cannot mutate a captured chat')
  assert.equal(mostRecentCompatibleBotChat(preset, [provider], [makeThread(captured)]), null)
  assert.equal(mostRecentCompatibleBotChat({ ...preset, toolScope: { mode: 'selected', names: ['read_file'] } }, [provider], [makeThread(captured)])?.id, 'bot-scope-chat')
  const { loadSettings, saveSettings, saveThreads, loadThreads } = await import('../store')
  const initial = await loadSettings()
  const recipeDefault = { ...preset, id: 'recipe_fixture', paused: true, toolScope: { mode: 'selected' as const, names: [] } }
  await saveSettings({ ...initial, providers: [provider], agentPresets: [preset, recipeDefault] })
  const restarted = await loadSettings()
  assert.deepEqual(restarted.agentPresets[0].toolScope, { mode: 'selected', names: ['glob', 'read_file'] })
  assert.deepEqual(restarted.agentPresets[1].toolScope, { mode: 'selected', names: [] }, 'a paused imported recipe keeps its empty selection through actual protected settings restart')
  await saveThreads([makeThread(captured)])
  assert.deepEqual((await loadThreads())[0].bot?.toolScope, { mode: 'selected', names: ['read_file'] }, 'actual protected thread/settings persistence keeps the old captured scope across restart')
  for (const [index, bad] of damaged.entries()) {
    await saveSettings({ ...initial, providers: [provider], agentPresets: [{ ...preset, id: `damaged_${index}`, toolScope: bad as any }] })
    const loaded = await loadSettings()
    assert.equal(loaded.agentPresets.length, 1)
    assert.deepEqual(loaded.agentPresets[0].toolScopeProblem?.value, bad)
    await saveSettings({ ...loaded, theme: 'light' })
    const again = await loadSettings()
    assert.deepEqual(again.agentPresets[0].toolScopeProblem?.value, bad)
    assert.equal(again.agentPresets[0].instructions, preset.instructions)
    assert.throws(() => captureBotConfig(again.agentPresets[0], [provider]), /Repair/)
    const repaired = { ...again.agentPresets[0], toolScope: { mode: 'selected' as const, names: [] }, toolScopeProblem: undefined }
    assert.deepEqual(captureBotConfig(repaired, [provider]).toolScope?.names, [])
  }

  const directThread = makeThread(captured), direct = context(directThread)
  assert.equal((await executeTool('write_file', { path: 'blocked.txt', content: 'blocked' }, direct)).ok, false)
  assert.equal((await executeTool('launch_subagent', { task: 'Bypass' }, direct)).ok, false)
  assert.equal((await executeRuntimeTool(plugin, 'fixture_plugin', {}, direct, new Set())).ok, false)
  assert.equal(hooks + childCalls, 0, 'direct plugin dispatch and delegation are refused before handlers')
  assert.equal((await executeTool('read_file', { path: 'notes.txt' }, direct)).ok, true)
  assert.equal(reads, 1)
  const disabled = scopeBotToolContext(captured, { ...direct, disabledCapabilityIds: ['tool:workspace-read'] })
  assert.deepEqual(disabled.allowedToolNames, [])
  assert.equal((await executeTool('read_file', { path: 'notes.txt' }, { ...direct, disabledCapabilityIds: ['tool:workspace-read'] })).ok, false)
  // VM and any supplied parent allowlist can only narrow a selected scope.
  assert.deepEqual(botAllowedToolNames(captured, ['glob']), [])
  assert.equal((await executeTool('read_file', { path: 'notes.txt' }, { ...direct, disabledToolNames: ['read_file'] })).ok, false)

  const mcpBot = captureBotConfig({ ...preset, toolScope: { mode: 'selected', names: ['read_file', ...selectedMcp.map((tool) => tool.callName)] } }, [provider])
  const mcpThread = makeThread(mcpBot), raw = context(mcpThread)
  assert.equal((await executeTool('tool_call', { name: hidden.callName, arguments: {} }, raw)).ok, false)
  assert.equal((await executeTool('tool_describe', { name: hidden.callName }, raw)).ok, false)
  const catalog = await executeTool('tool_search', { query: 'Synthetic', limit: 100 }, raw)
  assert.ok(!catalog.output.includes(hidden.callName) && !catalog.output.includes('UNSELECTED_MCP_METADATA_SENTINEL'))
  assert.equal((await executeTool(selectedMcp[0].callName, { value: 'synthetic' }, raw)).ok, true)
  assert.equal((await executeTool('tool_call', { name: selectedMcp[0].callName, arguments: { value: 'synthetic' } }, raw)).ok, true)
  const before = mcpCalls
  const indirect = scopeBotToolContext(mcpBot, raw)
  assert.equal((await indirect.callMcpTool!(hidden.callName, {})).ok, false)
  assert.equal(mcpCalls, before)
  let guestCalls = 0
  const vmThread = makeThread({ ...mcpBot, executionTarget: 'vm' })
  const guestContext = { ...raw, currentThread: vmThread, allowedToolNames: ['read_file'], botVm: { executeTool: async () => { guestCalls++; return { ok: true, output: 'Synthetic guest data.' } } } }
  assert.equal((await executeTool('tool_call', { name: selectedMcp[0].callName, arguments: { value: 'synthetic' } }, guestContext)).ok, false)
  assert.equal((await executeTool(selectedMcp[0].callName, { value: 'synthetic' }, guestContext)).ok, false)
  assert.equal((await executeTool('read_file', { path: 'notes.txt' }, guestContext)).ok, true)
  assert.equal(guestCalls, 1); assert.equal(mcpCalls, before, 'a selected host/MCP tool cannot widen the existing VM guest allowlist')

  reads = 0; mcpCalls = 0
  afterRead = () => { preset.toolScope = { mode: 'selected', names: ['write_file'] } }
  const events: any[] = [], controller = new AbortController()
  const deps = { thread: mcpThread, provider, model: 'fixture', approvalMode: 'suggest' as const, temperature: 0, reasoningEffort: 'low', autoCompact: false, maxSteps: 10, maxWallMs: 5000, signal: controller.signal, persist() {}, requestApproval: async () => { approvals++; return 'allow' as const }, pluginRuntime: plugin, toolContext: context(mcpThread), emit: (event: any) => { events.push(event) } }
  await assert.rejects(() => runAgent({ ...deps, toolContext: { ...deps.toolContext, botToolScopeGuard: () => { throw new Error('Cortheon is enabled on this aliased model endpoint.') } } }), /Cortheon/, 'a custom API provider must not bypass its host research guard just because it lacks the TensorFold label')
  await assert.rejects(() => runAgent({ ...deps, thread: { ...mcpThread, bot: { ...mcpBot, toolScope: null as any } } }), /Invalid bot tool scope/)
  await assert.rejects(() => runAgent({ ...deps, provider: { ...provider, transport: 'claude-cli' } }), /captured API model/)
  assert.equal(requests.length, 0, 'the captured scope cannot be handed to a CLI with its own tool executor')
  await runAgent(deps)
  assert.equal(reads, 1); assert.equal(mcpCalls, 1); assert.equal(approvals, 1, 'unselected calls never request approval, selected MCP actions still do')
  assert.equal(hooks + childCalls, 0)
  assert.equal(requests.length, 7)
  for (const request of requests) {
    assert.ok((request.tools ?? []).every((tool: any) => ['read_file', 'tool_search', 'tool_describe', 'tool_call'].includes(tool.function.name)))
    assert.ok(!JSON.stringify(request).includes('UNSELECTED_MCP_METADATA_SENTINEL'))
  }
  assert.ok(events.filter((event) => event.type === 'item_add' && event.item.state === 'denied').length >= 4)

  requests.length = 0; count = 0; mode = 'race'; afterRead = () => {}
  const emptyBot = captureBotConfig({ ...preset, toolScope: { mode: 'selected', names: [] } }, [provider])
  const emptyThread = makeThread(emptyBot), beforeEmptyReads = reads
  await runAgent({ ...deps, thread: emptyThread, toolContext: context(emptyThread) })
  assert.equal(reads, beforeEmptyReads)
  assert.equal(requests.length, 2)
  assert.ok(requests.every((request) => !request.tools?.length), 'an empty imported/captured selection advertises and executes no tools')

  requests.length = 0; count = 0; mode = 'readonlyWrite'; afterRead = () => {}
  const readonlyBot = captureBotConfig({ ...preset, approvalScope: 'readonly', toolScope: { mode: 'selected', names: ['write_file'] } }, [provider])
  const readonlyThread = makeThread(readonlyBot), beforeReadonlyApprovals = approvals
  await runAgent({ ...deps, thread: readonlyThread, approvalMode: 'readonly', toolContext: context(readonlyThread) })
  assert.equal(writes, 0); assert.equal(approvals, beforeReadonlyApprovals, 'selecting a mutating tool cannot elevate read-only mode or ask for escalation')

  requests.length = 0; count = 0; mode = 'cloud'; afterRead = () => {}
  const cloud = { ...provider, kind: 'cloud' as const }
  const cloudBot = captureBotConfig({ ...preset, toolScope: { mode: 'selected', names: ['read_file'] } }, [cloud])
  const cloudThread = makeThread(cloudBot), beforeCloudReads = reads
  await runAgent({ ...deps, thread: cloudThread, provider: cloud, toolContext: context(cloudThread) })
  assert.equal(reads, beforeCloudReads, 'an external planning-only route cannot execute even an explicitly selected tool')
  assert.equal(requests.length, 0, 'the selected scope does not relax the existing cloud HTTPS/private-endpoint policy')
  assert.ok(events.some((event) => event.type === 'error' && /Cloud provider URLs must use https/.test(event.message)))
  const planned: any[] = []
  const guarded = planningGuard({ provider: cloud, messages: [{ role: 'user', content: 'Synthetic plan.' }], tools: [{ type: 'function', function: { name: 'read_file' } }] } as any, async function* (input: any) {
    assert.equal(input.tools, undefined); assert.equal(input.approvalMode, 'readonly')
    yield { type: 'tool_calls', toolCalls: [{ id: 'unexpected', function: { name: 'read_file', arguments: '{}' } }] } as any
    yield { type: 'content', text: 'A synthetic plan only.' } as any
  })!
  for await (const chunk of guarded) planned.push(chunk)
  assert.ok(!planned.some((chunk) => chunk.type === 'tool_calls'), 'the actual external planner drops unsolicited tool calls after stripping schemas')

  requests.length = 0; count = 0; mode = 'race'
  const engineProvider = { ...provider, id: 'tensorfold', source: 'tensorfold' as const }
  const engineBot = captureBotConfig({ ...preset, providerId: engineProvider.id, approvalScope: 'readonly', toolScope: { mode: 'selected', names: ['read_file'] } }, [engineProvider])
  let cortheon = false, releases = 0
  const engineThread = makeThread(engineBot)
  const guard = () => { if (cortheon) throw new Error('Cortheon hidden research is unavailable to restricted bots.') }
  const engineDeps = { ...deps, thread: engineThread, provider: engineProvider, approvalMode: 'readonly' as const, toolContext: { ...context(engineThread), botToolScopeGuard: guard }, emit: (event: any) => { events.push(event) } }
  await assert.rejects(() => runAgent({ ...engineDeps, toolContext: { ...context(engineThread), botToolScopeGuard: undefined } }), /research guard is unavailable/)
  cortheon = true
  await assert.rejects(() => runAgent(engineDeps), /Cortheon/)
  assert.equal(requests.length, 0)
  cortheon = false
  setEngineServeHook(async () => ({ baseUrl: provider.baseUrl, apiKey: '', release: () => { releases++ } } as any))
  afterRead = () => { cortheon = true }
  await runAgent(engineDeps)
  assert.equal(requests.length, 1, 'a hidden research policy race blocks the next model request before selected tool output is sent')
  assert.equal(releases, 1)
  assert.ok(events.some((event) => event.type === 'error' && /Cortheon/.test(event.message)))
  console.log('Actual bot tool scope passed: protected restart/captured history, schema/direct/deferred MCP isolation, pre-hook refusals, approvals, no delegation/plugin fallback and Cortheon model-request race.')
} finally {
  setEngineServeHook(null)
  server.closeAllConnections(); await new Promise<void>((resolve) => server.close(() => resolve()))
  if (priorHome === undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME = priorHome
  await fs.rm(profile, { recursive: true, force: true })
}
