// Run: node node_modules/jiti/lib/jiti-cli.mjs electron/agent/loop-local-reliability.test.ts
// Exercise the actual homegrown loop through an isolated, fake local SSE endpoint.
// The only executable tool is an in-memory fixture; no model or shell is run.
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import type { ServerResponse } from 'node:http'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { runAgent } from './loop'
import { PluginRuntime } from './plugin-runtime'
import type { AgentEvent, Thread } from './types'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-local-reliability-'))
await fs.chmod(root, 0o700)
let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

type ScenarioOptions = {
  delta?: (body: any, requestIndex: number) => any
  respond?: (body: any, requestIndex: number, response: ServerResponse) => void
  result?: (args: any) => { ok: boolean; output: string; changedPaths?: string[] }
  maxSteps?: number
  maxWallMs?: number
  repeatLimit?: number
  abortOnRequest?: number
  builtinRead?: boolean
  builtinWrite?: boolean
  builtinCommand?: boolean
  mcpMutation?: boolean
  abortOnDelta?: boolean
  captureUnhandled?: boolean
  freezeWallClockAfterFirstRequest?: boolean
}
const frame = (delta: any, finishReason: string | null = null) => `data: ${JSON.stringify({ choices: [{ delta, finish_reason: finishReason }] })}\n\n`
const toolDelta = (args = '{"query":"same question"}', count = 1) => ({ tool_calls: Array.from({ length: count }, (_, index) => ({ index, id: `call-${index}`, type: 'function', function: { name: 'fixture_probe', arguments: args } })) })
const namedToolDelta = (name: string, args: Record<string, unknown>) => ({ tool_calls: [{ index: 0, id: 'fixture-call', type: 'function', function: { name, arguments: JSON.stringify(args) } }] })

async function scenario(options: ScenarioOptions = {}) {
  const realNow = Date.now
  let executions = 0
  const requests: any[] = []
  const events: AgentEvent[] = []
  const unhandledRejections: unknown[] = []
  const onUnhandled = (error: unknown) => { unhandledRejections.push(error) }
  if (options.captureUnhandled) process.on('unhandledRejection', onUnhandled)
  const controller = new AbortController()
  const schema = { type: 'function', function: { name: 'fixture_probe', description: 'In-memory test probe.', parameters: { type: 'object', properties: { query: { type: 'string' }, page: { type: 'number' } }, required: ['query'] } } }
  const runtime = new PluginRuntime()
  runtime.listToolSchemaEntries = () => [{ pluginId: 'fixture', pluginName: 'Fixture', schema }]
  runtime.getRegisteredTool = (name) => name === 'fixture_probe' ? {
    pluginId: 'fixture', schema, handler: async (args: any) => { executions++; return options.result?.(args) ?? { ok: true, output: 'The result is unchanged.' } },
  } : undefined
  const server = createServer((request, response) => {
    let raw = ''
    request.on('data', (value) => { raw += value })
    request.on('end', () => {
      const body = JSON.parse(raw)
      requests.push(body)
      if (requests.length === 1 && options.freezeWallClockAfterFirstRequest) {
        const observedWallTime = realNow()
        Date.now = () => observedWallTime
      }
      response.setHeader('Content-Type', 'text/event-stream')
      if (options.abortOnRequest === requests.length) controller.abort()
      if (options.respond) { options.respond(body, requests.length, response); return }
      response.writeHead(200)
      const hasTools = body.tools?.length > 0
      const delta = options.delta?.(body, requests.length) ?? (hasTools ? toolDelta() : { content: 'Stopped repeating the same action.' })
      response.end(`${frame(delta, delta.tool_calls ? 'tool_calls' : 'stop')}data: [DONE]\n\n`)
    })
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const provider = { id: 'fixture', name: 'Fixture', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, apiKey: '', models: [] }
  const thread: Thread = { id: 'fixture', title: 'Fixture', kind: options.builtinRead || options.builtinWrite || options.builtinCommand || options.mcpMutation ? 'task' : 'chat', cwd: root, providerId: provider.id, model: 'fixture', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Check this once and tell me the result.' }], timeline: [] }
  try {
    const startedAt = Date.now()
    await runAgent({ thread, provider, model: 'fixture', approvalMode: 'auto', temperature: 0, reasoningEffort: 'low', signal: controller.signal, persist() {}, maxSteps: options.maxSteps ?? 4, maxWallMs: options.maxWallMs, guardPolicy: options.repeatLimit === undefined ? undefined : { repeatLimit: options.repeatLimit } as any, autoCompact: false, pluginRuntime: runtime,
      toolContext: {
        ...(options.builtinRead || options.builtinWrite || options.builtinCommand ? { backend: {
          kind: options.builtinCommand ? 'ssh' : 'local', rootCwd: root,
          readFile: async () => { if (!options.builtinWrite) executions++; await new Promise<void>((resolve) => setImmediate(resolve)); return 'In-memory fixture file.' },
          writeFile: async () => { executions++ },
          exec: async (command: string) => {
            if (command !== 'npm test') return { code: 3, stdout: '', stderr: '' }
            executions++
            return { code: 0, stdout: 'Fixture tests passed.', stderr: '' }
          },
        } as any } : {}),
        ...(options.mcpMutation ? {
          mcpTools: [{ serverId: 'fixture', serverName: 'Fixture', approvalMode: 'always', name: 'mutate', callName: 'mcp__fixture__mutate', inputSchema: { type: 'object', properties: { item: { type: 'string' } } }, destructiveHint: true } as any],
          callMcpTool: async () => { executions++; return { ok: true, output: 'Fixture MCP mutation completed.' } },
        } : {}),
      },
      requestApproval: async () => 'allow', emit(event) {
      events.push(event)
      if (options.abortOnDelta && event.type === 'item_delta') controller.abort()
      if (event.type === 'item_add') thread.timeline.push(event.item)
      if (event.type === 'item_update') Object.assign(thread.timeline.find((item) => item.id === event.id) ?? {}, event.patch)
      if (event.type === 'item_delta') {
        const item = thread.timeline.find((item) => item.id === event.id)
        if (item) item.text = (item.text ?? '') + event.chunk
      }
    } })
    if (options.captureUnhandled) await new Promise<void>((resolve) => setTimeout(resolve, 20))
    return { executions, requests, events, thread, elapsedMs: Date.now() - startedAt, unhandledRejections }
  } finally {
    Date.now = realNow
    controller.abort()
    server.closeAllConnections()
    await new Promise<void>((resolve) => server.close(() => resolve()))
    if (options.captureUnhandled) process.removeListener('unhandledRejection', onUnhandled)
  }
}

const failures: Error[] = []
async function check(name: string, test: () => Promise<void>) {
  try { await test(); ok(name) } catch (error) { failures.push(error as Error); console.error(`  FAIL ${name}: ${(error as Error).message}`) }
}
try {
  await check('ignored repeated-tool warnings stop before another execution and close without tools', async () => {
    const result = await scenario()
    assert.equal(result.executions, 3, `identical successful tool executed ${result.executions} times`)
    assert.ok(result.thread.timeline.some((item) => item.kind === 'notice' && /repeat|repeating/i.test(item.text ?? '')), 'the user sees why repetition stopped')
    assert.equal(result.requests.at(-1).tools, undefined, 'the closing summary cannot call tools')
    assert.ok(result.events.some((event) => event.type === 'done'))
  })
  await check('argument whitespace and key order cannot evade the exact-action cap', async () => {
    const result = await scenario({ delta: (body, n) => body.tools?.length ? toolDelta(n % 2 ? '{"query":"same question","page":1}' : '{ "page": 1, "query": "same question" }') : { content: 'Done.' } })
    assert.equal(result.executions, 3)
  })
  await check('duplicate requests inside one tool batch cannot exceed the cap', async () => {
    const result = await scenario({ maxSteps: 1, delta: (body) => body.tools?.length ? toolDelta(undefined, 5) : { content: 'Done.' } })
    assert.equal(result.executions, 3)
    const assistant = result.thread.apiMessages.find((message) => message.tool_calls?.length)
    assert.equal(result.thread.apiMessages.filter((message) => message.role === 'tool').length, assistant?.tool_calls?.length, 'all blocked calls retain a protocol result')
  })
  await check('an opaque failed tool pauses before uncertain side effects can be repeated', async () => {
    const result = await scenario({ result: () => ({ ok: false, output: 'Fixture error.' }) })
    assert.equal(result.executions, 1)
  })
  await check('parallel builtin read batches reserve attempts before execution awaits', async () => {
    const result = await scenario({ maxSteps: 1, builtinRead: true, delta: (body) => body.tools?.length ? { tool_calls: Array.from({ length: 5 }, (_, index) => ({ index, id: `read-${index}`, type: 'function', function: { name: 'read_file', arguments: '{"path":"fixture.txt"}' } })) } : { content: 'Done.' } })
    assert.equal(result.executions, 3)
    assert.equal(result.thread.apiMessages.filter((message) => message.role === 'tool').length, 5)
  })
  await check('different useful arguments remain executable', async () => {
    const result = await scenario({ maxSteps: 6, delta: (body, n) => body.tools?.length ? toolDelta(JSON.stringify({ query: `question ${n}` })) : { content: 'Done.' }, result: (args) => ({ ok: true, output: `Result for ${args.query}` }) })
    assert.equal(result.executions, 6)
  })
  await check('the configured repeat limit is respected', async () => {
    const result = await scenario({ repeatLimit: 2 })
    assert.equal(result.executions, 2)
  })
  await check('a real workspace change allows useful re-checks without resetting its own action', async () => {
    const result = await scenario({ maxSteps: 7, builtinRead: true, delta: (body, n) => body.tools?.length ? n === 4 ? toolDelta('{"query":"edit the workspace"}') : namedToolDelta('read_file', { path: 'fixture.txt' }) : { content: 'Done.' }, result: () => ({ ok: true, output: 'Edited fixture.', changedPaths: ['fixture.txt'] }) })
    assert.equal(result.executions, 7)
    const writeLoop = await scenario({ result: () => ({ ok: true, output: 'Changed fixture.', changedPaths: ['fixture.txt'] }) })
    assert.equal(writeLoop.executions, 3)
  })
  await check('alternating mutating signatures retain their lifetime repeat counts', async () => {
    const result = await scenario({ maxSteps: 8, builtinWrite: true, delta: (body, n) => body.tools?.length ? namedToolDelta('write_file', { path: n % 2 ? 'fixture-a.txt' : 'fixture-b.txt', content: 'Same content.' }) : { content: 'Done.' } })
    assert.equal(result.executions, 6, `alternating writes executed ${result.executions} times`)
  })
  await check('workspace edits do not reset potentially mutating MCP calls', async () => {
    const result = await scenario({ maxSteps: 8, mcpMutation: true, delta: (body, n) => body.tools?.length ? n % 2 ? namedToolDelta('mcp__fixture__mutate', { item: 'same item' }) : toolDelta(JSON.stringify({ query: `edit ${n}` })) : { content: 'Done.' }, result: () => ({ ok: true, output: 'Edited fixture.', changedPaths: ['fixture.txt'] }) })
    assert.equal(result.executions, 6, `MCP mutation and edit cycle executed ${result.executions} times`)
  })
  await check('test commands can re-check after actual workspace changes', async () => {
    const result = await scenario({ maxSteps: 7, builtinCommand: true, delta: (body, n) => body.tools?.length ? n === 4 ? toolDelta('{"query":"edit the workspace"}') : namedToolDelta('run_command', { command: 'npm test' }) : { content: 'Done.' }, result: () => ({ ok: true, output: 'Edited fixture.', changedPaths: ['fixture.txt'] }) })
    assert.equal(result.executions, 7)
  })
  await check('meaningfully changing write arguments remain executable', async () => {
    const result = await scenario({ maxSteps: 6, builtinWrite: true, delta: (body, n) => body.tools?.length ? namedToolDelta('write_file', { path: 'fixture.txt', content: `Version ${n}.` }) : { content: 'Done.' } })
    assert.equal(result.executions, 6)
  })
  await check('repetitive budget-wrap-up content is cut off too', async () => {
    const repeated = 'I will check the same result carefully.\n'.repeat(80)
    const result = await scenario({ maxSteps: 1, delta: (body) => body.tools?.length ? toolDelta() : { content: repeated } })
    const last = result.thread.timeline.filter((item) => item.kind === 'assistant').at(-1)
    assert.ok((last?.text?.length ?? 0) <= 1200, `wrap-up retained ${last?.text?.length} repeating characters`)
    assert.match(last?.meta ?? '', /repeat/i)
  })
  await check('normal content and reasoning repetition both stop after bounded recovery', async () => {
    for (const field of ['content', 'reasoning_content']) {
      const result = await scenario({ maxSteps: 6, delta: () => ({ [field]: 'I will check the same result carefully.\n'.repeat(80) }) })
      assert.equal(result.requests.length, 2)
      assert.equal(result.executions, 0)
      assert.ok(result.thread.timeline.some((item) => item.kind === 'notice' && /kept repeating/i.test(item.text ?? '')))
      assert.ok(result.thread.timeline.filter((item) => item.kind === 'assistant' || item.kind === 'reasoning').every((item) => (item.text?.length ?? 0) <= 1200))
    }
  })
  await check('a wall deadline interrupts transient retry backoff instead of restarting the same request', async () => {
    const result = await scenario({ maxWallMs: 60, respond: (body, n, response) => {
      if (n === 1) { response.writeHead(503, { 'Content-Type': 'application/json' }); response.end('{"error":{"message":"overloaded"}}'); return }
      assert.equal(body.tools, undefined)
      response.end(`${frame({ content: 'Provider is unavailable; try again.' }, 'stop')}data: [DONE]\n\n`)
    } })
    assert.ok(result.elapsedMs < 350, `retry backoff exceeded the soft budget: ${result.elapsedMs}ms`)
    assert.equal(result.requests.length, 2)
    assert.equal(result.executions, 0)
  })
  await check('an expired retry signal closes without tools even if the wall clock moves back', async () => {
    const result = await scenario({ maxWallMs: 60, freezeWallClockAfterFirstRequest: true, respond: (_body, n, response) => {
      if (n === 1) { response.writeHead(503, { 'Content-Type': 'application/json' }); response.end('{"error":{"message":"overloaded"}}'); return }
      response.end(`${frame({ content: 'The provider is unavailable; review and continue later.' }, 'stop')}data: [DONE]\n\n`)
    } })
    assert.equal(result.requests.length, 2)
    assert.equal(result.requests[1].tools, undefined, 'monotonic deadline expiry cannot reopen tool dispatch when Date.now is behind')
    assert.equal(result.executions, 0)
  })
  await check('a wall budget interrupts a stalled stream and bounds the tool-free summary', async () => {
    const result = await scenario({ maxWallMs: 60, respond: (_body, _n, response) => {
      const timeout = setTimeout(() => response.end(`${frame({ content: 'Late response.' }, 'stop')}data: [DONE]\n\n`), 500)
      response.once('close', () => clearTimeout(timeout))
    } })
    assert.ok(result.elapsedMs < 350, `60ms soft budget took ${result.elapsedMs}ms`)
    assert.equal(result.requests.at(-1).tools, undefined)
    assert.equal(result.executions, 0)
    assert.ok(result.events.some((event) => event.type === 'done'))
    assert.equal(result.events.some((event) => event.type === 'error'), false)
  })
  await check('a final-summary deadline during retry backoff produces a stopped notice and clean completion', async () => {
    const result = await scenario({ maxSteps: 1, maxWallMs: 60, respond: (body, _n, response) => {
      if (body.tools?.length) { response.end(`${frame(toolDelta(), 'tool_calls')}data: [DONE]\n\n`); return }
      response.writeHead(503, { 'Content-Type': 'application/json' })
      response.end('{"error":{"message":"overloaded"}}')
    } })
    assert.ok(result.elapsedMs < 350)
    assert.equal(result.requests.length, 2, 'a timed-out summary must not start another retry')
    assert.ok(result.thread.timeline.some((item) => item.kind === 'notice' && /did not finish its final summary in time/.test(item.text ?? '')))
    assert.ok(result.events.some((event) => event.type === 'done'))
    assert.equal(result.events.some((event) => event.type === 'error'), false)
  })
  await check('user cancellation starts no recovery request or tool', async () => {
    const result = await scenario({ abortOnRequest: 1 })
    assert.equal(result.requests.length, 1)
    assert.equal(result.executions, 0)
    assert.ok(result.events.some((event) => event.type === 'done'))
  })
  await check('Stop during an ongoing HTTP SSE response has no unhandled abort rejection', async () => {
    let socketClosed = false
    const result = await scenario({ abortOnDelta: true, captureUnhandled: true, respond: (_body, _n, response) => {
      response.write(frame({ content: '1\n' }))
      const timer = setTimeout(() => response.end(`${frame({ content: '2\n' }, 'stop')}data: [DONE]\n\n`), 500)
      response.once('close', () => { socketClosed = true; clearTimeout(timer) })
    } })
    assert.deepEqual(result.unhandledRejections.map((error: any) => ({ name: error?.name, message: error?.message })), [], 'Stop must observe every rejected reader/transport promise')
    assert.equal(result.requests.length, 1)
    assert.equal(result.executions, 0)
    assert.ok(result.events.some((event) => event.type === 'done'))
    assert.equal(result.events.some((event) => event.type === 'error'), false)
    assert.equal(socketClosed, true)
    assert.ok(result.elapsedMs < 350, `Stop waited for the server to finish: ${result.elapsedMs}ms`)
  })
} finally {
  await fs.rm(root, { recursive: true, force: true })
}
console.log(`\n${passed} integrated local-loop checks passed.`)
if (failures.length) throw new AggregateError(failures, `${failures.length} integrated local-loop checks failed`)
