// Run: npx tsx electron/agent/trace.test.ts
import assert from 'node:assert/strict'
import { buildRunExport } from './export'
import { recordConnectorInteraction } from './connector-spine'
import { ensureAgentEventTrace } from './trace'
import type { AgentEvent, Thread, TraceContext } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('trace tests:')

{
  const event: AgentEvent = {
    runId: 'run-1',
    threadId: 'thread-1',
    type: 'item_add',
    item: { id: 'item-1', kind: 'tool', tool: 'mcp__repo__search', title: 'MCP search', createdAt: 10 },
  }
  const traced = ensureAgentEventTrace(event, 'test')
  assert.equal(traced.type, 'item_add')
  assert.ok(traced.trace?.traceId)
  assert.equal(traced.trace?.runId, 'run-1')
  assert.equal(traced.trace?.threadId, 'thread-1')
  assert.equal(traced.trace?.itemId, 'item-1')
  assert.equal(traced.trace?.surface, 'mcp')
  assert.equal(traced.type === 'item_add' ? traced.item.trace?.traceId : null, traced.trace?.traceId)

  const tableEvent: AgentEvent = {
    runId: 'run-1',
    threadId: 'thread-1',
    type: 'item_add',
    item: { id: 'item-table', kind: 'tool', tool: 'table_summary', title: 'Table summary', createdAt: 11 },
  }
  const tableTraced = ensureAgentEventTrace(tableEvent, 'test')
  assert.equal(tableTraced.trace?.surface, 'connector')
  ok('agent item_add events receive persisted trace context')
}

{
  const approval = ensureAgentEventTrace({
    runId: 'run-2',
    threadId: 'thread-2',
    type: 'approval_request',
    callId: 'call-1',
    itemId: 'item-2',
  }, 'approval')
  assert.equal(approval.trace?.callId, 'call-1')
  assert.equal(approval.trace?.itemId, 'item-2')
  assert.equal(approval.trace?.surface, 'main')
  ok('approval events carry call and item trace context')
}

{
  const event = recordConnectorInteraction({
    connectorId: 'email',
    pluginName: 'Email',
    action: 'send_request',
    direction: 'outbound',
    target: 'client@example.com',
    input: { subject: 'Hello', body: 'Draft' },
  })
  assert.ok(event.trace?.traceId)
  assert.equal(event.trace?.surface, 'connector')
  assert.equal(event.trace?.itemId, event.id)
  assert.match(event.trace?.origin ?? '', /Email:send_request/)
  ok('connector interactions receive connector trace context')
}

{
  const trace: TraceContext = {
    traceId: 'trace-visible',
    parentTraceId: 'trace-parent',
    runId: 'run-export',
    threadId: 'thread-export',
    itemId: 'cmd',
    callId: 'call-export',
    surface: 'tool',
    origin: 'run_command',
    at: 100,
  }
  const thread: Thread = {
    id: 'thread-export',
    title: 'Trace export',
    kind: 'task',
    projectId: null,
    cwd: '/repo',
    providerId: 'openai',
    model: 'gpt',
    createdAt: 1,
    updatedAt: 2,
    status: 'idle',
    apiMessages: [],
    timeline: [
      { id: 'cmd', kind: 'tool', tool: 'run_command', command: 'npm test', output: 'ok', state: 'ok', trace, createdAt: 100 },
    ],
  }
  const bundle = buildRunExport(thread, {}, 200)
  assert.equal(bundle.timeline[0].trace?.traceId, 'trace-visible')
  assert.equal(bundle.chronicle!.entries[0].trace?.runId, 'run-export')
  assert.equal(bundle.evidence!.traceIds[0].traceId, 'trace-visible')
  assert.equal(bundle.evidence!.traceIds[0].surface, 'tool')
  const shared = buildRunExport(thread, { mode: 'share' }, 200)
  assert.notEqual(shared.evidence!.traceIds[0].traceId, 'trace-visible')
  assert.ok(shared.evidence!.traceIds[0].traceId.startsWith('anon-'))
  ok('run exports propagate and anonymize trace context')
}

console.log(`trace tests: ${passed} passed`)
