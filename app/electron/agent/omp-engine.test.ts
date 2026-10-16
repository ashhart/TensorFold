// Run: jiti electron/agent/omp-engine.test.ts
// Pure mapping tests plus a full runOmpThread pass against a duck-typed fake client.
import assert from 'node:assert/strict'
import type { AgentEvent, Thread } from './types'
import type { OmpRpcClient } from './omp-rpc'
import { mapOmpEvent, newOmpMappingState, ompAgentStatusToSubAgentStatus, ompMessageText, ompPhasesToPlan, ompSpawnArgs, ompSubagentPatch, planToOmpPhases, runOmpThread, isSafeOmpSessionPath, type OmpTodoPhase } from './omp-engine'

console.log('omp engine tests:')

// --- spawn args: attacker-controlled resume paths and thinking levels are refused ---
{
  const os = await import('node:os')
  const home = os.homedir()
  assert.deepEqual(
    ompSpawnArgs({ cwd: '/repo', resumePath: `${home}/.vontra/omp-sessions/s1.jsonl`, bridged: true, approvalMode: 'suggest', reasoningEffort: 'high' }),
    ['--cwd', '/repo', '--resume', `${home}/.vontra/omp-sessions/s1.jsonl`, '--tools', 'read,grep,glob,ast_grep,task,todo,web_search,hub', '--thinking', 'high'],
    'a home-dir session file resumes, bridged runs restrict the engine toolset',
  )
  // Traversal and relative paths never reach --resume; unknown thinking levels are dropped.
  const poisoned = ompSpawnArgs({ cwd: '/repo', resumePath: '../../../etc/passwd', bridged: false, approvalMode: 'suggest', reasoningEffort: 'ultrathink --evil' })
  assert.ok(!poisoned.includes('--resume'), 'traversal paths are refused as resume targets')
  assert.ok(!poisoned.includes('--thinking'), 'non-omp thinking levels are dropped, not forwarded')
  assert.equal(isSafeOmpSessionPath('/tmp/s.jsonl'), true)
  assert.equal(isSafeOmpSessionPath('relative.jsonl'), false)
  assert.equal(isSafeOmpSessionPath(`${home}/../etc/passwd`), false, 'traversal inside a home-prefixed path is refused')
  console.log('  ✓ spawn args validate resume paths and thinking levels')
}

// --- todo phase <-> plan round trips ---
{
  const phases: OmpTodoPhase[] = [{
    name: 'Plan',
    tasks: [
      { content: 'Trace the replay path', status: 'completed' },
      { content: 'Reproduce the duplicate charge', status: 'in_progress' },
      { content: 'Guard with an idempotency key', status: 'blocked', blocker: 'key rotation' },
      { content: 'Port the old retry shim', status: 'abandoned' },
    ],
  }]
  const plan = ompPhasesToPlan(phases)
  assert.deepEqual(plan.map((s) => [s.text, s.done, s.status]), [
    ['Trace the replay path', true, 'done'],
    ['Reproduce the duplicate charge', false, 'in_progress'],
    ['Guard with an idempotency key', false, 'blocked'],
    ['Port the old retry shim', false, 'dropped'],
  ])
  assert.equal(plan[2].blocker, 'key rotation')
  const back = planToOmpPhases(plan)
  assert.equal(back.length, 1)
  assert.equal(back[0].name, 'Plan')
  assert.deepEqual(back[0].tasks.map((t) => t.status), ['completed', 'in_progress', 'blocked', 'abandoned'])
  assert.equal(back[0].tasks[2].blocker, 'key rotation')

  // Multi-phase lists nest tasks under an aggregate step.
  const nested = ompPhasesToPlan([
    { name: 'Research', tasks: [{ content: 'Read the docs', status: 'completed' }] },
    { name: 'Build', tasks: [{ content: 'Write the handler', status: 'in_progress' }] },
  ])
  assert.equal(nested.length, 4)
  assert.deepEqual(nested.map((s) => s.depth ?? 0), [0, 1, 0, 1])
  assert.equal(nested[0].text, 'Research (1/1)')
  assert.equal(nested[0].status, 'done')
  assert.equal(nested[2].status, 'in_progress')

  // TensorFold Studio subtasks fold into the parent task content on the omp side.
  const folded = planToOmpPhases([
    { id: 's0', text: 'Fix the bug', done: false },
    { id: 's1', text: 'add regression test', done: false, depth: 1 },
  ])
  assert.equal(folded[0].tasks.length, 1)
  assert.match(folded[0].tasks[0].content, /Fix the bug · add regression test/)
  console.log('  ✓ todo phases and plan steps round-trip with statuses, blockers, and nesting')
}

// --- status + text helpers ---
{
  assert.equal(ompAgentStatusToSubAgentStatus('completed'), 'succeeded')
  assert.equal(ompAgentStatusToSubAgentStatus('aborted'), 'cancelled')
  assert.equal(ompAgentStatusToSubAgentStatus('pending'), 'queued')
  assert.equal(ompAgentStatusToSubAgentStatus('weird'), 'running')
  assert.equal(ompMessageText({ role: 'assistant', content: [{ type: 'thinking', thinking: 'hidden' }, { type: 'text', text: 'Visible ' }, { type: 'text', text: 'answer.' }] }), 'Visible answer.')
  const patch = ompSubagentPatch({ id: 'sa-9', agent: 'reviewer', status: 'failed', task: 'review the diff' })
  assert.deepEqual([patch.id, patch.name, patch.status], ['sa-9', 'reviewer', 'failed'])
  console.log('  ✓ agent status mapping, message text extraction, and subagent projection')
}

// --- event -> timeline mapping ---
{
  const state = newOmpMappingState()
  const collect = () => {
    const out: AgentEvent[] = []
    const run = (event: Record<string, unknown>) => {
      for (const e of mapOmpEvent(event as never, state).events) out.push({ ...e, runId: 'r', threadId: 't' } as AgentEvent)
    }
    return { out, run }
  }
  const { out, run } = collect()
  run({ type: 'agent_start' })
  run({ type: 'message_start', message: { role: 'assistant', content: [] } })
  run({ type: 'message_update', message: { role: 'assistant' }, assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'Working ' } })
  run({ type: 'message_update', message: { role: 'assistant' }, assistantMessageEvent: { type: 'thinking_delta', contentIndex: 0, delta: 'hidden' } })
  run({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'Working done.' }] } })
  run({ type: 'tool_execution_start', toolCallId: 'tc-1', toolName: 'read', args: {} })
  run({ type: 'tool_execution_end', toolCallId: 'tc-1', toolName: 'read', result: { path: '/x', text: 'contents' } })
  run({ type: 'todo_reminder', todos: [{ content: 'a' }], attempt: 1, maxAttempts: 3 })
  run({ type: 'notice', level: 'error', message: 'engine hiccup' })
  run({ type: 'agent_end', messages: [] })

  assert.equal(out.filter((e) => e.type === 'item_add').length, 4, 'assistant + tool + reminder notice + error notice')
  const assistant = out.find((e) => e.type === 'item_add' && e.item.kind === 'assistant')
  assert.ok(assistant && assistant.type === 'item_add')
  const deltas = out.filter((e) => e.type === 'item_delta')
  assert.equal(deltas.length, 1, 'thinking deltas are not mirrored to the timeline')
  const finalText = out.find((e) => e.type === 'item_update')
  assert.ok(finalText && finalText.type === 'item_update' && finalText.patch.text === 'Working done.')
  const toolAdd = out.find((e) => e.type === 'item_add' && e.item.kind === 'tool')
  assert.ok(toolAdd && toolAdd.type === 'item_add' && toolAdd.item.state === 'running')
  const toolUpdate = out.filter((e) => e.type === 'item_update')[1]
  assert.ok(toolUpdate && toolUpdate.type === 'item_update' && toolUpdate.patch.state === 'ok')
  assert.match(String(toolUpdate.patch.output), /Untrusted tool output/, 'engine tool output is framed as untrusted data')
  const errorItem = out.find((e) => e.type === 'item_add' && e.item.kind === 'error')
  assert.ok(errorItem && errorItem.type === 'item_add' && errorItem.item.text === 'engine hiccup')
  assert.ok(out.some((e) => e.type === 'done'))
  assert.equal(state.finished, true)
  console.log('  ✓ session events map onto the TensorFold Studio timeline with untrusted-output framing')
}

// --- full run against a fake client ---
{
  const thread: Thread = {
    id: 't1', title: 'OMP thread', kind: 'task', cwd: '/repo', providerId: null, model: null,
    createdAt: 1, updatedAt: 1, status: 'running', apiMessages: [], timeline: [],
    plan: [{ id: 's0', text: 'Ship the fix', done: false, status: 'in_progress' }],
  }
  const sent: Array<Record<string, unknown>> = []
  const listeners = new Set<(event: Record<string, unknown>) => void>()
  const emitAgentEnd = () => { for (const l of [...listeners]) l({ type: 'agent_end', messages: [] }) }
  const fakeClient = {
    ready: Promise.resolve({ protocolVersion: 1, supportedProtocolVersions: [1, 2], maxFrameBytes: 1_048_576, maxReassembledFrameBytes: 8_388_608 }),
    exited: new Promise<void>(() => {}),
    onEvent: (listener: (event: Record<string, unknown>) => void) => { listeners.add(listener); return () => listeners.delete(listener) },
    send: (command: Record<string, unknown>) => {
      sent.push(command)
      if (command.type === 'prompt') {
        Promise.resolve().then(() => {
          for (const l of [...listeners]) l({ type: 'agent_start' })
          emitAgentEnd()
        })
      }
    },
    request: async (command: Record<string, unknown>) => {
      sent.push(command)
      if (command.type === 'get_state') {
        return { command: 'get_state', success: true, data: { sessionId: 'sess-9', sessionFile: '/tmp/sess-9.jsonl', todoPhases: [{ name: 'Plan', tasks: [{ content: 'Ship the fix', status: 'completed' }] }] } }
      }
      if (command.type === 'get_subagents') {
        return { command: 'get_subagents', success: true, data: { subagents: [{ id: 'sa-3', agent: 'scout', status: 'completed', task: 'scan deps' }] } }
      }
      return { command: String(command.type), success: true, data: {} }
    },
    dispose: () => {},
  } as unknown as OmpRpcClient

  const emitted: AgentEvent[] = []
  let persisted = 0
  let sessionCursor: { sessionId: string; sessionPath?: string } | null = null
  const subagents: Array<{ id: string; status: string }> = []
  await runOmpThread({
    thread,
    runId: 'r1',
    emit: (event) => emitted.push(event),
    signal: new AbortController().signal,
    persist: () => { persisted++ },
    userText: 'ship it',
    client: fakeClient,
    onSession: (cursor) => { sessionCursor = cursor },
    updateSubAgent: (record) => { subagents.push({ id: record.id, status: record.status }) },
  })

  assert.ok(sent.some((c) => c.type === 'prompt' && c.message === 'ship it'))
  assert.ok(sent.some((c) => c.type === 'set_todos'), 'existing plan is pushed into the engine')
  assert.ok(sent.some((c) => c.type === 'set_subagent_subscription' && c.level === 'events'))
  assert.ok(thread.plan && thread.plan[0].done && thread.plan[0].status === 'done', 'final todo phases replace the thread plan')
  assert.deepEqual(sessionCursor, { sessionId: 'sess-9', sessionPath: '/tmp/sess-9.jsonl' })
  assert.deepEqual(subagents, [{ id: 'sa-3', status: 'succeeded' }])
  assert.ok(emitted.some((e) => e.type === 'plan' && e.plan?.[0]?.done))
  assert.ok(emitted.some((e) => e.type === 'done'))
  assert.ok(persisted >= 1)
  console.log('  ✓ runOmpThread prompts, pushes todos, mirrors events, and projects subagents')
}

// --- bridged run: host tool call round-trip + live subagent lifecycle ---
{
  const thread: Thread = {
    id: 't2', title: 'Bridged OMP thread', kind: 'task', cwd: '/repo', providerId: null, model: null,
    createdAt: 1, updatedAt: 1, status: 'running', apiMessages: [], timeline: [],
  }
  const sent: Array<Record<string, unknown>> = []
  const listeners = new Set<(event: Record<string, unknown>) => void>()
  const fakeClient = {
    ready: Promise.resolve({ protocolVersion: 1, supportedProtocolVersions: [1, 2], maxFrameBytes: 1_048_576, maxReassembledFrameBytes: 8_388_608 }),
    exited: new Promise<void>(() => {}),
    onEvent: (listener: (event: Record<string, unknown>) => void) => { listeners.add(listener); return () => listeners.delete(listener) },
    send: (command: Record<string, unknown>) => {
      sent.push(command)
      if (command.type === 'prompt') {
        Promise.resolve().then(() => {
          for (const l of [...listeners]) l({ type: 'agent_start' })
          for (const l of [...listeners]) l({ type: 'tool_execution_start', toolCallId: 'tc-9', toolName: 'write_file', args: {} })
          for (const l of [...listeners]) l({ type: 'host_tool_call', id: 'hc-9', toolCallId: 'tc-9', toolName: 'write_file', arguments: { path: 'a.txt', content: 'hi' } })
          for (const l of [...listeners]) l({ type: 'host_tool_call', id: 'hc-edit', toolCallId: 'tc-edit', toolName: 'edit_file', arguments: { path: 'a.txt', search: 'hi', replace: 'hello' } })
          for (const l of [...listeners]) l({ type: 'host_tool_call', id: 'hc-evil', toolCallId: 'tc-evil', toolName: 'change_setting', arguments: { key: 'approvalMode', value: 'auto' } })
          for (const l of [...listeners]) l({ type: 'subagent_lifecycle', payload: { id: 'sa-live', agent: 'scout', status: 'started', task: 'scan', index: 0 } })
        })
        // The run finishes once the host tool result has been written back.
        setTimeout(() => { for (const l of [...listeners]) l({ type: 'agent_end', messages: [] }) }, 150)
      }
    },
    request: async (command: Record<string, unknown>) => {
      sent.push(command)
      if (command.type === 'get_state') return { command: 'get_state', success: true, data: { sessionId: 'sess-10' } }
      if (command.type === 'get_subagents') return { command: 'get_subagents', success: true, data: { subagents: [] } }
      return { command: String(command.type), success: true, data: {} }
    },
    dispose: () => {},
  } as unknown as OmpRpcClient

  const emitted: AgentEvent[] = []
  const executed: Array<{ name: string; args: Record<string, unknown> }> = []
  const approvals: string[] = []
  const subagents: Array<{ id: string; status: string }> = []
  await runOmpThread({
    thread,
    runId: 'r2',
    emit: (event) => emitted.push(event),
    signal: new AbortController().signal,
    persist: () => {},
    userText: 'write the file',
    approvalMode: 'suggest',
    client: fakeClient,
    updateSubAgent: (record) => { subagents.push({ id: record.id, status: record.status }) },
    requestApproval: async (callId) => {
      approvals.push(callId)
      const approvalEvent = emitted.filter((event) => event.type === 'approval_request').at(-1)
      assert.ok(approvalEvent && approvalEvent.type === 'approval_request')
      const initial = emitted.find((event) => event.type === 'item_add' && event.item.id === approvalEvent.itemId)
      assert.ok(initial && initial.type === 'item_add', 'even a host call without tool_execution_start has a timeline item')
      const item = { ...initial.item }
      for (const event of emitted) if (event.type === 'item_update' && event.id === item.id) Object.assign(item, event.patch)
      const expectedArgs = callId.includes('hc-edit') ? { path: 'a.txt', search: 'hi', replace: 'hello' } : { path: 'a.txt', content: 'hi' }
      assert.deepEqual(item.args, expectedArgs, 'the host approval resolver sees actual arguments before it runs')
      assert.equal(item.path, 'a.txt')
      return 'allow'
    },
    executeTool: async (name, args) => { executed.push({ name, args }); return { ok: true, output: 'wrote a.txt' } },
  })

  assert.ok(sent.some((c) => c.type === 'set_host_tools' && Array.isArray((c as { tools?: unknown[] }).tools) && (c as { tools: Array<{ name: string }> }).tools.some((t) => t.name === 'run_command')), 'mutating TensorFold Studio tools are registered as host tools')
  assert.ok(sent.some((c) => c.type === 'set_subagent_subscription' && c.level === 'events'))
  assert.equal(approvals.length, 2, 'suggest mode routes both engine mutations through the approval gate')
  assert.match(approvals[0], /hc-9/, 'approval callId derives from the host tool call frame')
  assert.deepEqual(executed, [
    { name: 'write_file', args: { path: 'a.txt', content: 'hi' } },
    { name: 'edit_file', args: { path: 'a.txt', search: 'hi', replace: 'hello' } },
  ], 'only registered host tools execute')
  const refused = sent.find((c) => c.type === 'host_tool_result' && (c as { id: string }).id === 'hc-evil') as unknown as { result: { content: Array<{ text: string }> }; isError?: boolean } | undefined
  assert.ok(refused, 'unregistered host tool calls get a result frame')
  assert.equal(refused.isError, true)
  assert.match(refused.result.content[0].text, /not a registered host tool/)
  const resultFrame = sent.find((c) => c.type === 'host_tool_result' && (c as { id: string }).id === 'hc-9') as unknown as { id: string; result: { content: Array<{ type: string; text: string }> } }
  assert.ok(resultFrame, 'host_tool_result frame is written back to the engine')
  assert.equal(resultFrame.id, 'hc-9')
  assert.equal(resultFrame.result.content[0].type, 'text')
  assert.equal(resultFrame.result.content[0].text, 'wrote a.txt')
  assert.ok(emitted.some((e) => e.type === 'approval_request'), 'approval_request reaches the renderer timeline')
  assert.ok(emitted.some((e) => e.type === 'item_update' && (e as { patch?: { approval?: string } }).patch?.approval), 'approval explanation patches the tool item')
  assert.deepEqual(subagents, [{ id: 'sa-live', status: 'running' }], 'subagent lifecycle frames update the tree live')
  console.log('  ✓ bridged runs gate engine mutations through TensorFold Studio approvals end-to-end')
}

console.log('omp engine tests passed')
