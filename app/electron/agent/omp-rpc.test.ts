// Run: jiti electron/agent/omp-rpc.test.ts
// Exercises the OMP RPC transport against a fixture child process that speaks the
// protocol exactly as the installed omp does: ready frame, id-correlated responses,
// rpc_chunk-split large payloads, streamed session events.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { OmpRpcClient, reassembleRpcChunks } from './omp-rpc'

// --- pure reassembly ---
{
  const payload = JSON.stringify({ type: 'response', id: 'x', command: 'get_state', success: true, data: { sessionId: 'sess-1', text: 'a'.repeat(64) } })
  const size = Math.ceil(payload.length / 3)
  const chunks = [0, 1, 2].map((index) => ({ chunkId: 'c1', index, count: 3, data: payload.slice(index * size, (index + 1) * size) }))
  const { complete, pending } = reassembleRpcChunks(chunks)
  assert.equal(pending, 0)
  assert.deepEqual((complete[0].frame as { data: { sessionId: string } }).data.sessionId, 'sess-1')
  const partial = reassembleRpcChunks(chunks.slice(0, 2))
  assert.equal(partial.pending, 1)
  assert.equal(partial.complete.length, 0)
  console.log('  ✓ rpc_chunk reassembly joins ordered parts and reports pending groups')
}

// --- fixture child that speaks the omp rpc protocol ---
const root = mkdtempSync(path.join(os.tmpdir(), 'vontra-omp-rpc-'))
const fixture = path.join(root, 'fake-omp.mjs')
writeFileSync(fixture, `
import { setTimeout as delay } from 'node:timers/promises'
const write = (frame) => process.stdout.write(JSON.stringify(frame) + '\\n')
write({ type: 'ready', protocolVersion: 1, supportedProtocolVersions: [1, 2], maxFrameBytes: 1048576, maxReassembledFrameBytes: 67108864 })
const bigState = JSON.stringify({ type: 'response', id: 'PENDING', command: 'get_state', success: true, data: { sessionId: 'sess-42', sessionFile: '/tmp/sess-42.jsonl', todoPhases: [{ name: 'Plan', tasks: [{ content: 'Do the thing', status: 'in_progress' }] }] } })
let buffer = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', (chunk) => {
  buffer += chunk
  let nl = buffer.indexOf('\\n')
  while (nl >= 0) {
    const line = buffer.slice(0, nl).trim()
    buffer = buffer.slice(nl + 1)
    if (line) handle(JSON.parse(line))
    nl = buffer.indexOf('\\n')
  }
})
async function handle(command) {
  if (command.type === 'negotiate_protocol') {
    write({ type: 'response', id: command.id, command: 'negotiate_protocol', success: true, data: { protocolVersion: 1 } })
  } else if (command.type === 'set_subagent_subscription' || command.type === 'set_todos') {
    write({ type: 'response', id: command.id, command: command.type, success: true, data: {} })
  } else if (command.type === 'get_state') {
    // Large payload delivered as rpc_chunk split — exercises client reassembly.
    const frame = bigState.replace('PENDING', command.id)
    const third = Math.ceil(frame.length / 3)
    for (let index = 0; index < 3; index++) {
      write({ type: 'rpc_chunk', chunkId: command.id + '-chunk', index, count: 3, byteLength: frame.length, data: frame.slice(index * third, (index + 1) * third) })
    }
  } else if (command.type === 'get_subagents') {
    write({ type: 'response', id: command.id, command: 'get_subagents', success: true, data: { subagents: [{ id: 'sa-7', agent: 'scout', status: 'completed', task: 'research deps', index: 0 }] } })
  } else if (command.type === 'prompt') {
    write({ type: 'prompt_result', id: command.id, agentInvoked: true })
    write({ type: 'agent_start' })
    write({ type: 'message_start', message: { role: 'assistant', content: [] } })
    write({ type: 'message_update', message: { role: 'assistant' }, assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'Hello ' } })
    write({ type: 'message_update', message: { role: 'assistant' }, assistantMessageEvent: { type: 'text_delta', contentIndex: 0, delta: 'OMP.' } })
    write({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: 'Hello OMP.' }] } })
    write({ type: 'agent_end', messages: [], isTerminal: true })
  } else if (command.type === 'steer') {
    write({ type: 'notice', level: 'info', message: 'steered: ' + command.message })
  }
}
await delay(60000)
`, 'utf8')

const child = spawn(process.execPath, [fixture], { stdio: ['pipe', 'pipe', 'pipe'] })
const client = new OmpRpcClient({ cwd: root, child, silenceTimeoutMs: 10_000 })
const events: string[] = []
client.onEvent((event) => { if (event.type !== 'ready') events.push(event.type) })

const ready = await client.ready
assert.equal(ready.protocolVersion, 1)
assert.deepEqual(ready.supportedProtocolVersions, [1, 2])
const negotiate = await client.request({ type: 'negotiate_protocol', protocolVersion: 1 })
assert.equal(negotiate.success, true)
const state = await client.request<{ data?: { sessionId?: string; todoPhases?: unknown[] } }>({ type: 'get_state' })
assert.equal(state.data?.sessionId, 'sess-42', 'chunked get_state response reassembles into its correlated request')
assert.equal(Array.isArray(state.data?.todoPhases), true)

client.send({ type: 'prompt', message: 'do the work' })
await new Promise((resolve) => setTimeout(resolve, 600))
assert.deepEqual(events, ['prompt_result', 'agent_start', 'message_start', 'message_update', 'message_update', 'message_end', 'agent_end'], 'session events stream in protocol order (prompt_result included)')

const subagents = await client.request<{ data?: { subagents?: Array<{ id: string }> } }>({ type: 'get_subagents' })
assert.equal(subagents.data?.subagents?.[0]?.id, 'sa-7')
client.dispose()
await client.exited
console.log('  ✓ transport correlates commands, reassembles chunks, streams events, and exits cleanly')

console.log('omp rpc transport tests passed')
