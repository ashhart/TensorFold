import assert from 'node:assert/strict'
import { McpHost, createMcpServer } from './mcp-client'

const host = new McpHost()
const config = createMcpServer({ id: 'framing-fixture', name: 'Framing fixture', command: '/bin/echo' })
const runtime = host as any
const state = runtime.getState(config)
let failures = 0

function check(name: string, run: () => void) {
  try { run(); console.log(`PASS ${name}`) }
  catch (error) { failures++; console.error(`FAIL ${name}`, error) }
}

try {
  check('A null MCP server stdout frame cannot throw into the Electron process', () => {
    assert.doesNotThrow(() => runtime.handleStdout(state, Buffer.from('null\n')))
  })
  check('MCP responses preserve UTF-8 characters across stdout chunk boundaries', () => {
    let result: unknown
    state.pending.set(71, { resolve: (value: unknown) => { result = value }, reject: () => {}, timer: setTimeout(() => {}, 1000) })
    const encoded = Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: 71, result: { text: 'Unicode fixture: 🔐' } }) + '\n')
    const split = encoded.indexOf(Buffer.from('🔐')) + 1
    runtime.handleStdout(state, encoded.subarray(0, split))
    runtime.handleStdout(state, encoded.subarray(split))
    assert.deepEqual(result, { text: 'Unicode fixture: 🔐' })
  })
  check('Invalid response envelopes cannot resolve a pending MCP request', () => {
    let resolved = false
    state.pending.set(72, { resolve: () => { resolved = true }, reject: () => {}, timer: setTimeout(() => {}, 1000) })
    runtime.handleStdout(state, Buffer.from('{"jsonrpc":"1.0","id":72,"result":{"trusted":true}}\n'))
    assert.equal(resolved, false)
  })
  check('The MCP frame byte limit applies before parsing newline-terminated input', () => {
    const frame = JSON.stringify({ jsonrpc: '2.0', id: 73, result: 'x'.repeat(8 * 1024 * 1024) }) + '\n'
    runtime.handleStdout(state, Buffer.from(frame))
    assert.equal(state.runtime.status, 'error')
    assert.match(state.runtime.lastError, /oversized|byte limit/i)
    assert.equal(state.buffer, '')
  })
} finally { await host.shutdown() }
assert.equal(failures, 0, `${failures} MCP client framing regressions failed`)
