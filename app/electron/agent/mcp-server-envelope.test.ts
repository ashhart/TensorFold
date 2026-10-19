import assert from 'node:assert/strict'
import { PassThrough } from 'node:stream'
import { handleMcpRequest, runMcpServerStdio, type McpServerDeps } from './mcp-server'

const deps: McpServerDeps = { tools: [], callTool: async () => { throw new Error('Unexpected tool dispatch') } }
let failures = 0
for (const value of [null, [], true, 2, 'debug', { id: 1, method: 'ping' }, { jsonrpc: '2.0', id: {}, method: 'ping' }]) {
  try {
    const response = await handleMcpRequest(deps, value as any)
    assert.equal(response?.error?.code, -32600, 'Malformed envelopes return JSON-RPC Invalid Request')
    console.log(`PASS invalid envelope ${JSON.stringify(value)}`)
  } catch (error) { failures++; console.error(`FAIL invalid envelope ${JSON.stringify(value)}`, error) }
}
assert.equal(failures, 0, `${failures} MCP server envelope regressions failed`)

const input = new PassThrough()
const output = new PassThrough()
let wire = ''
output.on('data', (chunk: Buffer) => { wire += chunk.toString('utf8') })
runMcpServerStdio(deps, input, output)
input.end('null\n{broken-json\n{"jsonrpc":"2.0","id":9,"method":"ping"}\n')
await new Promise<void>((resolve) => setImmediate(resolve))
const responses = wire.trim().split('\n').map((line) => JSON.parse(line))
assert.ok(responses.some((response) => response.error?.code === -32600))
assert.ok(responses.some((response) => response.error?.code === -32700))
assert.deepEqual(responses.find((response) => response.id === 9)?.result, {})
console.log('PASS malformed stdio messages preserve subsequent MCP requests')
