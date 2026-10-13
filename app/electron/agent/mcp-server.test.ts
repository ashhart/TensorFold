import assert from 'node:assert/strict'
import { handleMcpRequest, vontraToolsToMcp, MCP_SERVER_PROTOCOL_VERSION, type McpServerDeps } from './mcp-server'

const tools = [
  { name: 'read_file', description: 'Read a file', inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } },
]
let lastCall: { name: string; args: Record<string, any> } | null = null
const deps: McpServerDeps = {
  tools,
  serverInfo: { name: 'TensorFold Studio', version: '0.1.0' },
  callTool: async (name, args) => {
    lastCall = { name, args }
    if (name === 'read_file') return { ok: true, output: 'file contents' }
    throw new Error(`unknown tool: ${name}`)
  },
}

async function run() {
  // initialize -> advertises tools capability + server info + protocol version
  const init = await handleMcpRequest(deps, { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: MCP_SERVER_PROTOCOL_VERSION } })
  assert.equal(init?.id, 1)
  assert.equal(init?.result.serverInfo.name, 'TensorFold Studio')
  assert.ok(init?.result.capabilities.tools, 'advertises a tools capability')
  assert.equal(init?.result.protocolVersion, MCP_SERVER_PROTOCOL_VERSION)

  // notifications/initialized -> no response (notifications have no id)
  const note = await handleMcpRequest(deps, { jsonrpc: '2.0', method: 'notifications/initialized' })
  assert.equal(note, null, 'notifications get no response')

  // tools/list -> the MCP tool list
  const list = await handleMcpRequest(deps, { jsonrpc: '2.0', id: 2, method: 'tools/list' })
  assert.equal(list?.id, 2)
  assert.equal(list?.result.tools.length, 1)
  assert.equal(list?.result.tools[0].name, 'read_file')
  assert.ok(list?.result.tools[0].inputSchema.properties.path)

  // tools/call success -> ToolResult wrapped as MCP text content
  const call = await handleMcpRequest(deps, { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'read_file', arguments: { path: 'a.txt' } } })
  assert.equal(call?.id, 3)
  assert.deepEqual(lastCall, { name: 'read_file', args: { path: 'a.txt' } })
  assert.equal(call?.result.isError, false)
  assert.equal(call?.result.content[0].type, 'text')
  assert.equal(call?.result.content[0].text, 'file contents')

  // tools/call where the tool throws -> isError result (NOT a JSON-RPC error)
  const fail = await handleMcpRequest(deps, { jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'nope', arguments: {} } })
  assert.equal(fail?.result.isError, true, 'tool failures are reported via isError, not protocol error')
  assert.match(fail?.result.content[0].text, /unknown tool/)

  // tools/call with no name -> JSON-RPC invalid params error
  const bad = await handleMcpRequest(deps, { jsonrpc: '2.0', id: 5, method: 'tools/call', params: {} })
  assert.ok(bad?.error, 'missing tool name is a protocol error')
  assert.equal(bad?.error?.code, -32602)

  // unknown method -> method not found
  const unk = await handleMcpRequest(deps, { jsonrpc: '2.0', id: 6, method: 'frobnicate' })
  assert.equal(unk?.error?.code, -32601)

  // vontraToolsToMcp: OpenAI function-tool schema -> MCP tool def
  const mcp = vontraToolsToMcp([
    { type: 'function', function: { name: 'glob', description: 'Find files', parameters: { type: 'object', properties: { pattern: { type: 'string' } } } } },
  ])
  assert.equal(mcp.length, 1)
  assert.equal(mcp[0].name, 'glob')
  assert.equal(mcp[0].description, 'Find files')
  assert.deepEqual(mcp[0].inputSchema, { type: 'object', properties: { pattern: { type: 'string' } } })

  console.log('mcp-server tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
