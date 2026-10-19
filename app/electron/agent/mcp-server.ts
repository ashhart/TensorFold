// TensorFold-Studio-as-MCP-server: exposes TensorFold's tools over the Model Context Protocol
// (JSON-RPC 2.0 over stdio) so external MCP clients — Codex CLI/app, Claude
// CLI/app, or any MCP-aware agent — can call TensorFold's capabilities. This is the
// outbound half of the MCP bridge (the inbound half is the McpHost client in
// mcp-client.ts). Built on Node built-ins; no SDK dependency.

import readline from 'node:readline'
import { isMcpEnvelope } from './mcp-envelope'

export const MCP_SERVER_PROTOCOL_VERSION = '2025-06-18'

export interface McpServerToolDef {
  name: string
  description?: string
  inputSchema: Record<string, unknown>
}

export interface McpServerToolResult {
  ok: boolean
  output: string
}

export interface McpServerDeps {
  /** Tools to advertise (MCP shape). Use vontraToolsToMcp() to convert TensorFold schemas. */
  tools: McpServerToolDef[]
  /** Execute a tool call and return its textual result. */
  callTool: (name: string, args: Record<string, any>) => Promise<McpServerToolResult>
  serverInfo?: { name: string; version: string }
}

interface JsonRpcResponse {
  jsonrpc: '2.0'
  id: string | number | null
  result?: any
  error?: { code: number; message: string }
}

const DEFAULT_SERVER_INFO = { name: 'TensorFold', version: '0.1.0' }

/** Convert TensorFold's OpenAI function-tool schemas (buildToolSchemas output) into MCP tool defs. */
export function vontraToolsToMcp(
  schemas: Array<{ type?: string; function?: { name?: string; description?: string; parameters?: Record<string, unknown> } }>,
): McpServerToolDef[] {
  const out: McpServerToolDef[] = []
  for (const schema of schemas) {
    const fn = schema?.function
    const name = String(fn?.name ?? '').trim()
    if (!name) continue
    out.push({
      name,
      description: fn?.description ? String(fn.description) : undefined,
      inputSchema: fn?.parameters && typeof fn.parameters === 'object' ? fn.parameters : { type: 'object', properties: {} },
    })
  }
  return out
}

function ok(id: string | number, result: any): JsonRpcResponse {
  return { jsonrpc: '2.0', id, result }
}
function fail(id: string | number | null, code: number, message: string): JsonRpcResponse {
  return { jsonrpc: '2.0', id, error: { code, message } }
}

/**
 * Handle one JSON-RPC message. Returns a response for requests, or null for
 * notifications (no id). Pure and dependency-injected so it is fully unit-testable.
 */
export async function handleMcpRequest(deps: McpServerDeps, message: unknown): Promise<JsonRpcResponse | null> {
  if (!isMcpEnvelope(message) || !message.method) return fail(null, -32600, 'Invalid Request')
  const { id, method } = message
  if (id == null) return null // notification — no response
  const serverInfo = deps.serverInfo ?? DEFAULT_SERVER_INFO

  if (method === 'initialize') {
    return ok(id, {
      protocolVersion: MCP_SERVER_PROTOCOL_VERSION,
      capabilities: { tools: { listChanged: false } },
      serverInfo,
    })
  }
  if (method === 'ping') return ok(id, {})
  if (method === 'tools/list') return ok(id, { tools: deps.tools })
  if (method === 'tools/call') {
    const name = String(message.params?.name ?? '').trim()
    if (!name) return fail(id, -32602, 'tools/call requires a tool name')
    const args = message.params?.arguments && typeof message.params.arguments === 'object' ? message.params.arguments : {}
    try {
      const result = await deps.callTool(name, args)
      return ok(id, { content: [{ type: 'text', text: result.output ?? '' }], isError: !result.ok })
    } catch (e: any) {
      // Tool failures are surfaced as tool errors (isError), not JSON-RPC protocol errors.
      return ok(id, { content: [{ type: 'text', text: e?.message ? String(e.message) : String(e) }], isError: true })
    }
  }
  return fail(id, -32601, `Method not found: ${method}`)
}

/** Run the server over line-delimited JSON-RPC on stdio (or injected streams). */
export function runMcpServerStdio(
  deps: McpServerDeps,
  input: NodeJS.ReadableStream = process.stdin,
  output: NodeJS.WritableStream = process.stdout,
): void {
  const rl = readline.createInterface({ input })
  rl.on('line', (line) => {
    const raw = String(line || '').trim()
    if (!raw) return
    let message: unknown
    try {
      message = JSON.parse(raw)
    } catch {
      output.write(`${JSON.stringify(fail(null, -32700, 'Parse error'))}\n`)
      return
    }
    void handleMcpRequest(deps, message)
      .then((response) => {
        if (response) output.write(`${JSON.stringify(response)}\n`)
      })
      .catch((e: any) => {
        const id = isMcpEnvelope(message) ? message.id : null
        if (id != null) output.write(`${JSON.stringify(fail(id, -32603, e?.message ? String(e.message) : String(e)))}\n`)
      })
  })
}
