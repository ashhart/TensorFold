import assert from 'node:assert/strict'
import { externalEffectApprovalReason } from './approval-policy'
import { buildToolSchemas, executeTool, isReadOnlyTool } from './tools.ts'
import {
  buildDeferredToolCatalog,
  formatToolCatalogSearch,
  searchDeferredToolCatalog,
  TOOL_DISCLOSURE_BRIDGE_NAMES,
} from './tool-disclosure'
import type { McpToolDefinition } from './types'

const mcpTools: McpToolDefinition[] = Array.from({ length: 28 }, (_, index) => ({
  serverId: 'srv',
  serverName: 'demo',
  name: `tool_${index}`,
  callName: `mcp__demo__tool_${index}`,
  title: index === 7 ? 'Issue Tracker Search' : `Tool ${index}`,
  description: index === 7 ? 'Search tickets, issues, and backlog notes.' : `Demo extension tool ${index}.`,
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string' },
    },
  },
  approvalMode: 'ask',
  readOnlyHint: true,
  destructiveHint: false,
  openWorldHint: true,
}))

const catalog = buildDeferredToolCatalog(mcpTools)
const hits = searchDeferredToolCatalog(catalog, { query: 'issue tickets', limit: 5 })

assert.equal(catalog.tools.length, 28)
assert.equal(hits[0].name, 'mcp__demo__tool_7')
assert.match(formatToolCatalogSearch({ query: 'issue tickets', matches: hits, omitted: 0 }), /Issue Tracker Search/)
console.log('ok deferred tool catalog searches MCP tools by title and description')

const schemas = buildToolSchemas(mcpTools, { deferredToolThreshold: 8 }) as any[]
const names = schemas.map((schema) => schema.function?.name)

for (const bridgeName of TOOL_DISCLOSURE_BRIDGE_NAMES) assert.ok(names.includes(bridgeName), `${bridgeName} visible`)
assert.ok(!names.includes('mcp__demo__tool_7'), 'large MCP surface is deferred from direct schemas')
assert.equal(isReadOnlyTool('tool_search'), true)
assert.equal(isReadOnlyTool('tool_describe'), true)
assert.equal(isReadOnlyTool('tool_call'), false)
assert.match(externalEffectApprovalReason('tool_call', { name: 'mcp__demo__tool_7' })!, /deferred tool/i)
console.log('ok large MCP surfaces are replaced by approval-safe bridge tools')

const toolSearch = await executeTool('tool_search', { query: 'backlog', limit: 3 }, { cwd: '/repo', mcpTools })
assert.equal(toolSearch.ok, true)
assert.match(toolSearch.output, /mcp__demo__tool_7/)

const toolDescribe = await executeTool('tool_describe', { name: 'mcp__demo__tool_7' }, { cwd: '/repo', mcpTools })
assert.equal(toolDescribe.ok, true)
assert.match(toolDescribe.output, /Issue Tracker Search/)

let called: { name: string; args: Record<string, any> } | null = null
const toolCall = await executeTool('tool_call', { name: 'mcp__demo__tool_7', arguments: { query: 'P0' } }, {
  cwd: '/repo',
  mcpTools,
  callMcpTool: async (name, args) => {
    called = { name, args }
    return { ok: true, output: `called ${name} ${args.query}`, title: 'MCP result' }
  },
})
assert.equal(toolCall.ok, true)
assert.deepEqual(called, { name: 'mcp__demo__tool_7', args: { query: 'P0' } })
assert.match(toolCall.output, /called mcp__demo__tool_7 P0/)
console.log('ok disclosure bridge tools search, describe, and call deferred MCP tools')
