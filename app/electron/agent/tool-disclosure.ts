import type { McpToolDefinition } from './types'

export const TOOL_DISCLOSURE_BRIDGE_NAMES = ['tool_search', 'tool_describe', 'tool_call'] as const
export const DEFAULT_DEFERRED_TOOL_THRESHOLD = 24

export interface DeferredToolCatalogEntry {
  name: string
  title: string
  description: string
  serverName: string
  inputSchema: Record<string, unknown>
}

export interface DeferredToolCatalog {
  tools: DeferredToolCatalogEntry[]
}

export interface DeferredToolSearchOptions {
  query: string
  limit?: number
}

export interface DeferredToolSearchResult {
  query: string
  matches: DeferredToolCatalogEntry[]
  omitted: number
}

export const TOOL_DISCLOSURE_BRIDGE_SCHEMAS = [
  {
    type: 'function',
    function: {
      name: 'tool_search',
      description: 'Search deferred extension tools by name, server, title, and description. Use this when the needed MCP/plugin tool is not directly visible.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          limit: { type: 'number', description: 'Maximum tools to return. Defaults to 8, capped at 20.' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'tool_describe',
      description: 'Read the schema and description for one deferred extension tool returned by tool_search.',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Deferred tool call name such as mcp__server__tool.' },
        },
        required: ['name'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'tool_call',
      description: 'Call one deferred extension tool by name with JSON arguments. This is approval-gated because deferred tools may have external side effects.',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Deferred tool call name such as mcp__server__tool.' },
          arguments: { type: 'object', description: 'JSON arguments for the deferred tool.' },
        },
        required: ['name'],
      },
    },
  },
]

export function shouldDeferMcpTools(mcpTools: McpToolDefinition[], threshold = DEFAULT_DEFERRED_TOOL_THRESHOLD): boolean {
  return mcpTools.length > Math.max(0, Math.floor(threshold))
}

export function buildDeferredToolCatalog(mcpTools: McpToolDefinition[]): DeferredToolCatalog {
  return {
    tools: mcpTools.map((tool) => ({
      name: tool.callName,
      title: tool.title || tool.name || tool.callName,
      description: tool.description || `Call ${tool.name} on ${tool.serverName}.`,
      serverName: tool.serverName,
      inputSchema: schemaObject(tool.inputSchema),
    })),
  }
}

export function searchDeferredToolCatalog(catalog: DeferredToolCatalog, options: DeferredToolSearchOptions): DeferredToolCatalogEntry[] {
  const terms = queryTerms(options.query)
  const limit = clampLimit(options.limit, 8, 1, 20)
  if (!terms.length) return []
  return catalog.tools
    .map((tool) => ({ tool, score: scoreTool(tool, terms) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.tool.name.localeCompare(b.tool.name))
    .slice(0, limit)
    .map((entry) => entry.tool)
}

export function findDeferredTool(catalog: DeferredToolCatalog, name: string): DeferredToolCatalogEntry | null {
  const needle = String(name ?? '').trim()
  if (!needle) return null
  return catalog.tools.find((tool) => tool.name === needle) ?? null
}

export function formatToolCatalogSearch(result: DeferredToolSearchResult): string {
  if (!result.matches.length) return `No deferred tools matched "${result.query}".`
  const lines = result.matches.map((tool, index) => [
    `${index + 1}. ${tool.name}`,
    `   title: ${tool.title}`,
    `   server: ${tool.serverName}`,
    `   description: ${tool.description}`,
  ].join('\n'))
  if (result.omitted) lines.push(`...${result.omitted} more matches omitted.`)
  return lines.join('\n\n')
}

export function formatDeferredToolDescription(tool: DeferredToolCatalogEntry | null): string {
  if (!tool) return 'No deferred tool found for that name.'
  return [
    `Tool: ${tool.name}`,
    `Title: ${tool.title}`,
    `Server: ${tool.serverName}`,
    `Description: ${tool.description}`,
    'Input schema:',
    JSON.stringify(tool.inputSchema, null, 2),
  ].join('\n')
}

function scoreTool(tool: DeferredToolCatalogEntry, terms: string[]): number {
  const haystack = `${tool.name} ${tool.title} ${tool.serverName} ${tool.description}`.toLowerCase()
  let score = 0
  for (const term of terms) {
    if (tool.name.toLowerCase().includes(term)) score += 4
    if (tool.title.toLowerCase().includes(term)) score += 3
    if (tool.description.toLowerCase().includes(term)) score += 2
    if (haystack.includes(term)) score += 1
  }
  return score
}

function queryTerms(query: string): string[] {
  return [...new Set(String(query ?? '').toLowerCase().split(/[^a-z0-9_/-]+/).map((term) => term.trim()).filter((term) => term.length >= 2))]
}

function clampLimit(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Math.floor(Number(value))
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(min, Math.min(max, parsed))
}

function schemaObject(value: Record<string, unknown>): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value
    : { type: 'object', properties: {} }
}
