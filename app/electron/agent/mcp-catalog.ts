import type { McpPromptDefinition, McpResourceDefinition, McpServerConfig, McpToolCallResult, McpToolDefinition } from './types'
import { isRecord } from './mcp-envelope'

export function normalizeTool(config: McpServerConfig, tool: any, name: string, index: number, used?: Set<string>): McpToolDefinition {
  const annotations = isRecord(tool?.annotations) ? tool.annotations : {}
  return {
    serverId: config.id,
    serverName: config.name,
    approvalMode: config.approvalMode,
    name,
    callName: uniqueCallName(config.id, name, index, used),
    title: typeof tool?.title === 'string' ? tool.title : undefined,
    description: typeof tool?.description === 'string' ? tool.description : undefined,
    inputSchema: isRecord(tool?.inputSchema) ? tool.inputSchema : { type: 'object', properties: {} },
    readOnlyHint: annotations.readOnlyHint === true,
    destructiveHint: annotations.destructiveHint === true,
    openWorldHint: annotations.openWorldHint === true,
  }
}

export function normalizeMcpResource(config: McpServerConfig, resource: any, index: number): McpResourceDefinition | null {
  const uri = typeof resource?.uri === 'string' ? resource.uri.trim() : ''
  if (!uri) return null
  const title = typeof resource?.title === 'string' && resource.title.trim() ? resource.title.trim() : undefined
  const rawName = typeof resource?.name === 'string' ? resource.name.trim() : ''
  const name = rawName || title || uri || `resource-${index + 1}`
  return {
    serverId: config.id,
    serverName: config.name,
    uri,
    name,
    title,
    description: typeof resource?.description === 'string' && resource.description.trim() ? resource.description.trim() : undefined,
    mimeType: typeof resource?.mimeType === 'string' && resource.mimeType.trim() ? resource.mimeType.trim() : undefined,
  }
}

export function normalizeMcpPrompt(config: McpServerConfig, prompt: any, _index: number): McpPromptDefinition | null {
  const name = typeof prompt?.name === 'string' ? prompt.name.trim() : ''
  if (!name) return null
  const args: any[] = Array.isArray(prompt?.arguments) ? prompt.arguments : []
  return {
    serverId: config.id,
    serverName: config.name,
    name,
    title: typeof prompt?.title === 'string' && prompt.title.trim() ? prompt.title.trim() : undefined,
    description: typeof prompt?.description === 'string' && prompt.description.trim() ? prompt.description.trim() : undefined,
    arguments: args.flatMap((arg) => {
      const argName = typeof arg?.name === 'string' ? arg.name.trim() : ''
      if (!argName) return []
      return [{
        name: argName,
        description: typeof arg?.description === 'string' && arg.description.trim() ? arg.description.trim() : undefined,
        required: arg?.required === true,
      }]
    }),
  }
}

function uniqueCallName(serverId: string, toolName: string, index: number, used?: Set<string>): string {
  const base = base60(`mcp__${safeName(serverId)}__${safeName(toolName)}`, index)
  if (!used) return base
  if (!used.has(base)) { used.add(base); return base }
  let n = 1
  let name = `${base.slice(0, 57)}_${n}`
  while (used.has(name)) { n++; name = `${base.slice(0, 57)}_${n}` }
  used.add(name)
  return name
}

function base60(name: string, index: number): string {
  return name.length <= 60 ? name : `${name.slice(0, 52)}_${index}`
}

function safeName(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]/g, '_').replace(/_+/g, '_').replace(/^_+|_+$/g, '') || 'tool'
}

export function formatToolResult(tool: McpToolDefinition, result: any): McpToolCallResult {
  const chunks: string[] = []
  const content = Array.isArray(result?.content) ? result.content : []
  for (const item of content) {
    if (item?.type === 'text' && typeof item.text === 'string') chunks.push(item.text)
    else if (item?.type === 'resource_link' && item.uri) chunks.push(`[resource] ${item.name || item.uri}: ${item.uri}`)
    else if (item?.type === 'image') chunks.push(`[image] ${item.mimeType || 'image'} ${item.data ? `${String(item.data).length} bytes base64` : ''}`.trim())
    else if (item?.type) chunks.push(`[${item.type}] ${JSON.stringify(item).slice(0, 2000)}`)
  }
  if (isRecord(result?.structuredContent)) chunks.push(JSON.stringify(result.structuredContent, null, 2))
  const output = chunks.join('\n\n').trim() || JSON.stringify(result ?? {}, null, 2)
  return {
    ok: result?.isError !== true,
    output,
    title: tool.title || tool.name,
    meta: `${tool.serverName} MCP`,
  }
}
