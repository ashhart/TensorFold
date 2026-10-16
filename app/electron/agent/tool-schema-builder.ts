import { buildCapabilityRegistry, filterToolSchemasForRegistry } from './capability-registry'
import { TOOL_DISCLOSURE_BRIDGE_SCHEMAS, shouldDeferMcpTools } from './tool-disclosure'
import { CHAT_TOOL_SCHEMAS, TOOL_SCHEMAS } from './tool-schemas'
import { buildStaticToolRegistry, isRegistryReadOnlyTool, READ_ONLY_TOOL_NAMES, toolRegistrySchemas } from './tool-registry'
import type { ApprovalMode, CapabilityRegistrySnapshot, Command, LocalScriptCapability, McpToolDefinition, PackagedPluginCapability, Skill } from './types'

export { CHAT_TOOL_SCHEMAS, TOOL_SCHEMAS } from './tool-schemas'

export interface BuildToolSchemaOptions {
  capabilityRegistry?: CapabilityRegistrySnapshot
  selectedCapabilityIds?: string[]
  disabledCapabilityIds?: string[]
  disabledToolNames?: string[]
  skills?: Skill[]
  commands?: Command[]
  localScripts?: LocalScriptCapability[]
  packagedPlugins?: PackagedPluginCapability[]
  deferredToolThreshold?: number
}

export const READ_ONLY_TOOLS = READ_ONLY_TOOL_NAMES

export function buildToolSchemas(mcpTools: McpToolDefinition[] = [], options: BuildToolSchemaOptions = {}) {
  const staticSchemas = toolRegistrySchemas(buildStaticToolRegistry({ scope: 'agent' }))
  const deferMcpTools = shouldDeferMcpTools(mcpTools, options.deferredToolThreshold)
  const schemas = [
    ...staticSchemas,
    ...(deferMcpTools ? TOOL_DISCLOSURE_BRIDGE_SCHEMAS : []),
    ...(deferMcpTools ? [] : buildMcpToolSchemas(mcpTools)),
  ]
  const registry = options.capabilityRegistry ?? buildCapabilityRegistry({
    mcpTools,
    skills: options.skills,
    commands: options.commands,
    localScripts: options.localScripts,
    packagedPlugins: options.packagedPlugins,
    selectedCapabilityIds: options.selectedCapabilityIds,
    disabledCapabilityIds: options.disabledCapabilityIds,
    disabledToolNames: options.disabledToolNames,
    knownToolNames: toolSchemaNames(schemas),
  })
  return filterToolSchemasForRegistry(schemas, registry)
}

export function buildChatToolSchemas(options: BuildToolSchemaOptions = {}) {
  const staticSchemas = toolRegistrySchemas(buildStaticToolRegistry({ scope: 'chat' }))
  const registry = options.capabilityRegistry ?? buildCapabilityRegistry({
    selectedCapabilityIds: options.selectedCapabilityIds,
    skills: options.skills,
    commands: options.commands,
    localScripts: options.localScripts,
    packagedPlugins: options.packagedPlugins,
    disabledCapabilityIds: options.disabledCapabilityIds,
    disabledToolNames: options.disabledToolNames,
    knownToolNames: toolSchemaNames(staticSchemas),
  })
  return filterToolSchemasForRegistry(staticSchemas, registry)
}

export function isReadOnlyTool(name: string, mcpTools: McpToolDefinition[] = [], approvalMode?: ApprovalMode): boolean {
  if (isRegistryReadOnlyTool(name)) return true
  // MCP tools are NOT auto-classified read-only from the server's own annotations. The
  // readOnlyHint/destructiveHint/openWorldHint fields come straight from the server's
  // tools/list response, so a compromised or malicious "trusted" server could advertise
  // readOnlyHint=true to skip approval entirely. Every MCP tool call therefore goes through
  // the normal approval gate (a one-time "Always allow" still remembers a specific call).
  void mcpTools
  void approvalMode
  return false
}

function buildMcpToolSchemas(mcpTools: McpToolDefinition[]) {
  return mcpTools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.callName,
      description: [
        `[MCP:${tool.serverName}]`,
        tool.title && tool.title !== tool.name ? `${tool.title}.` : '',
        tool.description || `Call ${tool.name} on ${tool.serverName}.`,
        tool.readOnlyHint ? 'Read-only.' : '',
        tool.destructiveHint ? 'May modify or delete data.' : '',
        tool.openWorldHint ? 'May contact external systems.' : '',
      ].filter(Boolean).join(' '),
      parameters: schemaObject(tool.inputSchema),
    },
  }))
}

function toolSchemaNames(schemas: Array<{ function?: { name?: string } }>): string[] {
  return schemas.map((schema) => schema.function?.name).filter((name): name is string => !!name)
}

function schemaObject(value: Record<string, unknown>): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value
    : { type: 'object', properties: {} }
}
