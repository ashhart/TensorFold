import { buildToolSchemas } from './tool-schema-builder'
import { normalizeTicketCapabilities, ticketDisallowedToolNames } from './tickets'
import type { TicketCapability } from './types'
import type { ToolContext } from './tools'

type TicketToolContext = Partial<ToolContext> & Record<string, any>

/**
 * Turn a host tool context into the executable context for one ticket. This is a
 * defense-in-depth companion to schema filtering: callbacks and sensitive source stores
 * for forbidden lanes are removed, so a stale/hallucinated tool call cannot recover them.
 */
export function restrictTicketToolContext(context: TicketToolContext, input: unknown): TicketToolContext {
  const capabilities = normalizeTicketCapabilities(input, ['read'])
  const allowed = new Set<TicketCapability>(capabilities)
  const schemas = buildToolSchemas(context.mcpTools as any, {
    disabledCapabilityIds: context.disabledCapabilityIds,
    disabledToolNames: context.disabledToolNames,
    skills: context.skills,
    commands: context.commands,
    localScripts: context.localScripts,
    packagedPlugins: context.packagedPlugins,
  })
  const schemaNames = schemas.map((schema: any) => schema.function?.name).filter((name: unknown): name is string => typeof name === 'string')
  const disabledToolNames = [...new Set([
    ...(context.disabledToolNames ?? []),
    ...ticketDisallowedToolNames(schemaNames, capabilities),
  ])]
  const connectorAllowed = allowed.has('connector')
  const browserAllowed = allowed.has('browser')
  const mcpAllowed = allowed.has('mcp')
  const writeAllowed = allowed.has('write')

  return {
    ...context,
    disabledToolNames,
    networkCapabilityAllowed: allowed.has('network'),
    allowSandboxNetwork: allowed.has('network') && context.allowSandboxNetwork === true,
    sensitiveRetrievalAllowed: connectorAllowed && context.sensitiveRetrievalAllowed !== false,
    sensitiveRetrievalReason: connectorAllowed
      ? context.sensitiveRetrievalReason
      : 'ticket allowedCapabilities excludes connector-backed retrieval sources',
    emailSettings: connectorAllowed ? context.emailSettings : undefined,
    slackSettings: connectorAllowed ? context.slackSettings : undefined,
    calendarSettings: connectorAllowed ? context.calendarSettings : undefined,
    documentSettings: connectorAllowed ? context.documentSettings : undefined,
    connectorEvents: connectorAllowed ? context.connectorEvents : [],
    recordEmailDraft: connectorAllowed ? context.recordEmailDraft : undefined,
    recordEmailEvent: connectorAllowed ? context.recordEmailEvent : undefined,
    recordCalendarEvent: connectorAllowed ? context.recordCalendarEvent : undefined,
    recordMessageEvent: connectorAllowed ? context.recordMessageEvent : undefined,
    browserSettings: browserAllowed ? context.browserSettings : undefined,
    runBrowserAutomation: browserAllowed ? context.runBrowserAutomation : undefined,
    runControllerAction: browserAllowed ? context.runControllerAction : undefined,
    mcpTools: mcpAllowed ? context.mcpTools : [],
    callMcpTool: mcpAllowed ? context.callMcpTool : undefined,
    registerArtifact: writeAllowed ? context.registerArtifact : undefined,
    fileSuggestion: writeAllowed ? context.fileSuggestion : undefined,
    // Ticket workers are already bounded workers. Nested runAgent subagents would rebuild a
    // fresh unrestricted host context, so keep that escape hatch out of this surface.
    launchSubAgent: undefined,
  }
}
