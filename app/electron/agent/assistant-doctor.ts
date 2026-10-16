import { DEFAULT_DEFERRED_TOOL_THRESHOLD, shouldDeferMcpTools } from './tool-disclosure'
import { buildStaticToolRegistry } from './tool-registry'
import type { Command, LocalScriptCapability, McpToolDefinition, PackagedPluginCapability, Skill, Thread } from './types'

export interface AssistantDoctorInput {
  currentThread?: Thread | null
  threads?: Thread[]
  mcpTools?: McpToolDefinition[]
  skills?: Skill[]
  commands?: Command[]
  localScripts?: LocalScriptCapability[]
  packagedPlugins?: PackagedPluginCapability[]
  disabledCapabilityIds?: string[]
  deferredToolThreshold?: number
}

export interface AssistantDoctorReport {
  promptSnapshot: {
    present: boolean
    fingerprint: string | null
    sections: Record<'base' | 'persona' | 'goal' | 'capabilities', number>
  }
  toolSurface: {
    staticTools: number
    mcpTools: number
    deferredMcpTools: boolean
    deferredToolThreshold: number
    skills: number
    commands: number
    localScripts: number
    packagedPlugins: number
    disabledCapabilities: string[]
  }
  memoryAndRecall: {
    sessionThreads: number
    sessionRecallReady: boolean
  }
}

export function buildAssistantDoctorReport(input: AssistantDoctorInput = {}): AssistantDoctorReport {
  const snapshot = input.currentThread?.systemPromptSnapshot ?? null
  const mcpTools = input.mcpTools ?? []
  const threshold = normalizedThreshold(input.deferredToolThreshold)
  const staticTools = buildStaticToolRegistry({ scope: 'agent' }).length
  const sessionThreads = input.threads?.length ?? 0

  return {
    promptSnapshot: {
      present: !!snapshot,
      fingerprint: snapshot?.fingerprint ?? null,
      sections: {
        base: snapshot?.sections.base.length ?? 0,
        persona: snapshot?.sections.persona.length ?? 0,
        goal: snapshot?.sections.goal.length ?? 0,
        capabilities: snapshot?.sections.capabilities.length ?? 0,
      },
    },
    toolSurface: {
      staticTools,
      mcpTools: mcpTools.length,
      deferredMcpTools: shouldDeferMcpTools(mcpTools, threshold),
      deferredToolThreshold: threshold,
      skills: (input.skills ?? []).filter((skill) => skill.enabled !== false).length,
      commands: input.commands?.length ?? 0,
      localScripts: input.localScripts?.length ?? 0,
      packagedPlugins: input.packagedPlugins?.length ?? 0,
      disabledCapabilities: [...new Set(input.disabledCapabilityIds ?? [])].sort(),
    },
    memoryAndRecall: {
      sessionThreads,
      sessionRecallReady: sessionThreads > 0,
    },
  }
}

export function formatAssistantDoctorReport(report: AssistantDoctorReport): string {
  const prompt = report.promptSnapshot
  const tools = report.toolSurface
  const recall = report.memoryAndRecall
  return [
    'Assistant doctor',
    `Prompt snapshot: ${prompt.present ? `present (${prompt.fingerprint})` : 'missing'}`,
    `Prompt section bytes: base=${prompt.sections.base}, persona=${prompt.sections.persona}, goal=${prompt.sections.goal}, capabilities=${prompt.sections.capabilities}`,
    `Static tools: ${tools.staticTools}`,
    `MCP tools: ${tools.mcpTools}${tools.deferredMcpTools ? ' (deferred behind bridge)' : ''}`,
    `Deferred threshold: ${tools.deferredToolThreshold}`,
    `Skills: ${tools.skills}; Commands: ${tools.commands}; Local scripts: ${tools.localScripts}; Packaged plugins: ${tools.packagedPlugins}`,
    `Disabled capabilities: ${tools.disabledCapabilities.length ? tools.disabledCapabilities.join(', ') : 'none'}`,
    `Session recall: ${recall.sessionRecallReady ? 'ready' : 'empty'} (${recall.sessionThreads} threads)`,
  ].join('\n')
}

function normalizedThreshold(value: unknown): number {
  const parsed = Math.floor(Number(value))
  if (!Number.isFinite(parsed)) return DEFAULT_DEFERRED_TOOL_THRESHOLD
  return Math.max(0, parsed)
}
