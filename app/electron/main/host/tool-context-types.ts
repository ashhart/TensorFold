import type { AgentEvent, Project, Provider, ReasoningEffortSetting, Skill, SubAgent, Thread, WorkspaceLocation } from '../../agent/types'
import type { SshManager } from './ssh'

export interface ReadOnlySubAgentInput {
      task: string
      parentLabel: string
      cwd: string
      projectId?: string | null
      providerAccountId?: string | null
      project?: Project | null
      workspace?: WorkspaceLocation | null
      reasoningEffort?: ReasoningEffortSetting
      signal: AbortSignal
      memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy
      incognito?: boolean
      memoryOwner?: { id: string; revision: number }
      maxSteps?: number
      maxWallMs?: number
    }

export interface ToolContextRuntime {
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal; subAgentId?: string; memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy; memoryOwner?: { id: string; revision: number } }): any
  runReadOnlySubAgent(
    agent: SubAgent,
    provider: Provider,
    model: string,
    input: ReadOnlySubAgentInput,
  ): Promise<{ agent: SubAgent; output: string }>
}

export interface ToolContextRuntimeDeps {
  sshManager: SshManager
  persistSubAgents(): void
  persistThreads(): void
  applyEventToThread(thread: Thread, event: AgentEvent): void
  emitToRenderer(event: AgentEvent): void
  emitPluginEvent(event: AgentEvent): void
  renderProjectMemoryBucket(project: Project | undefined): string
  recordEmailDraft: any
  recordEmailEvent: any
  recordCalendarEvent: any
  recordMessageEvent: any
  runBrowserAutomation: any
  mcpToolContext(): any
  // Ends a thread's run; the computer-use indicator's Stop uses it.
  cancelThreadRun(threadId: string): void
  // Trusted repo-authored skills for a project (VST-010), served from the scan cache so
  // this stays synchronous. Absent → global skills only.
  projectSkills?(projectId: string | null | undefined, workspace?: WorkspaceLocation | null): Skill[]
  // Observer hooks (VST-012): subagent lifecycle observation, fail-open.
  observe?: (kind: 'subagent.start' | 'subagent.end', payload: Record<string, unknown>) => void
}
