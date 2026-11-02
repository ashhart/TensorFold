import { type DocumentEvidencePort } from './document-evidence-tool'
import type { CreateArtifactInput } from './artifacts'
import path from 'node:path'
import { type BrowserAutomationInput, type BrowserAutomationResult } from './browser-automation'
import { type ComputerUseInput } from './computer-use/actions'
import type { UserQuestionInput, UserQuestionOutcome, UserQuestionRun } from './types'
import type { PhoneToolInput } from './types'
import { type ControllerActionInput, type ControllerActionResult } from './controller-actions'
import { type SandboxArtifactBundle } from './sandbox'
import { type SessionSearchOptions, type SessionSearchResult } from './session-recall'
import { type WorkspaceBackend } from './workspace-backend'
import type { BrowserConnectorSettings, CalendarConnectorSettings, Command, ConnectorEvent, ConnectorSourceRef, DiffLine, DocumentConnectorSettings, EmailConnectorSettings, EmailDraftInput, LocalScriptCapability, McpToolCallResult, McpToolDefinition, Mission, PackagedPluginCapability, Artifact, ProjectMemoryEntry, Room, Skill, SkillSuggestion, SlackConnectorSettings, SubAgent, SubAgentMessage, SubAgentMessageKind, Thread, Ticket, VisualizationSpec } from './types'

type PersonalMessageConnectorId = Extract<ConnectorEvent['connectorId'], 'telegram' | 'whatsapp' | 'imessage'>

export interface LaunchSubAgentToolInput {
  task: string
  name?: string | null
  model?: string | null
  providerId?: string | null
  accountId?: string | null
  agent?: string | null
  background?: boolean
}

export interface LaunchSubAgentToolResult {
  agent: SubAgent
  output: string
}

export interface SubAgentMailboxSendInput {
  agentId: string
  text: string
  kind?: SubAgentMessageKind
  replyToId?: string | null
}

export interface SubAgentMailboxReadInput {
  agentId: string
  unreadOnly?: boolean
  markRead?: boolean
}

export interface ToolContext {
  // Host checks hidden engine research before every restricted-bot model request.
  botToolScopeGuard?: () => void
  askUser?: (input: UserQuestionInput, run: UserQuestionRun) => Promise<UserQuestionOutcome>
  userQuestionRunId?: string
  userQuestionItemId?: string
  // A strict executable allowlist, also enforced before any host dispatch.
  allowedToolNames?: string[]
  // An isolated guest dispatcher; its presence never permits host fallback.
  botVm?: { executeTool(name: string, args: Record<string, any>, signal?: AbortSignal): Promise<ToolResult>; assertModelPrivacy?: () => void }
  phoneControlAvailable?: boolean
  phoneLocalOnly?: boolean
  phonePrivacyGuard?: () => string | null
  googlePrivacyGuard?: () => string | null
  runPhoneControl?: (input: PhoneToolInput) => Promise<{ ok: boolean; output: string; title: string }>
  runGoogleWorkspaceTool?: (name: string, args: Record<string, any>) => Promise<ToolResult>
  cwd: string
  // Cancels in-flight tool work when the owning run is stopped.
  signal?: AbortSignal
  memoryPolicy?: import('./types').ThreadMemoryPolicy
  currentThread?: Thread
  // Explicit workspace identity, retained even while an SSH backend is disconnected.
  workspace?: import('./types').WorkspaceLocation | null
  // When set to an SSH backend, file/shell tools execute on the remote host.
  // Absent => local fs (a LocalBackend is created on demand).
  backend?: WorkspaceBackend
  emailSettings?: EmailConnectorSettings
  slackSettings?: SlackConnectorSettings
  calendarSettings?: CalendarConnectorSettings
  documentSettings?: DocumentConnectorSettings
  documentLibrary?: DocumentEvidencePort
  browserSettings?: BrowserConnectorSettings
  threads?: Thread[]
  // Re-read canonical source privacy at dispatch instead of retaining a stale permissive list.
  recallThreads?: () => Thread[]
  recordRecallSources?: (threadIds: string[]) => void
  assertRecallSourcesAllowed?: () => void
  // Host-backed derived index. The in-memory scan remains the portable fallback.
  searchSessions?: (options: SessionSearchOptions) => SessionSearchResult | Promise<SessionSearchResult>
  connectorEvents?: ConnectorEvent[]
  missions?: Mission[]
  recallMissions?: () => Mission[]
  rooms?: Room[]
  tickets?: Ticket[]
  projectMemory?: ProjectMemoryEntry[]
  // Host-owned project scope and live Trash state, re-read after asynchronous collection.
  recallProjectMemory?: () => ProjectMemoryEntry[]
  // First-class artifact registration (VST-011): the host persists the record and links it
  // to this run's provenance. Absent (subagents, quick completions) → save_artifact refuses.
  registerArtifact?: (input: CreateArtifactInput) => Artifact
  recordEmailDraft?: (input: EmailDraftInput, sources?: ConnectorSourceRef[]) => Promise<ConnectorEvent>
  recordEmailEvent?: (event: ConnectorEvent) => Promise<ConnectorEvent>
  recordCalendarEvent?: (event: ConnectorEvent) => Promise<ConnectorEvent>
  recordMessageEvent?: (event: ConnectorEvent) => Promise<ConnectorEvent>
  launchSubAgent?: (input: LaunchSubAgentToolInput) => Promise<LaunchSubAgentToolResult>
  listSubAgents?: () => SubAgent[]
  currentSubAgentId?: string
  sendSubAgentMessage?: (input: SubAgentMailboxSendInput) => SubAgentMessage
  readSubAgentMessages?: (input: SubAgentMailboxReadInput) => SubAgentMessage[]
  mcpTools?: McpToolDefinition[]
  skills?: Skill[]
  commands?: Command[]
  skillSuggestions?: SkillSuggestion[]
  // File a consent-first Skill/Command suggestion (the user accepts it later). Injected
  // by the host; the agent never creates skills itself.
  fileSuggestion?: (input: import('./suggestions').SuggestionInput) => void
  localScripts?: LocalScriptCapability[]
  packagedPlugins?: PackagedPluginCapability[]
  callMcpTool?: (callName: string, args: Record<string, any>) => Promise<McpToolCallResult>
  runBrowserAutomation?: (input: BrowserAutomationInput) => Promise<BrowserAutomationResult>
  runControllerAction?: (input: ControllerActionInput) => Promise<ControllerActionResult>
  runComputerUse?: (input: ComputerUseInput) => Promise<{ ok: boolean; output: string; title: string; meta?: string }>
  disabledCapabilityIds?: string[]
  // Per-run executable boundary (for example a ticket's allowedCapabilities). Hidden
  // schemas are also rejected by the loop before execution, not merely omitted from prompt.
  disabledToolNames?: string[]
  writeManagedOutput?: (suggestedRef: string, content: string) => Promise<string>
  readManagedOutput?: (ref: string) => Promise<string | null>
  networkCapabilityAllowed?: boolean
  sensitiveRetrievalAllowed?: boolean
  sensitiveRetrievalReason?: string
  // When true, sandboxed shell commands may reach the network (off by default —
  // the sandbox denies network so a task can't exfiltrate or fetch unexpectedly).
  // Set from the "Allow network in task sandbox" setting.
  allowSandboxNetwork?: boolean
}

export interface ToolResult {
  ok: boolean
  output: string
  // display hints
  title?: string
  path?: string
  command?: string
  diff?: DiffLine[]
  // Workspace-relative paths demonstrably mutated by this tool result.
  changedPaths?: string[]
  visualization?: VisualizationSpec
  meta?: string
  subAgent?: SubAgent
  sources?: ConnectorSourceRef[]
  documentEvidence?: import('../../shared/document-library').DocumentEvidenceRequest[]
  sandboxArtifacts?: SandboxArtifactBundle
  artifacts?: import('./types').ToolArtifactSummary
  appAction?: import('./types').AppAction
}
