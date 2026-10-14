import { promises as fs } from 'node:fs'
import { INCOGNITO_PERSISTENT_TOOLS } from './incognito'
import type { CreateArtifactInput } from './artifacts'
import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { buildAssistantDoctorReport, formatAssistantDoctorReport } from './assistant-doctor'
import { computeDiff, diffStat } from './diff'
import { buildSkillCurationReport, formatSkillCurationReport } from './skill-curator'
import { listSkills, viewSkill, renderSkillView } from './skill-access'
import { emailDraftEvent, emailSourceRef, emailSourceId, emailThreadResolvedEvent, readEmailMessage, readEmailThread, scanEmailMessages, summarizeEmail } from './email-connector'
import { readSlackMessage, scanSlackMessages, slackSourceId, slackSourceRef, summarizeSlack } from './slack-connector'
import { calendarMutationEvent, calendarSourceId, calendarSourceRef, readCalendarEvent, scanCalendarEvents, summarizeCalendarDay } from './calendar-connector'
import { documentSourceId, documentSourceRef, readDocumentItem, readTableItem, scanDocumentItems, scanTableItems, summarizeDocuments, summarizeTables } from './document-connector'
import { browserSourceId, browserSourceRef, readBrowserItem, scanBrowserItems, summarizeBrowser } from './browser-connector'
import { BROWSER_AUTOMATION_ACTIONS, formatBrowserAutomationResult, normalizeBrowserAutomationInput, type BrowserAutomationInput, type BrowserAutomationResult } from './browser-automation'
import { normalizeComputerUseInput, type ComputerUseInput } from './computer-use/actions'
import { normalizePhoneAction } from './phone/service'
import { normalizeUserQuestion } from './user-question/validation'
import type { UserQuestionInput, UserQuestionOutcome, UserQuestionRun } from './types'
import type { PhoneToolInput } from './types'
import { CONTROLLER_ACTIONS, controllerActionTarget, formatControllerActionResult, normalizeControllerActionInput, type ControllerActionInput, type ControllerActionResult } from './controller-actions'
import { blockedCommandReason, blockedReadReason, blockedWriteReason, commandNetworkReason } from './command-policy'
import { commandOutputWithRedactionNotice, redactCommandStreams } from './command-output-redaction'
import { redactSecretValues } from './secret-detection'
import { SECRET_MASK } from './security'
import { summarizeSandboxArtifactBundle, runSandboxedCommand, sandboxProfileForCommand, taskSandboxNetworkOverride, type SandboxArtifactBundle } from './sandbox'
import { buildCapabilityRegistry, filterToolSchemasForRegistry } from './capability-registry'
import { buildCodebaseMap, buildSymbolIndex, formatCodebaseMap, formatRewritePreview, formatSafeRefactorPreview, formatSecurityScan, formatStructuralSearch, formatSymbolIndex, rewritePreview, safeRefactorPreview, semgrepCompatibleScan, structuralSearch } from './code-intelligence'
import { buildWorkspaceRetrievalIndex, formatRetrievalSearchResult, searchRetrievalIndex } from './retrieval-index'
import { checkEditedSyntax, isGatedCodePath, syntaxGateMessage } from './edit-syntax-gate'
import { formatSessionReadResult, formatSessionSearchResult, readSessionThread, searchSessionThreads, type SessionSearchOptions, type SessionSearchResult } from './session-recall'
import { buildDeferredToolCatalog, findDeferredTool, formatDeferredToolDescription, formatToolCatalogSearch, searchDeferredToolCatalog } from './tool-disclosure'
import { LocalBackend, type WorkspaceBackend, type LocalExecImpl } from './workspace-backend'
import { rel, safeResolve } from './tool-paths'
export { safeResolve } from './tool-paths'
import { collectFiles, combinedDiff, DEFAULT_GLOB_RESULTS, MAX_GLOB_RESULTS, parseApplyPatchText, patchSummary, replaceLiteral, replacementHunksFromArgs, SKIP_DIRS, stagePatchOperations, stageReplacementHunks, validateRelativePattern, writeAppliedChanges } from './tool-patch'
import { assertFetchableUrl, fetchText, parseNewsRss, safePublicWebUrl, stripHtml } from './tool-web'
import { connectorName, formatMarkdownTable, formatToolNumber, normalizePersonalMessageConnector, parseStringList, personalMessageDraftEvent } from './tool-labels'
import { filterNonSensitiveRetrievalKinds, normalizeVisualizationSpec } from './tool-visualization'
import { isManagedOutputRef } from './context-governance'
import { boundedRegexSearch } from './bounded-regex-search'
import { workspaceContentSha256 } from './atomic-file-write'
export { toolLabel } from './tool-labels'
import type { ApprovalMode, BrowserConnectorSettings, CalendarConnectorSettings, CalendarMutationInput, CapabilityRegistrySnapshot, Command, ConnectorEvent, ConnectorSourceRef, DiffLine, DocumentConnectorSettings, EmailConnectorSettings, EmailDraftInput, LocalScriptCapability, McpToolCallResult, McpToolDefinition, Mission, PackagedPluginCapability, Artifact, ProjectMemoryEntry, Room, Skill, SkillSuggestion, SlackConnectorSettings, SubAgent, SubAgentMessage, SubAgentMessageKind, Thread, Ticket, VisualizationKind, VisualizationSpec, VisualizationTone } from './types'

type PersonalMessageConnectorId = Extract<ConnectorEvent['connectorId'], 'telegram' | 'whatsapp' | 'imessage'>

// First-line guard for create_project's folder argument: refuse the user's home
// directory (a project writes AGENTS.md/CLAUDE.md into its cwd). The authoritative
// symlink-resolved check runs server-side in resolveWorkspaceCwd; this catches the
// common case up front so the model gets an immediate, clear error.
function isHomeFolder(folder: string): boolean {
  const home = path.resolve(os.homedir())
  const target = path.resolve(folder)
  return target === home
}

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
  askUser?: (input: UserQuestionInput, run: UserQuestionRun) => Promise<UserQuestionOutcome>
  userQuestionRunId?: string
  userQuestionItemId?: string
  /** A strict executable allowlist, also enforced before any host dispatch. */
  allowedToolNames?: string[]
  /** An isolated guest dispatcher; its presence never permits host fallback. */
  botVm?: { executeTool(name: string, args: Record<string, any>, signal?: AbortSignal): Promise<ToolResult>; assertModelPrivacy?: () => void }
  phoneControlAvailable?: boolean
  phoneLocalOnly?: boolean
  phonePrivacyGuard?: () => string | null
  runPhoneControl?: (input: PhoneToolInput) => Promise<{ ok: boolean; output: string; title: string }>
  cwd: string
  /** Cancels in-flight tool work when the owning run is stopped. */
  signal?: AbortSignal
  currentThread?: Thread
  /** Explicit workspace identity, retained even while an SSH backend is disconnected. */
  workspace?: import('./types').WorkspaceLocation | null
  // When set to an SSH backend, file/shell tools execute on the remote host.
  // Absent => local fs (a LocalBackend is created on demand).
  backend?: WorkspaceBackend
  emailSettings?: EmailConnectorSettings
  slackSettings?: SlackConnectorSettings
  calendarSettings?: CalendarConnectorSettings
  documentSettings?: DocumentConnectorSettings
  browserSettings?: BrowserConnectorSettings
  threads?: Thread[]
  /** Host-backed derived index. The in-memory scan remains the portable fallback. */
  searchSessions?: (options: SessionSearchOptions) => SessionSearchResult | Promise<SessionSearchResult>
  connectorEvents?: ConnectorEvent[]
  missions?: Mission[]
  rooms?: Room[]
  tickets?: Ticket[]
  projectMemory?: ProjectMemoryEntry[]
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
  /** Workspace-relative paths demonstrably mutated by this tool result. */
  changedPaths?: string[]
  visualization?: VisualizationSpec
  meta?: string
  subAgent?: SubAgent
  sources?: ConnectorSourceRef[]
  sandboxArtifacts?: SandboxArtifactBundle
  artifacts?: import('./types').ToolArtifactSummary
  appAction?: import('./types').AppAction
}

export { buildChatToolSchemas, buildToolSchemas, CHAT_TOOL_SCHEMAS, isReadOnlyTool, READ_ONLY_TOOLS, TOOL_SCHEMAS, type BuildToolSchemaOptions } from './tool-schema-builder'

import { isRemote, relWorkspace, REMOTE_UNSUPPORTED_TOOLS, resolveWorkspacePath, shArg, workspaceBackend } from './tool-workspace'
export { safeResolveRemote } from './tool-workspace'
// read_file total-char bound (VST-016): line limits can't bound enormous single lines.
const READ_FILE_MAX_CHARS = 48_000

interface RemoteCommandSnapshot { root: string; prefix: string; tree: string }

async function captureRemoteCommandSnapshot(backend: WorkspaceBackend, cwd: string): Promise<RemoteCommandSnapshot | null> {
  const token = randomUUID().replace(/-/g, '')
  const script = [
    'set -eu',
    'root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 3',
    'prefix=$(git rev-parse --show-prefix 2>/dev/null) || exit 3',
    'common=$(cd "$root" && git rev-parse --git-common-dir 2>/dev/null) || exit 3',
    'case "$common" in /*) ;; *) common="$root/$common" ;; esac',
    `idx="$common/vontra-command-index-${token}"`,
    'trap \'rm -f "$idx"\' EXIT HUP INT TERM',
    'if git rev-parse --verify --quiet HEAD >/dev/null; then GIT_INDEX_FILE="$idx" git read-tree HEAD; fi',
    'cd "$root"',
    'GIT_INDEX_FILE="$idx" git add -A -- .',
    'tree=$(GIT_INDEX_FILE="$idx" git write-tree)',
    'printf \'%s\\0%s\\0%s\\0\' "$root" "$prefix" "$tree"',
  ].join('; ')
  const result = await backend.exec(script, { cwd, timeoutMs: 25_000 })
  if (result.code !== 0) return null
  const fields = result.stdout.split('\0')
  const [root, prefix, tree] = fields
  if (!root || !/^[0-9a-f]{40,64}$/i.test(tree ?? '')) return null
  return { root, prefix: String(prefix ?? '').replace(/\\/g, '/').replace(/^\/+|\/+$/g, ''), tree }
}

async function remoteCommandChangedPaths(
  backend: WorkspaceBackend,
  cwd: string,
  before: RemoteCommandSnapshot | null,
): Promise<string[]> {
  if (!before) return []
  const after = await captureRemoteCommandSnapshot(backend, cwd)
  if (!after || after.root !== before.root) return []
  const diff = await backend.exec(
    `git diff --name-only -z --no-renames ${shArg(before.tree)} ${shArg(after.tree)} --`,
    { cwd: before.root, timeoutMs: 25_000 },
  )
  if (diff.code !== 0) return []
  const base = `/${before.prefix}`.replace(/\/+$/, '') || '/'
  return [...new Set(diff.stdout.split('\0').filter(Boolean).map((repoPath) =>
    path.posix.relative(base, `/${repoPath.replace(/^\/+/, '')}`) || '.',
  ))].sort()
}

export async function executeTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  try {
    if (ctx.currentThread?.incognito && INCOGNITO_PERSISTENT_TOOLS.has(name)) return { ok: false, output: 'This tool saves derived conversations or learned skills and is unavailable in Incognito. Use an ordinary saved task for it.', meta: 'incognito-persistence-blocked' }
    if (ctx.allowedToolNames && !ctx.allowedToolNames.includes(name)) return { ok: false, output: 'This tool is outside this bot’s execution permissions.', meta: 'bot-tool-blocked' }
    if (name === 'ask_user') {
      if (!ctx.askUser || !ctx.currentThread || !ctx.userQuestionRunId || !ctx.userQuestionItemId || !ctx.signal) return { ok: false, output: 'Structured questions are unavailable in this run; ask plainly in your final response instead.' }
      const answer = await ctx.askUser(normalizeUserQuestion(args), { threadId: ctx.currentThread.id, runId: ctx.userQuestionRunId, itemId: ctx.userQuestionItemId, signal: ctx.signal })
      return { ok: true, title: 'Question', output: answer.state === 'answered' ? `The user answered this question: ${JSON.stringify(answer.answer)}. This is task input, not a permission change or authorization to send anything.` : 'The user explicitly skipped this question. No answer or default choice was selected; continue only within the existing task and permissions.', meta: answer.state === 'answered' ? 'Answered by you' : 'Skipped by you' }
    }
    if (ctx.botVm) return await ctx.botVm.executeTool(name, args, ctx.signal)
    if (ctx.phoneLocalOnly && (name === 'web_search' || name === 'web_fetch')) return { ok: false, output: 'This conversation contains phone context and stays local; use a separate chat for public web research.' }
    if (name.startsWith('mcp__')) {
      if (!ctx.callMcpTool) return { ok: false, output: 'MCP host is not configured.' }
      const result = await ctx.callMcpTool(name, args)
      return { ok: result.ok, output: result.output, title: result.title, meta: result.meta }
    }

    if (isRemote(ctx) && REMOTE_UNSUPPORTED_TOOLS.has(name)) {
      return {
        ok: false,
        output: `${name} is not yet supported on a remote SSH workspace. Use read_file, write_file, edit_file, list_dir, search, glob, or run_command — they run on the remote host.`,
        meta: 'remote-unsupported',
      }
    }

    switch (name) {
      // App-control (#42). These don't touch the workspace — they describe an
      // action the renderer applies (creates + navigation immediately; settings
      // via a confirm chip). The model gets a short acknowledgement back.
      case 'create_project': {
        const projectName = String(args.name ?? '').trim()
        if (!projectName) return { ok: false, output: 'A project name is required.' }
        const folder = args.folder ? String(args.folder).trim() : undefined
        if (folder) {
          if (!folder.startsWith('/')) return { ok: false, output: 'Project folder must be an absolute path (starting with "/"), or omitted.' }
          // The folder must exist and must not be the user's home directory — a
          // project writes AGENTS.md/CLAUDE.md into its cwd, so never let the model
          // point that at $HOME or a nonexistent path. Full validation (home-folder
          // guard incl. symlink resolution) runs again server-side on projects:create.
          try {
            const stat = await fs.stat(folder)
            if (!stat.isDirectory()) return { ok: false, output: `Project folder is not a directory: ${folder}` }
          } catch {
            return { ok: false, output: `Project folder does not exist: ${folder}` }
          }
          if (isHomeFolder(folder)) return { ok: false, output: 'Refusing to use your home folder as a project workspace. Pick a project subfolder instead.' }
        }
        return { ok: true, title: 'Create project', meta: projectName, output: `Creating project “${projectName}”${folder ? ` in ${folder}` : ''}.`, appAction: { kind: 'create_project', payload: { name: projectName, folder } } }
      }
      case 'create_room': {
        const roomName = String(args.name ?? '').trim() || 'New room'
        return { ok: true, title: 'Create room', meta: roomName, output: `Creating room “${roomName}”.`, appAction: { kind: 'create_room', payload: { name: roomName, projectId: args.projectId ? String(args.projectId) : undefined } } }
      }
      case 'create_kanban_ticket': {
        const ticketTitle = String(args.title ?? '').trim()
        if (!ticketTitle) return { ok: false, output: 'A ticket title is required.' }
        const TYPES = ['epic', 'task', 'subticket', 'bug'], PRIORITIES = ['low', 'medium', 'high', 'urgent']
        const type = args.type != null ? String(args.type) : undefined
        const priority = args.priority != null ? String(args.priority) : undefined
        if (type && !TYPES.includes(type)) return { ok: false, output: `Invalid ticket type "${type}". Use one of: ${TYPES.join(', ')}.` }
        if (priority && !PRIORITIES.includes(priority)) return { ok: false, output: `Invalid priority "${priority}". Use one of: ${PRIORITIES.join(', ')}.` }
        return { ok: true, title: 'Add kanban ticket', meta: ticketTitle, output: `Adding ticket “${ticketTitle}” to the board.`, appAction: { kind: 'create_kanban', payload: { title: ticketTitle, body: args.body ? String(args.body) : '', type, priority, projectId: args.projectId ? String(args.projectId) : undefined } } }
      }
      case 'open_view': {
        const VIEWS = ['home', 'projects', 'settings', 'plugins', 'automations', 'commandCentre', 'assistant']
        const view = String(args.view ?? '').trim()
        if (!VIEWS.includes(view)) return { ok: false, output: `Unknown view "${view}". Use one of: ${VIEWS.join(', ')}.` }
        return { ok: true, title: 'Open', meta: view, output: `Opening ${view}.`, appAction: { kind: 'open_view', payload: { view } } }
      }
      case 'change_setting': {
        const SETTING_VALUES: Record<string, string[]> = { theme: ['system', 'light', 'dark'], approvalMode: ['suggest', 'auto', 'readonly'], reasoningEffort: ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultrathink', 'ultracode', 'dynamic', '512'] }
        const key = String(args.key ?? '').trim()
        const value = args.value != null ? String(args.value).trim() : ''
        if (!SETTING_VALUES[key]) return { ok: false, output: `Unknown setting "${key}". Use one of: ${Object.keys(SETTING_VALUES).join(', ')}.` }
        if (!SETTING_VALUES[key].includes(value)) return { ok: false, output: `Invalid value "${value}" for ${key}. Use one of: ${SETTING_VALUES[key].join(', ')}.` }
        return { ok: true, title: 'Change setting', meta: key, output: `Proposed: set ${key} to ${value}. Waiting for you to confirm.`, appAction: { kind: 'change_setting', payload: { key, value }, state: 'pending' } }
      }
      case 'suggest_skill': {
        // Propose (never create) a reusable Skill or Command. The user reviews + accepts.
        const kind = String(args.kind ?? 'skill').trim() === 'command' ? 'command' : 'skill'
        const name = String(args.name ?? '').trim()
        const body = String(args.body ?? '').trim()
        if (!name || !body) return { ok: false, output: 'suggest_skill needs a name and a body.' }
        if (!ctx.fileSuggestion) return { ok: true, title: 'Suggest', meta: name, output: 'Suggestions are unavailable here.' }
        ctx.fileSuggestion({
          kind,
          name,
          body,
          description: args.description != null ? String(args.description) : '',
          argHint: args.argHint != null ? String(args.argHint) : undefined,
          rationale: args.rationale != null ? String(args.rationale) : '',
        })
        return { ok: true, title: kind === 'command' ? 'Suggest command' : 'Suggest skill', meta: name, output: `Filed a ${kind} suggestion "${name}". The user can review and save it from Skills & Commands — nothing was created.` }
      }
      case 'skills_list': {
        const list = listSkills(ctx.skills)
        if (!list.length) return { ok: true, title: 'Skills', output: 'No saved skills.' }
        return { ok: true, title: 'Skills', meta: `${list.length}`, output: list.map((s) => `- ${s.name}: ${s.description || '(no description)'}`).join('\n') }
      }
      case 'skill_view': {
        const skill = viewSkill(ctx.skills, String(args.name ?? ''))
        if (!skill) return { ok: false, output: `No enabled skill named "${args.name}". Use skills_list to see available skills.` }
        return { ok: true, title: 'Skill', meta: skill.name, output: renderSkillView(skill) }
      }
      case 'list_dir': {
        const dir = resolveWorkspacePath(ctx, args.path)
        const entries = await workspaceBackend(ctx).readdir(dir)
        const sorted = entries
          .filter((e) => !e.name.startsWith('.'))
          .sort((a, b) => Number(b.isDirectory) - Number(a.isDirectory) || a.name.localeCompare(b.name))
        const list = sorted.map((e) => (e.isDirectory ? `${e.name}/` : e.name)).join('\n')
        return { ok: true, output: list || '(empty)', title: relWorkspace(ctx, dir), path: relWorkspace(ctx, dir), meta: `${sorted.length} items` }
      }

      case 'save_artifact': {
        if (!ctx.registerArtifact || !ctx.currentThread) return { ok: false, output: 'Artifact registration is not available in this run context.', meta: 'unavailable' }
        const title = String(args.title ?? '').trim()
        if (!title) return { ok: false, output: 'save_artifact requires a title.', meta: 'invalid' }
        const hasPath = typeof args.path === 'string' && args.path.trim().length > 0
        const hasContent = typeof args.content === 'string' && args.content.length > 0
        if (!hasPath && !hasContent) return { ok: false, output: 'Provide the path of an existing file OR inline content.', meta: 'invalid' }
        if (hasPath && hasContent) return { ok: false, output: 'Provide a path OR content, not both.', meta: 'invalid' }
        if (hasPath && (ctx.workspace?.kind === 'ssh' || ctx.backend?.kind === 'ssh')) {
          return {
            ok: false,
            output: 'Path-backed artifacts from SSH workspaces cannot be previewed or exported safely yet. Save bounded inline content, or copy the output into a managed local artifact first.',
            meta: 'remote_unsupported',
          }
        }
        let filePath: string | null = null
        let sizeBytes: number | null = null
        if (hasPath) {
          // Same confinement as read_file: the path must resolve inside the workspace,
          // and the file must actually exist (an artifact is evidence, not a promise). Use
          // metadata for the size: reading the entire file merely to register it defeats the
          // bounded preview/export path and lets a huge artifact exhaust local or SSH memory.
          filePath = resolveWorkspacePath(ctx, args.path)
          try {
            const info = await workspaceBackend(ctx).stat(filePath)
            if (!info.isFile || !Number.isSafeInteger(info.size) || info.size < 0) throw new Error('not a regular file')
            sizeBytes = info.size
          } catch {
            return { ok: false, output: `Cannot inspect ${args.path} — the artifact must be an existing regular file.`, meta: 'missing' }
          }
        }
        const artifact = ctx.registerArtifact({
          title,
          kind: typeof args.kind === 'string' ? args.kind : null,
          path: filePath,
          inlineText: hasContent ? redactSecretValues(String(args.content), SECRET_MASK).text : null,
          sizeBytes,
          provenance: {
            threadId: ctx.currentThread.id,
            tool: 'save_artifact',
            workspace: ctx.workspace ?? { kind: 'local', path: ctx.cwd },
            sourceRef: typeof args.source_ref === 'string' && args.source_ref.trim() ? args.source_ref.trim() : null,
          },
        })
        return { ok: true, output: `Registered artifact "${artifact.title}" (${artifact.kind}).`, title: `Artifact: ${artifact.title}`, meta: artifact.kind }
      }

      case 'read_file': {
        const requestedPath = String(args.path ?? '')
        const managed = isManagedOutputRef(requestedPath)
        if (!managed) {
          const readBlock = blockedReadReason(requestedPath)
          if (readBlock) return { ok: false, output: `Blocked by TensorFold safety policy: ${readBlock}.`, meta: 'blocked' }
        }
        const file = managed ? requestedPath : resolveWorkspacePath(ctx, requestedPath)
        const offsetValue = Number(args.offset ?? 1)
        const limitValue = Number(args.limit ?? 2000)
        const offset = Math.max(1, Number.isFinite(offsetValue) ? Math.trunc(offsetValue) : 1)
        const limit = Math.max(1, Number.isFinite(limitValue) ? Math.trunc(limitValue) : 2000)
        const backend = managed ? null : workspaceBackend(ctx)
        if (backend?.readFileRange) {
          const range = await backend.readFileRange(file, { offset, limit, maxChars: READ_FILE_MAX_CHARS, signal: ctx.signal })
          const note = !range.bounded ? '' : range.longLine
            ? `\n[read_file bounded by TensorFold: line ${offset} alone is longer than the output limit; showing its first ${READ_FILE_MAX_CHARS.toLocaleString()} chars. Use the search tool to find what you need inside it.]`
            : `\n[read_file bounded by TensorFold: showing ${range.lines} lines within the output limit. Continue with offset=${range.nextOffset}, or use a smaller limit.]`
          const redacted = redactSecretValues(range.text + note, SECRET_MASK)
          const count = redacted.hits.length
          const label = relWorkspace(ctx, file)
          return {
            ok: true, output: commandOutputWithRedactionNotice(redacted.text, count), title: label, path: label,
            meta: `${range.totalLines ?? `at least ${offset + range.lines - 1}`} lines${range.bounded ? ' · bounded' : ''}${count ? ` · redacted=${count}` : ''}`,
          }
        }
        const raw = managed ? await ctx.readManagedOutput?.(requestedPath) : await backend!.readFile(file, { signal: ctx.signal })
        if (raw == null) return { ok: false, output: 'Managed output was not found or is no longer available.', title: requestedPath, path: requestedPath, meta: 'missing' }
        const allLines = raw.split('\n')
        const slice = allLines.slice(offset - 1, offset - 1 + limit)
        const numbered = slice.map((l, i) => `${String(offset + i).padStart(5)}\t${l}`).join('\n')
        // Char bound (VST-016): the line limit alone cannot bound a file with enormous lines
        // (minified bundles, single-line JSON). Cut at the last whole line under the budget —
        // or mid-line when even ONE line exceeds it — and say exactly how to continue.
        if (numbered.length > READ_FILE_MAX_CHARS) {
          const cutAt = numbered.lastIndexOf('\n', READ_FILE_MAX_CHARS)
          const kept = cutAt > 0 ? numbered.slice(0, cutAt) : numbered.slice(0, READ_FILE_MAX_CHARS)
          const shownLines = cutAt > 0 ? kept.split('\n').length : 1
          const note = cutAt > 0
            ? `[read_file bounded by TensorFold: this range is ${numbered.length.toLocaleString()} chars; showing ${shownLines} of the ${slice.length} lines requested (file has ${allLines.length} lines). Continue with offset=${offset + shownLines}, or use a smaller limit.]`
            : `[read_file bounded by TensorFold: line ${offset} alone is ${numbered.length.toLocaleString()}+ chars (file has ${allLines.length} lines); showing its first ${READ_FILE_MAX_CHARS.toLocaleString()} chars. Use the search tool to find what you need inside it.]`
          const label = managed ? requestedPath : relWorkspace(ctx, file)
          const redactedBounded = redactSecretValues(`${kept}\n${note}`, SECRET_MASK)
          const boundedCount = managed ? 0 : redactedBounded.hits.length
          return {
            ok: true,
            output: commandOutputWithRedactionNotice(redactedBounded.text, boundedCount),
            title: label,
            path: label,
            meta: `${allLines.length} lines · bounded${boundedCount ? ` · redacted=${boundedCount}` : ''}`,
          }
        }
        const label = managed ? requestedPath : relWorkspace(ctx, file)
        const redacted = redactSecretValues(numbered, SECRET_MASK)
        const count = managed ? 0 : redacted.hits.length
        return {
          ok: true,
          output: commandOutputWithRedactionNotice(redacted.text, count),
          title: label,
          path: label,
          meta: `${allLines.length} lines${count ? ` · redacted=${count}` : ''}`,
        }
      }

      case 'write_file': {
        const backend = workspaceBackend(ctx)
        const file = resolveWorkspacePath(ctx, args.path)
        const writeBlock = blockedWriteReason(relWorkspace(ctx, file))
        if (writeBlock) return { ok: false, output: `Blocked by TensorFold safety policy: ${writeBlock}.`, meta: 'blocked' }
        let before = ''
        let existed = false
        try {
          before = await backend.readFile(file, { signal: ctx.signal })
          existed = true
        } catch (error) {
          // A read failure for an existing path is not proof that the path is new.
          // Fail closed instead of replacing a permission error or raced entry.
          if (await backend.exists(file)) throw error
        }
        const newContent = args.content ?? ''
        // Syntax gate: never overwrite an EXISTING code file with something that no longer
        // parses (hand the error back so the model self-corrects). A brand-new file is only
        // flagged, not blocked — scaffolding can legitimately be written in pieces. Fail-open.
        let gateNote = ''
        if (isGatedCodePath(file) && newContent) {
          const syn = await checkEditedSyntax(file, newContent)
          if (!syn.ok) {
            if (existed) return { ok: false, output: syntaxGateMessage(relWorkspace(ctx, file), syn.error || 'parse error'), title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
            gateNote = `\n⚠ Heads up: this new file may have a syntax error — ${syn.error}`
          }
        }
        await backend.writeFile(file, newContent, existed
          ? { expectedSha256: workspaceContentSha256(before) }
          : { ifExists: 'error' })
        const diff = computeDiff(before, newContent)
        const { added, removed } = diffStat(diff)
        return {
          ok: true,
          output: `${existed ? `Updated ${relWorkspace(ctx, file)}` : `Created ${relWorkspace(ctx, file)}`}${gateNote}`,
          title: relWorkspace(ctx, file),
          path: relWorkspace(ctx, file),
          diff,
          changedPaths: [relWorkspace(ctx, file)],
          meta: `+${added} −${removed}`,
        }
      }

      case 'edit_file': {
        const backend = workspaceBackend(ctx)
        const file = resolveWorkspacePath(ctx, args.path)
        const writeBlock = blockedWriteReason(relWorkspace(ctx, file))
        if (writeBlock) return { ok: false, output: `Blocked by TensorFold safety policy: ${writeBlock}.`, meta: 'blocked' }
        const before = await backend.readFile(file, { signal: ctx.signal })
        const oldS = args.old_string ?? ''
        const count = oldS ? before.split(oldS).length - 1 : 0
        if (count === 0) return { ok: false, output: `String not found in ${relWorkspace(ctx, file)}.`, title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
        if (count > 1) return { ok: false, output: `String is not unique (${count} matches) in ${relWorkspace(ctx, file)}.`, title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
        const after = replaceLiteral(before, oldS, args.new_string ?? '')
        // Syntax gate: edit_file always modifies an existing file, so a broken result is a
        // mistake — block and report it rather than persisting unparseable code.
        if (isGatedCodePath(file)) {
          const syn = await checkEditedSyntax(file, after)
          if (!syn.ok) return { ok: false, output: syntaxGateMessage(relWorkspace(ctx, file), syn.error || 'parse error'), title: relWorkspace(ctx, file), path: relWorkspace(ctx, file) }
        }
        await backend.writeFile(file, after, { expectedSha256: workspaceContentSha256(before) })
        const diff = computeDiff(before, after)
        const { added, removed } = diffStat(diff)
        return { ok: true, output: `Edited ${relWorkspace(ctx, file)}`, title: relWorkspace(ctx, file), path: relWorkspace(ctx, file), diff, changedPaths: [relWorkspace(ctx, file)], meta: `+${added} −${removed}` }
      }

      case 'apply_patch': {
        const changes = typeof args.patch === 'string' && args.patch.trim()
          ? await stagePatchOperations(ctx.cwd, parseApplyPatchText(args.patch))
          : await stageReplacementHunks(ctx.cwd, replacementHunksFromArgs(args))
        // Syntax gate (all-or-nothing): if any hunk would leave an EXISTING code file
        // unparseable, write NONE of them and report — so a multi-file patch can't half-apply
        // a broken edit. New files added by the patch (no prior content) are not blocked.
        for (const ch of changes) {
          if (ch.kind === 'delete' || ch.kind === 'move' || !ch.before) continue
          if (!isGatedCodePath(ch.finalPath) || !ch.after) continue
          const syn = await checkEditedSyntax(ch.finalPath, ch.after)
          if (!syn.ok) return { ok: false, output: syntaxGateMessage(ch.finalPath, syn.error || 'parse error'), title: ch.finalPath, path: ch.finalPath }
        }
        await writeAppliedChanges(ctx.cwd, changes)
        const diff = combinedDiff(changes)
        const { added, removed } = diffStat(diff)
        const onePath = changes.length === 1 ? changes[0].finalPath : undefined
        return {
          ok: true,
          output: patchSummary(changes),
          title: changes.length === 1 ? onePath : 'Patch',
          path: onePath,
          diff,
          changedPaths: [...new Set(changes.flatMap((change) => [change.filePath, change.finalPath]).filter(Boolean))],
          meta: `${changes.length} file${changes.length === 1 ? '' : 's'} · +${added} −${removed}`,
        }
      }

      case 'web_search': {
        const query = String(args.query ?? '').trim()
        if (!query) return { ok: false, output: 'Search query is required.', title: 'Web search' }
        const rawLimit = Number(args.limit ?? 5)
        const limit = Math.max(1, Math.min(8, Number.isFinite(rawLimit) ? Math.floor(rawLimit) : 5))
        const url = new URL('https://news.google.com/rss/search')
        url.searchParams.set('q', query)
        url.searchParams.set('hl', 'en-US')
        url.searchParams.set('gl', 'US')
        url.searchParams.set('ceid', 'US:en')
        const fetched = await fetchText(url, 10_000)
        if (fetched.status < 200 || fetched.status >= 300) {
          return { ok: false, output: `Search failed: HTTP ${fetched.status} ${fetched.statusText}`.trim(), title: query, meta: 'web' }
        }
        const results = parseNewsRss(fetched.text, limit)
        const output = results.map((result, index) => [
          `${index + 1}. ${result.title}`,
          `   ${result.url}`,
          result.snippet ? `   ${result.snippet}` : '',
        ].filter(Boolean).join('\n')).join('\n\n')
        return { ok: true, output: output || 'No web results found.', title: query, meta: `${results.length} results` }
      }

      case 'web_fetch': {
        // assertFetchableUrl validates the host string AND DNS-resolves it, so a
        // public name that resolves to an internal/metadata IP is rejected before
        // we fetch. fetchText also re-checks every redirect hop.
        const url = await assertFetchableUrl(String(args.url ?? ''))
        const fetched = await fetchText(url, 10_000)
        const rawMax = Number(args.max_chars ?? 6000)
        const maxChars = Math.max(500, Math.min(12_000, Number.isFinite(rawMax) ? Math.floor(rawMax) : 6000))
        const body = fetched.contentType.includes('html') ? stripHtml(fetched.text) : fetched.text.replace(/\s+/g, ' ').trim()
        const title = fetched.contentType.includes('html')
          ? stripHtml(/<title[^>]*>([\s\S]*?)<\/title>/i.exec(fetched.text)?.[1] ?? url.hostname)
          : url.hostname
        const output = [
          `URL: ${url.href}`,
          `Status: ${fetched.status} ${fetched.statusText}`.trim(),
          `Content-Type: ${fetched.contentType || 'unknown'}`,
          '',
          body.slice(0, maxChars) || '(no readable text)',
        ].join('\n')
        return { ok: fetched.status >= 200 && fetched.status < 300, output, title, path: url.href, meta: 'web' }
      }

      case 'search': {
        const query: string = args.query ?? ''
        const base = resolveWorkspacePath(ctx, args.path)
        if (isRemote(ctx)) {
          const backend = workspaceBackend(ctx)
          // `cwd` is consumed by a remote shell, so lexical containment is not enough:
          // an in-root path may be a symlink to another tenant or `/etc`. Require the SSH
          // host's canonical proof before grep starts walking it.
          if (!backend.assertContained) throw new Error('Remote backend cannot verify workspace containment.')
          await backend.assertContained(base)
          const excludes = [...SKIP_DIRS].map((d) => `--exclude-dir=${shArg(d)}`).join(' ')
          const cmd = `grep -rInE -- ${shArg(query)} ${excludes} . 2>/dev/null | head -n 100`
          const res = await backend.exec(cmd, { cwd: base })
          // Drop hits from credential files: grep dumps line contents, so without this the
          // remote path would surface in-repo .env/credentials that read_file blocks (the
          // local walk already filters these; mirror it here).
          const hits = res.stdout.split('\n').filter(Boolean)
            .map((l) => l.replace(/^\.\//, ''))
            .filter((l) => !blockedReadReason(l.split(':', 1)[0]))
          return { ok: true, output: hits.length ? hits.join('\n') : 'No matches.', title: `"${query}"`, meta: `${hits.length} matches` }
        }
        const hits = (await boundedRegexSearch(base, query, SKIP_DIRS))
          .map((line) => line.replace(/^\.\//, ''))
          .filter((line) => !blockedReadReason(line.split(':', 1)[0]))
          .slice(0, 100)
          .map((line) => {
            const match = /^([^:]+):(\d+):(.*)$/.exec(line)
            return match ? `${rel(ctx.cwd, path.join(base, match[1]))}:${match[2]}: ${match[3].trim().slice(0, 200)}` : line.slice(0, 240)
          })
        return { ok: true, output: hits.length ? hits.join('\n') : 'No matches.', title: `"${query}"`, meta: `${hits.length} matches` }
      }

      case 'glob': {
        const pattern = validateRelativePattern(String(args.pattern ?? ''))
        const base = resolveWorkspacePath(ctx, args.path)
        const rawMax = Number(args.max_results ?? DEFAULT_GLOB_RESULTS)
        const max = Math.max(1, Math.min(MAX_GLOB_RESULTS, Number.isFinite(rawMax) ? Math.floor(rawMax) : DEFAULT_GLOB_RESULTS))
        if (isRemote(ctx)) {
          const backend = workspaceBackend(ctx)
          // `find` has the same remote-cwd symlink boundary as grep above.
          if (!backend.assertContained) throw new Error('Remote backend cannot verify workspace containment.')
          await backend.assertContained(base)
          const namePart = pattern.split('/').pop() || pattern
          const cmd = `find . -type f -name ${shArg(namePart)} -not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/dist/*' 2>/dev/null | sed 's|^\\./||' | head -n ${max}`
          const res = await backend.exec(cmd, { cwd: base })
          const remoteHits = res.stdout.split('\n').filter(Boolean).filter((p) => !blockedReadReason(p))
          return {
            ok: true,
            output: remoteHits.length ? remoteHits.join('\n') : 'No matches.',
            title: pattern,
            path: args.path ? relWorkspace(ctx, base) : undefined,
            meta: `${remoteHits.length}${remoteHits.length >= max ? '+' : ''} files`,
          }
        }
        const hits = await collectFiles(ctx.cwd, base, pattern, max)
        return {
          ok: true,
          output: hits.length ? hits.join('\n') : 'No matches.',
          title: pattern,
          path: args.path ? rel(ctx.cwd, base) : undefined,
          meta: `${hits.length}${hits.length >= max ? '+' : ''} files`,
        }
      }

      case 'retrieval_search': {
        const sensitiveAllowed = ctx.sensitiveRetrievalAllowed !== false
        const requestedKinds = Array.isArray(args.kinds) ? args.kinds : typeof args.kinds === 'string' ? args.kinds : undefined
        const effectiveKinds = sensitiveAllowed ? requestedKinds : filterNonSensitiveRetrievalKinds(requestedKinds)
        const index = await buildWorkspaceRetrievalIndex({
          cwd: ctx.cwd,
          path: typeof args.path === 'string' ? args.path : undefined,
          query: typeof args.query === 'string' ? args.query : undefined,
          kinds: effectiveKinds,
          maxCodeFiles: Number.isFinite(Number(args.max_code_files)) ? Number(args.max_code_files) : undefined,
          maxConnectorItems: Number.isFinite(Number(args.max_connector_items)) ? Number(args.max_connector_items) : undefined,
          emailSettings: sensitiveAllowed ? ctx.emailSettings : undefined,
          slackSettings: sensitiveAllowed ? ctx.slackSettings : undefined,
          calendarSettings: sensitiveAllowed ? ctx.calendarSettings : undefined,
          documentSettings: sensitiveAllowed ? ctx.documentSettings : undefined,
          browserSettings: sensitiveAllowed ? ctx.browserSettings : undefined,
          connectorEvents: sensitiveAllowed ? ctx.connectorEvents : undefined,
          threads: sensitiveAllowed ? ctx.threads : undefined,
          missions: sensitiveAllowed ? ctx.missions : undefined,
          rooms: sensitiveAllowed ? ctx.rooms : undefined,
          tickets: sensitiveAllowed ? ctx.tickets : undefined,
          projectMemory: sensitiveAllowed ? ctx.projectMemory : undefined,
        })
        const result = searchRetrievalIndex(index, {
          query: typeof args.query === 'string' ? args.query : undefined,
          kinds: effectiveKinds,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : undefined,
        })
        const output = formatRetrievalSearchResult(result)
        const dlpNotice = sensitiveAllowed ? '' : `Sensitive retrieval sources omitted (${ctx.sensitiveRetrievalReason || 'policy disabled connector, chat, document, and memory retrieval for this run'}). Only code and artifact records were indexed.\n\n`
        return {
          ok: true,
          output: dlpNotice + output,
          title: 'Retrieval search',
          meta: `${result.hits.length} hits · ${index.records.length} indexed${sensitiveAllowed ? '' : ' · sensitive sources omitted'}`,
        }
      }

      case 'session_search': {
        const options = {
          query: String(args.query ?? ''),
          projectId: typeof args.projectId === 'string' && args.projectId.trim() ? args.projectId.trim() : undefined,
          currentThreadId: typeof args.currentThreadId === 'string' && args.currentThreadId.trim() ? args.currentThreadId.trim() : undefined,
          includeCurrent: args.includeCurrent === true,
          includeArchived: args.includeArchived === true,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : undefined,
        }
        const result = ctx.searchSessions
          ? await ctx.searchSessions(options)
          : searchSessionThreads(ctx.threads ?? [], options)
        return {
          ok: true,
          output: formatSessionSearchResult(result),
          title: 'Session search',
          meta: `${result.matches.length} matches${result.omitted ? ` · ${result.omitted} omitted` : ''}`,
        }
      }

      case 'session_read': {
        const result = readSessionThread(ctx.threads ?? [], {
          threadId: String(args.threadId ?? ''),
          maxMessages: Number.isFinite(Number(args.maxMessages)) ? Number(args.maxMessages) : undefined,
        })
        return {
          ok: !!result.thread,
          output: formatSessionReadResult(result),
          title: 'Session read',
          meta: result.thread ? `${result.messages.length} messages${result.omittedBefore ? ` · ${result.omittedBefore} earlier omitted` : ''}` : 'not found',
          path: result.thread?.id,
        }
      }

      case 'assistant_doctor': {
        const report = buildAssistantDoctorReport({
          currentThread: ctx.currentThread,
          threads: ctx.threads,
          mcpTools: ctx.mcpTools,
          skills: ctx.skills,
          commands: ctx.commands,
          localScripts: ctx.localScripts,
          packagedPlugins: ctx.packagedPlugins,
          disabledCapabilityIds: ctx.disabledCapabilityIds,
        })
        return {
          ok: true,
          output: formatAssistantDoctorReport(report),
          title: 'Assistant doctor',
          meta: report.promptSnapshot.present ? `prompt ${report.promptSnapshot.fingerprint}` : 'prompt snapshot missing',
        }
      }

      case 'skill_curator': {
        const report = buildSkillCurationReport({
          skills: ctx.skills,
          commands: ctx.commands,
          pendingSuggestions: ctx.skillSuggestions,
          staleAfterDays: Number.isFinite(Number(args.stale_after_days)) ? Number(args.stale_after_days) : undefined,
        })
        return {
          ok: true,
          output: formatSkillCurationReport(report),
          title: 'Skill curator',
          meta: `${report.findings.length} findings`,
        }
      }

      case 'tool_search': {
        const catalog = buildDeferredToolCatalog(ctx.mcpTools ?? [])
        const matches = searchDeferredToolCatalog(catalog, {
          query: String(args.query ?? ''),
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : undefined,
        })
        const result = {
          query: String(args.query ?? '').trim(),
          matches,
          omitted: Math.max(0, catalog.tools.length - matches.length),
        }
        return {
          ok: true,
          output: formatToolCatalogSearch(result),
          title: 'Tool search',
          meta: `${matches.length} matches · ${catalog.tools.length} deferred`,
        }
      }

      case 'tool_describe': {
        const catalog = buildDeferredToolCatalog(ctx.mcpTools ?? [])
        const tool = findDeferredTool(catalog, String(args.name ?? ''))
        return {
          ok: !!tool,
          output: formatDeferredToolDescription(tool),
          title: 'Tool describe',
          meta: tool?.name ?? 'not found',
        }
      }

      case 'tool_call': {
        const name = String(args.name ?? '').trim()
        const catalog = buildDeferredToolCatalog(ctx.mcpTools ?? [])
        const tool = findDeferredTool(catalog, name)
        if (!tool) return { ok: false, output: `No deferred tool found named "${name}". Use tool_search first.`, title: 'Tool call' }
        if (!name.startsWith('mcp__')) return { ok: false, output: 'Only deferred MCP tools can be called through tool_call.', title: 'Tool call' }
        if (!ctx.callMcpTool) return { ok: false, output: 'MCP host is not configured.', title: 'Tool call' }
        const callArgs = args.arguments && typeof args.arguments === 'object' && !Array.isArray(args.arguments)
          ? args.arguments as Record<string, any>
          : {}
        const result = await ctx.callMcpTool(name, callArgs)
        return { ok: result.ok, output: result.output, title: result.title ?? 'Tool call', meta: result.meta ?? name }
      }

      case 'ast_grep_search': {
        const result = await structuralSearch(ctx.cwd, {
          pattern: String(args.pattern ?? ''),
          path: typeof args.path === 'string' ? args.path : undefined,
          maxMatches: Number.isFinite(Number(args.max_matches)) ? Number(args.max_matches) : undefined,
        })
        return {
          ok: true,
          output: formatStructuralSearch(result),
          title: 'Structural search',
          meta: `${result.matches.length}${result.truncated ? '+' : ''} matches`,
        }
      }

      case 'ast_grep_rewrite_preview': {
        const result = await rewritePreview(ctx.cwd, {
          pattern: String(args.pattern ?? ''),
          rewrite: String(args.rewrite ?? ''),
          path: typeof args.path === 'string' ? args.path : undefined,
          maxReplacements: Number.isFinite(Number(args.max_replacements)) ? Number(args.max_replacements) : undefined,
        })
        return {
          ok: true,
          output: formatRewritePreview(result),
          title: 'Rewrite preview',
          diff: result.diff,
          meta: `${result.files.length} files · +${result.added} −${result.removed} · not applied`,
        }
      }

      case 'semgrep_scan': {
        const result = await semgrepCompatibleScan(ctx.cwd, {
          path: typeof args.path === 'string' ? args.path : undefined,
          autofix: Boolean(args.autofix),
          maxFindings: Number.isFinite(Number(args.max_findings)) ? Number(args.max_findings) : undefined,
        })
        const blocking = result.findings.filter((finding) => finding.blocking).length
        return {
          ok: blocking === 0,
          output: formatSecurityScan(result),
          title: 'Semgrep scan',
          diff: result.diff,
          meta: `${result.findings.length} findings · ${blocking} blocking · ${result.proposedFixes.length} fixes`,
        }
      }

      case 'codebase_map': {
        const result = await buildCodebaseMap(ctx.cwd, {
          path: typeof args.path === 'string' ? args.path : undefined,
          maxFiles: Number.isFinite(Number(args.max_files)) ? Number(args.max_files) : undefined,
        })
        return {
          ok: true,
          output: formatCodebaseMap(result),
          title: 'Codebase map',
          meta: `${result.filesScanned} files · ${result.imports.length} imports · ${result.symbols.length} symbols`,
        }
      }

      case 'symbol_index': {
        const result = await buildSymbolIndex(ctx.cwd, {
          path: typeof args.path === 'string' ? args.path : undefined,
          query: typeof args.query === 'string' ? args.query : undefined,
          maxFiles: Number.isFinite(Number(args.max_files)) ? Number(args.max_files) : undefined,
          maxSymbols: Number.isFinite(Number(args.max_symbols)) ? Number(args.max_symbols) : undefined,
        })
        return {
          ok: true,
          output: formatSymbolIndex(result),
          title: 'Symbol index',
          meta: `${result.filesScanned} files · ${result.symbols.length}${result.truncated ? '+' : ''} symbols`,
        }
      }

      case 'safe_refactor_preview': {
        const result = await safeRefactorPreview(ctx.cwd, {
          pattern: String(args.pattern ?? ''),
          rewrite: String(args.rewrite ?? ''),
          path: typeof args.path === 'string' ? args.path : undefined,
          maxReplacements: Number.isFinite(Number(args.max_replacements)) ? Number(args.max_replacements) : undefined,
        })
        const blocking = result.scan.findings.filter((finding) => finding.blocking).length
        return {
          ok: blocking === 0,
          output: formatSafeRefactorPreview(result),
          title: 'Safe refactor preview',
          diff: result.rewrite.diff,
          meta: `${result.search.matches.length} matches · ${result.rewrite.files.length} files · ${blocking} blocking · not applied`,
        }
      }

      case 'email_list': {
        if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
        const { messages } = await scanEmailMessages(ctx.emailSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          unreadOnly: Boolean(args.unreadOnly),
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
        })
        const lines = messages.map((message) => {
          const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
          return `- [${emailSourceId(message)}] ${message.id} ${message.unread ? '[unread]' : '[read]'} ${when} | ${message.from || 'Unknown'} | ${message.subject || '(no subject)'} | ${message.snippet}`
        })
        return { ok: true, output: lines.length ? lines.join('\n') : 'No email messages matched.\n\nSources: none.', title: 'Email list', meta: `${messages.length} messages`, sources: messages.map(emailSourceRef) }
      }

      case 'email_read': {
        if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
        const message = await readEmailMessage(ctx.emailSettings, String(args.id || ''))
        const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
        const header = [
          `ID: ${message.id}`,
          `Source ID: ${emailSourceId(message)}`,
          `Source: ${message.sourceLabel}`,
          `From: ${message.from || 'Unknown'}`,
          `To: ${message.to || '(not listed)'}`,
          `Date: ${when}`,
          `Subject: ${message.subject || '(no subject)'}`,
        ].join('\n')
        return { ok: true, output: `${header}\n\n${message.body}`, title: message.subject || 'Email', path: message.id, meta: `${message.size} bytes`, sources: [emailSourceRef(message)] }
      }

      case 'email_thread': {
        if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
        const thread = await readEmailThread(ctx.emailSettings, String(args.id || ''))
        const parts = thread.messages.map((message, index) => {
          const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
          return [
            `#${index + 1} [${emailSourceId(message)}] ${message.id}${message.unread ? ' [unread]' : ''}`,
            `From: ${message.from || 'Unknown'}`,
            `To: ${message.to || '(not listed)'}`,
            `Date: ${when}`,
            `Subject: ${message.subject || '(no subject)'}`,
            '',
            message.body || message.snippet,
          ].join('\n')
        })
        const sources = thread.messages.map(emailSourceRef)
        return { ok: true, output: `${thread.summary}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}\n\n${parts.join('\n\n---\n\n')}`, title: thread.root.subject || 'Email thread', path: thread.root.id, meta: `${thread.messages.length} messages`, sources }
      }

      case 'email_summary': {
        if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
        const result = await summarizeEmail(ctx.emailSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          unreadOnly: Boolean(args.unreadOnly),
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 50,
        })
        return { ok: true, output: result.summary, title: 'Email summary', sources: result.sources, meta: `${result.sources.length} sources` }
      }

      case 'email_compose': {
        const input: EmailDraftInput = {
          mode: args.mode === 'send' ? 'send' : 'draft',
          to: String(args.to || '').trim(),
          subject: String(args.subject || '').trim(),
          body: String(args.body || '').trim(),
          threadKey: typeof args.threadKey === 'string' ? args.threadKey : undefined,
          inReplyTo: typeof args.inReplyTo === 'string' ? args.inReplyTo : undefined,
        }
        if (!input.to || !input.subject || !input.body) return { ok: false, output: 'Email compose requires to, subject, and body.' }
        let sources: ConnectorSourceRef[] = []
        if (ctx.emailSettings && (input.threadKey || input.inReplyTo)) {
          const thread = await readEmailThread(ctx.emailSettings, input.threadKey || input.inReplyTo || '')
          sources = thread.messages.map(emailSourceRef)
          input.threadKey = `email:${thread.root.id}`
          input.inReplyTo = thread.root.id
        }
        const event = ctx.recordEmailDraft ? await ctx.recordEmailDraft(input, sources) : emailDraftEvent(input, Date.now(), sources)
        const sourceLines = event.sources?.length ? `\n\nSources:\n${event.sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}` : ''
        const status = event.requiresApproval ? 'waiting for approval' : 'logged locally'
        return {
          ok: true,
          output: `${event.action} ${status} for ${input.to}.${sourceLines}`,
          title: event.action === 'reply_requested' ? 'Email reply approval' : input.mode === 'send' ? 'Email send request' : event.action === 'reply_draft_created' ? 'Email reply draft' : 'Email draft',
          path: event.threadKey,
          meta: event.approvalId ? `approval ${event.approvalId}` : event.action,
          sources: event.sources ?? sources,
        }
      }

      case 'email_resolve': {
        if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
        const thread = await readEmailThread(ctx.emailSettings, String(args.id || ''))
        const event = emailThreadResolvedEvent({ id: String(args.id || ''), note: typeof args.note === 'string' ? args.note : undefined }, thread)
        const recorded = ctx.recordEmailEvent ? await ctx.recordEmailEvent(event) : event
        const sources = recorded.sources ?? thread.messages.map(emailSourceRef)
        return {
          ok: true,
          output: `Email thread marked resolved locally: ${thread.summary}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
          title: 'Email thread resolved',
          path: recorded.threadKey,
          meta: `${thread.messages.length} messages`,
          sources,
        }
      }

      case 'message_list': {
        if (!ctx.slackSettings) return { ok: false, output: 'Slack connector is not configured.' }
        const { messages } = await scanSlackMessages(ctx.slackSettings, {
          channel: typeof args.channel === 'string' ? args.channel : undefined,
          query: typeof args.query === 'string' ? args.query : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
        })
        const lines = messages.map((message) => {
          const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
          return `- [${slackSourceId(message)}] #${message.channelName} ${when} | ${message.senderName || 'Unknown'} | ${message.snippet}`
        })
        return { ok: true, output: lines.length ? lines.join('\n') : 'No Slack messages matched.\n\nSources: none.', title: 'Message list', meta: `${messages.length} messages`, sources: messages.map(slackSourceRef) }
      }

      case 'message_read': {
        if (!ctx.slackSettings) return { ok: false, output: 'Slack connector is not configured.' }
        const message = await readSlackMessage(ctx.slackSettings, String(args.id || ''))
        const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
        const source = slackSourceRef(message)
        return {
          ok: true,
          output: [
            `ID: ${message.id}`,
            `Source ID: ${source.id}`,
            `Source: ${message.sourceLabel}`,
            `Channel: #${message.channelName}`,
            `Sender: ${message.senderName || 'Unknown'}`,
            `Date: ${when}`,
            message.threadTs ? `Thread: ${message.threadTs}` : '',
            '',
            message.text,
            '',
            'Sources:',
            `- ${source.id} | ${source.title}`,
          ].filter((line) => line !== '').join('\n'),
          title: `#${message.channelName}`,
          path: message.id,
          meta: message.senderName || 'Slack',
          sources: [source],
        }
      }

      case 'message_summary': {
        if (!ctx.slackSettings) return { ok: false, output: 'Slack connector is not configured.' }
        const result = await summarizeSlack(ctx.slackSettings, {
          channel: typeof args.channel === 'string' ? args.channel : undefined,
          query: typeof args.query === 'string' ? args.query : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 50,
        })
        return { ok: true, output: result.summary, title: 'Message summary', sources: result.sources, meta: `${result.sources.length} sources` }
      }

      case 'message_compose': {
        const connectorId = normalizePersonalMessageConnector(args.connectorId)
        const target = String(args.target || '').trim()
        const body = String(args.body || '').trim()
        const threadKey = typeof args.threadKey === 'string' ? args.threadKey.trim() : ''
        if (!connectorId) return { ok: false, output: 'message_compose requires connectorId to be telegram, whatsapp, or imessage.' }
        if (!target || !body) return { ok: false, output: 'message_compose requires target and body.' }
        const event = personalMessageDraftEvent({ connectorId, target, body, threadKey: threadKey || undefined })
        const recorded = ctx.recordMessageEvent ? await ctx.recordMessageEvent(event) : event
        return {
          ok: true,
          output: `${recorded.action} waiting for approval for ${connectorId}:${target}. No external message was sent.`,
          title: `${connectorName(connectorId)} message approval`,
          path: recorded.threadKey,
          meta: recorded.approvalId ? `approval ${recorded.approvalId}` : recorded.action,
        }
      }

      case 'calendar_list': {
        if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
        const { events } = await scanCalendarEvents(ctx.calendarSettings, {
          day: typeof args.day === 'string' ? args.day : undefined,
          from: typeof args.from === 'string' || typeof args.from === 'number' ? args.from : undefined,
          to: typeof args.to === 'string' || typeof args.to === 'number' ? args.to : undefined,
          query: typeof args.query === 'string' ? args.query : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
        })
        const lines = events.map((event) => {
          const start = event.startAt ? new Date(event.startAt).toLocaleString() : 'no start'
          const end = event.endAt ? new Date(event.endAt).toLocaleString() : ''
          return `- [${calendarSourceId(event)}] ${event.id} | ${start}${end ? ` - ${end}` : ''} | ${event.title || '(untitled)'}${event.location ? ` | ${event.location}` : ''}`
        })
        return { ok: true, output: lines.length ? lines.join('\n') : 'No calendar events matched.\n\nSources: none.', title: 'Calendar list', meta: `${events.length} events`, sources: events.map(calendarSourceRef) }
      }

      case 'calendar_read': {
        if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
        const event = await readCalendarEvent(ctx.calendarSettings, String(args.id || ''))
        const start = event.startAt ? new Date(event.startAt).toLocaleString() : 'no start'
        const end = event.endAt ? new Date(event.endAt).toLocaleString() : '(not set)'
        const lines = [
          `ID: ${event.id}`,
          `Source ID: ${calendarSourceId(event)}`,
          `Source: ${event.sourceLabel}`,
          `When: ${event.allDay ? 'all day ' : ''}${start} - ${end}`,
          `Title: ${event.title || '(untitled)'}`,
          event.location ? `Location: ${event.location}` : '',
          event.organizer ? `Organizer: ${event.organizer}` : '',
          event.attendees.length ? `Attendees: ${event.attendees.join(', ')}` : '',
          event.status ? `Status: ${event.status}` : '',
          '',
          event.description || '(no description)',
        ].filter((line) => line !== '').join('\n')
        return { ok: true, output: lines, title: event.title || 'Calendar event', path: event.id, meta: event.allDay ? 'all day' : 'timed event', sources: [calendarSourceRef(event)] }
      }

      case 'calendar_summary': {
        if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
        const result = await summarizeCalendarDay(ctx.calendarSettings, {
          day: typeof args.day === 'string' ? args.day : undefined,
          query: typeof args.query === 'string' ? args.query : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 40,
        })
        return { ok: true, output: result.summary, title: 'Calendar day summary', meta: `${result.sources.length} sources`, sources: result.sources }
      }

      case 'calendar_create': {
        if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
        const enabledSources = ctx.calendarSettings.sources.filter((source) => source.enabled)
        const source = typeof args.sourceId === 'string'
          ? enabledSources.find((item) => item.id === args.sourceId)
          : enabledSources[0]
        if (!source) return { ok: false, output: 'No enabled calendar source is available for the create request.' }
        const input: CalendarMutationInput = {
          mode: 'create',
          sourceId: source.id,
          title: String(args.title || '').trim(),
          startAt: typeof args.startAt === 'string' || typeof args.startAt === 'number' ? args.startAt : null,
          endAt: typeof args.endAt === 'string' || typeof args.endAt === 'number' ? args.endAt : null,
          allDay: typeof args.allDay === 'boolean' ? args.allDay : undefined,
          location: typeof args.location === 'string' ? args.location : undefined,
          description: typeof args.description === 'string' ? args.description : undefined,
          organizer: typeof args.organizer === 'string' ? args.organizer : undefined,
          attendees: parseStringList(args.attendees),
        }
        if (!input.title || !input.startAt) return { ok: false, output: 'calendar_create requires title and startAt.' }
        const event = calendarMutationEvent(input)
        const recorded = ctx.recordCalendarEvent ? await ctx.recordCalendarEvent(event) : event
        return {
          ok: true,
          output: `${recorded.action} waiting for approval. Calendar file is not modified until approval.`,
          title: 'Calendar create approval',
          path: recorded.threadKey,
          meta: recorded.approvalId ? `approval ${recorded.approvalId}` : recorded.action,
          sources: recorded.sources,
        }
      }

      case 'calendar_update': {
        if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
        const id = String(args.id || '').trim()
        if (!id) return { ok: false, output: 'calendar_update requires id.' }
        const existing = await readCalendarEvent(ctx.calendarSettings, id)
        const input: CalendarMutationInput = {
          mode: 'update',
          id,
          sourceId: existing.sourceId,
          title: typeof args.title === 'string' && args.title.trim() ? args.title.trim() : existing.title,
          startAt: typeof args.startAt === 'string' || typeof args.startAt === 'number' ? args.startAt : existing.startAt,
          endAt: typeof args.endAt === 'string' || typeof args.endAt === 'number' ? args.endAt : existing.endAt,
          allDay: typeof args.allDay === 'boolean' ? args.allDay : existing.allDay,
          location: typeof args.location === 'string' ? args.location : existing.location,
          description: typeof args.description === 'string' ? args.description : existing.description,
          organizer: typeof args.organizer === 'string' ? args.organizer : existing.organizer,
          attendees: parseStringList(args.attendees) ?? existing.attendees,
        }
        const event = calendarMutationEvent(input, Date.now(), existing)
        const recorded = ctx.recordCalendarEvent ? await ctx.recordCalendarEvent(event) : event
        const sources = recorded.sources ?? [calendarSourceRef(existing)]
        return {
          ok: true,
          output: `${recorded.action} waiting for approval. Original event cited below; calendar file is not modified until approval.\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
          title: 'Calendar update approval',
          path: recorded.threadKey,
          meta: recorded.approvalId ? `approval ${recorded.approvalId}` : recorded.action,
          sources,
        }
      }

      case 'document_list': {
        if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
        const { documents } = await scanDocumentItems(ctx.documentSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          path: typeof args.path === 'string' ? args.path : undefined,
          extension: typeof args.extension === 'string' ? args.extension : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
        })
        const lines = documents.map((doc) => `- [${documentSourceId(doc)}] ${doc.relativePath} | ${doc.title || '(untitled)'} | ${doc.excerpt}`)
        return { ok: true, output: lines.length ? lines.join('\n') : 'No documents matched.\n\nSources: none.', title: 'Document list', meta: `${documents.length} documents`, sources: documents.map(documentSourceRef) }
      }

      case 'document_read': {
        if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
        const doc = await readDocumentItem(ctx.documentSettings, String(args.id || ''))
        const source = documentSourceRef(doc)
        return {
          ok: true,
          output: [
            `ID: ${doc.id}`,
            `Source ID: ${source.id}`,
            `Source: ${doc.sourceLabel}`,
            `Path: ${doc.relativePath}`,
            `Modified: ${doc.modifiedAt ? new Date(doc.modifiedAt).toLocaleString() : 'unknown'}`,
            '',
            doc.content,
            '',
            'Sources:',
            `- ${source.id} | ${source.title}`,
          ].join('\n'),
          title: doc.title || doc.relativePath,
          path: doc.relativePath,
          meta: `${Math.ceil(doc.size / 1024)} KB`,
          sources: [source],
        }
      }

      case 'document_summary': {
        if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
        const result = await summarizeDocuments(ctx.documentSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          path: typeof args.path === 'string' ? args.path : undefined,
          extension: typeof args.extension === 'string' ? args.extension : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 12,
        })
        return { ok: true, output: result.summary, title: 'Document summary', meta: `${result.sources.length} sources`, sources: result.sources }
      }

      case 'table_list': {
        if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
        const { tables } = await scanTableItems(ctx.documentSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          path: typeof args.path === 'string' ? args.path : undefined,
          extension: typeof args.extension === 'string' ? args.extension : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
        })
        const lines = tables.map((table) => `- [${documentSourceId(table)}] ${table.relativePath} | ${table.rowCount} rows x ${table.columnCount} columns | ${table.columns.join(', ')}`)
        return { ok: true, output: lines.length ? lines.join('\n') : 'No tables matched.\n\nSources: none.', title: 'Table list', meta: `${tables.length} tables`, sources: tables.map(documentSourceRef) }
      }

      case 'table_read': {
        if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
        const table = await readTableItem(ctx.documentSettings, String(args.id || ''))
        const source = documentSourceRef(table)
        const limit = Math.max(1, Math.min(Number(args.limit) || 20, 100))
        return {
          ok: true,
          output: [
            `ID: ${table.id}`,
            `Source ID: ${source.id}`,
            `Source: ${table.sourceLabel}`,
            `Path: ${table.relativePath}`,
            `Shape: ${table.rowCount} rows x ${table.columnCount} columns`,
            table.numericColumns.length ? `Numeric columns: ${table.numericColumns.map((col) => `${col.name} avg ${formatToolNumber(col.average)}`).join('; ')}` : 'Numeric columns: none detected',
            '',
            formatMarkdownTable(table.columns, table.rows.slice(0, limit)),
            '',
            'Sources:',
            `- ${source.id} | ${source.title}`,
          ].join('\n'),
          title: table.title || table.relativePath,
          path: table.relativePath,
          meta: `${table.rowCount} rows x ${table.columnCount} cols`,
          sources: [source],
        }
      }

      case 'table_summary': {
        if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
        const result = await summarizeTables(ctx.documentSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          path: typeof args.path === 'string' ? args.path : undefined,
          extension: typeof args.extension === 'string' ? args.extension : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 12,
        })
        return { ok: true, output: result.summary, title: 'Table summary', meta: `${result.sources.length} sources`, sources: result.sources }
      }

      case 'browser_list': {
        if (!ctx.browserSettings) return { ok: false, output: 'Browser connector is not configured.' }
        const { items } = await scanBrowserItems(ctx.browserSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
        })
        const lines = items.map((item) => {
          const when = item.capturedAt ? new Date(item.capturedAt).toLocaleString() : 'unknown'
          return `- [${browserSourceId(item)}] ${item.title || item.url} | ${when} | ${item.url}${item.excerpt ? ` | ${item.excerpt}` : ''}`
        })
        return { ok: true, output: lines.length ? lines.join('\n') : 'No browser sources matched.\n\nSources: none.', title: 'Browser list', meta: `${items.length} sources`, sources: items.map(browserSourceRef) }
      }

      case 'browser_read': {
        if (!ctx.browserSettings) return { ok: false, output: 'Browser connector is not configured.' }
        const item = await readBrowserItem(ctx.browserSettings, String(args.id || ''))
        const source = browserSourceRef(item)
        return {
          ok: true,
          output: item.content,
          title: item.title || item.url,
          path: item.url,
          meta: item.kind === 'current_tab' ? 'current tab' : 'history export',
          sources: [source],
        }
      }

      case 'browser_summary': {
        if (!ctx.browserSettings) return { ok: false, output: 'Browser connector is not configured.' }
        const result = await summarizeBrowser(ctx.browserSettings, {
          query: typeof args.query === 'string' ? args.query : undefined,
          limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 12,
        })
        return { ok: true, output: result.summary, title: 'Browser summary', meta: `${result.sources.length} sources`, sources: result.sources }
      }

      case 'browser_automation': {
        const input = normalizeBrowserAutomationInput(args)
        if (!ctx.runBrowserAutomation) return { ok: false, output: 'Browser automation runner is not available in this environment.' }
        const result = await ctx.runBrowserAutomation(input)
        return {
          ok: result.ok,
          output: formatBrowserAutomationResult(result),
          title: `Browser ${result.action}`,
          path: result.screenshot?.path,
          meta: `session=${result.sessionId}; tab=${result.tabId}${result.loadState ? `; load=${result.loadState}` : ''}`,
        }
      }

      case 'controller_action': {
        const input = normalizeControllerActionInput(args, { ownAppName: 'TensorFold' })
        if (!ctx.runControllerAction) return { ok: false, output: 'Controller action runner is not available in this environment.' }
        const result = await ctx.runControllerAction(input)
        return {
          ok: result.ok,
          output: formatControllerActionResult(result),
          title: `Controller ${result.action}`,
          meta: `${result.appName} · ${result.target}`,
        }
      }

      case 'computer_use': {
        const input = normalizeComputerUseInput(args)
        if (!ctx.runComputerUse) return { ok: false, output: 'Computer use is not available in this environment.' }
        const result = await ctx.runComputerUse(input)
        return { ok: result.ok, output: result.output, title: result.title, meta: result.meta }
      }

      case 'phone_observe':
      case 'phone_control': {
        if (!ctx.phoneControlAvailable || !ctx.runPhoneControl) return { ok: false, title: 'Phone control', output: 'Open Phone in this chat, connect your device, select a local model and choose Allow this chat first.' }
        const input: PhoneToolInput = name === 'phone_observe' ? { action: 'observe', ...(typeof args.sessionId === 'string' ? { sessionId: args.sessionId } : {}) } : normalizePhoneAction(args)
        return ctx.runPhoneControl(input)
      }

      case 'launch_subagent': {
        if (!ctx.launchSubAgent) return { ok: false, output: 'Sub-agent launcher is not available in this run.' }
        const task = String(args.task ?? '').trim()
        if (!task) return { ok: false, output: 'launch_subagent requires a task.' }
        const launched = await ctx.launchSubAgent({
          task,
          name: typeof args.name === 'string' ? args.name : null,
          model: typeof args.model === 'string' ? args.model : null,
          providerId: typeof args.providerId === 'string' ? args.providerId : null,
          accountId: typeof args.accountId === 'string' ? args.accountId : null,
          agent: typeof args.agent === 'string' ? args.agent : null,
          background: args.background === true,
        })
        return {
          ok: launched.agent.status !== 'failed' && launched.agent.status !== 'cancelled',
          output: launched.output,
          title: launched.agent.name,
          path: launched.agent.parent.label,
          meta: `sub-agent ${launched.agent.status}`,
          subAgent: launched.agent,
        }
      }

      case 'launch_subagents': {
        if (!ctx.launchSubAgent) return { ok: false, output: 'Sub-agent launcher is not available in this run.' }
        const launchSubAgent = ctx.launchSubAgent
        const rawTasks = Array.isArray(args.tasks) ? args.tasks : []
        const tasks = rawTasks
          .map((item) => typeof item === 'string'
            ? { task: item, name: '', model: '', providerId: '', accountId: '', agent: '', background: undefined as boolean | undefined }
            : {
              task: String(item?.task ?? '').trim(),
              name: String(item?.name ?? '').trim(),
              model: String(item?.model ?? '').trim(),
              providerId: String(item?.providerId ?? '').trim(),
              accountId: String(item?.accountId ?? '').trim(),
              agent: typeof item?.agent === 'string' ? item.agent.trim() : (typeof args.agent === 'string' ? args.agent.trim() : ''),
              background: typeof item?.background === 'boolean' ? item.background : undefined,
            })
          .filter((item) => item.task)
          .slice(0, 3)
        if (!tasks.length) return { ok: false, output: 'launch_subagents requires at least one task.' }

        const results = await Promise.all(tasks.map(async (task, index) => {
          try {
            const launched = await launchSubAgent({
              task: task.task,
              name: task.name || null,
              model: task.model || null,
              providerId: task.providerId || null,
              accountId: task.accountId || null,
              agent: task.agent || null,
              background: task.background ?? (args.background === true),
            })
            const label = launched.agent.name || task.name || `Sub-agent ${index + 1}`
            return {
              ok: launched.agent.status !== 'failed' && launched.agent.status !== 'cancelled',
              output: `${index + 1}. ${label}: ${launched.output}`,
            }
          } catch (error: any) {
            return { ok: false, output: `${index + 1}. ${task.name || `Sub-agent ${index + 1}`}: failed: ${error?.message ?? error}` }
          }
        }))
        const omitted = rawTasks.length > tasks.length ? `\n\n${rawTasks.length - tasks.length} extra task(s) were not launched; batch limit is 3.` : ''
        return {
          ok: results.every((result) => result.ok),
          output: results.map((result) => result.output).join('\n') + omitted,
          title: 'Sub-agent batch',
          meta: `${tasks.length} launched${rawTasks.length > tasks.length ? ` · ${rawTasks.length - tasks.length} omitted` : ''}`,
        }
      }

      case 'subagent_status': {
        const ids = Array.isArray(args.agent_ids)
          ? new Set(args.agent_ids.map((value: unknown) => String(value)).filter(Boolean))
          : null
        const currentThreadId = ctx.currentThread?.id
        const agents = (ctx.listSubAgents?.() ?? [])
          .filter((agent) => ids?.size ? ids.has(agent.id) : !currentThreadId || agent.threadId === currentThreadId)
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .slice(0, 30)
        if (!agents.length) return { ok: true, output: 'No matching sub-agents.', title: 'Sub-agent status', meta: '0 agents' }
        const output = agents.map((agent, index) => {
          const unread = agent.messages.filter((message) =>
            (message.audience === 'parent' || message.audience === 'both') && message.readByParentAt === null,
          )
          const questions = unread.filter((message) => message.kind === 'question').length
          return [
          `${index + 1}. ${agent.name} [${agent.status}]`,
          `   id: ${agent.id}`,
          `   task: ${agent.task}`,
          unread.length ? `   attention: ${unread.length} unread${questions ? ` · ${questions} question${questions === 1 ? '' : 's'}` : ''}` : '',
          agent.result ? `   result: ${agent.result.slice(0, 3000)}` : '',
          agent.error ? `   error: ${agent.error.slice(0, 1000)}` : '',
          ].filter(Boolean).join('\n')
        }).join('\n\n')
        return { ok: true, output, title: 'Sub-agent status', meta: `${agents.length} agents` }
      }

      case 'subagent_send': {
        if (!ctx.sendSubAgentMessage) return { ok: false, output: 'Sub-agent mailbox is not available in this run.' }
        const agentId = String(args.agent_id ?? ctx.currentSubAgentId ?? '').trim()
        const text = String(args.text ?? '').trim()
        if (!agentId || !text) return { ok: false, output: 'subagent_send requires an agent id and message text.' }
        if (ctx.currentSubAgentId && agentId !== ctx.currentSubAgentId) {
          return { ok: false, output: 'A child sub-agent may only write to its own mailbox.' }
        }
        const kind = typeof args.kind === 'string' ? args.kind as SubAgentMessageKind : undefined
        const message = ctx.sendSubAgentMessage({
          agentId,
          text,
          kind,
          replyToId: typeof args.reply_to_id === 'string' ? args.reply_to_id : null,
        })
        return {
          ok: true,
          output: `Mailbox ${message.kind} sent. id=${message.id}`,
          title: 'Sub-agent message',
          path: agentId,
          meta: message.kind,
        }
      }

      case 'subagent_inbox': {
        if (!ctx.readSubAgentMessages) return { ok: false, output: 'Sub-agent mailbox is not available in this run.' }
        const agentId = String(args.agent_id ?? ctx.currentSubAgentId ?? '').trim()
        if (!agentId) return { ok: false, output: 'subagent_inbox requires an agent id.' }
        if (ctx.currentSubAgentId && agentId !== ctx.currentSubAgentId) {
          return { ok: false, output: 'A child sub-agent may only read its own mailbox.' }
        }
        const messages = ctx.readSubAgentMessages({
          agentId,
          unreadOnly: args.unread_only !== false,
          markRead: args.mark_read !== false,
        })
        const output = messages.length
          ? messages.map((message) => [
            `[${message.id}] ${message.kind} from ${message.sender}${message.replyToId ? ` (reply to ${message.replyToId})` : ''}`,
            message.text,
          ].join('\n')).join('\n\n')
          : 'No matching mailbox messages.'
        return { ok: true, output, title: 'Sub-agent inbox', path: agentId, meta: `${messages.length} messages` }
      }

      case 'visualize': {
        const visualization = normalizeVisualizationSpec(args)
        const counts = [
          visualization.nodes?.length ? `${visualization.nodes.length} nodes` : '',
          visualization.bars?.length ? `${visualization.bars.length} bars` : '',
          visualization.metrics?.length ? `${visualization.metrics.length} metrics` : '',
          visualization.events?.length ? `${visualization.events.length} events` : '',
        ].filter(Boolean).join(' · ')
        return {
          ok: true,
          output: `Rendered ${visualization.kind} visualization "${visualization.title}".`,
          title: visualization.title,
          visualization,
          meta: counts || visualization.kind,
        }
      }

      case 'run_command': {
        const command: string = args.command ?? ''
        const blocked = blockedCommandReason(command)
        if (blocked) return { ok: false, output: `Blocked by TensorFold safety policy: ${blocked}.`, command, meta: 'blocked' }
        const networkReason = ctx.networkCapabilityAllowed === false ? commandNetworkReason(command) : null
        if (networkReason) return { ok: false, output: `Blocked by ticket capability policy: ${networkReason}; add the network capability to run it.`, command, meta: 'blocked' }
        const profile = sandboxProfileForCommand(command)
        if (isRemote(ctx)) {
          const startedRemote = Date.now()
          const backend = workspaceBackend(ctx)
          const mutationBaseline = profile === 'read-only' ? null : await captureRemoteCommandSnapshot(backend, ctx.cwd).catch(() => null)
          const res = await backend.exec(command, { cwd: ctx.cwd, timeoutMs: 120_000, signal: ctx.signal })
          const changedPaths = await remoteCommandChangedPaths(backend, ctx.cwd, mutationBaseline).catch(() => [])
          const msRemote = Date.now() - startedRemote
          const streams = redactCommandStreams(res.stdout, res.stderr)
          const out = [streams.stdout, streams.stderr].filter(Boolean).join('\n').trim()
          return {
            ok: res.code === 0,
            output: commandOutputWithRedactionNotice(out || (res.code === 0 ? '(no output)' : `Exited with code ${res.code}`), streams.redactionCount),
            command,
            meta: `exit ${res.code} · ${msRemote}ms · ssh${streams.redactionCount ? ` · redacted=${streams.redactionCount}` : ''}`,
            changedPaths,
          }
        }
        const started = Date.now()
        const result = await runSandboxedCommand(command, {
          cwd: ctx.cwd,
          profile,
          // Only a command already classified as network-bearing can use the task
          // network toggle. Opaque build/interpreter profiles remain network-denied.
          allowNetwork: taskSandboxNetworkOverride(profile, ctx.allowSandboxNetwork),
          timeoutMs: 120_000,
          maxBuffer: 8 * 1024 * 1024,
          signal: ctx.signal,
        })
        const ms = Date.now() - started
        const out = [result.stdout, result.stderr].filter(Boolean).join('\n').trim()
        const redactionCount = result.redactionCount ?? 0
        // The artifact record rides as metadata for the UI; the model sees only the command output.
        return {
          ok: result.code === 0,
          output: commandOutputWithRedactionNotice(out || (result.code === 0 ? '(no output)' : `Exited with code ${result.code}`), redactionCount),
          command,
          meta: `exit ${result.code} · ${ms}ms · sandbox=${result.profile} · ${result.implementation}${result.artifacts ? ` · artifacts=${result.artifacts.id}` : ''}${redactionCount ? ` · redacted=${redactionCount}` : ''}`,
          sandboxArtifacts: result.artifacts,
          ...(result.artifacts ? { artifacts: summarizeSandboxArtifactBundle(result.artifacts) } : {}),
          changedPaths: [...new Set([
            ...(result.changedPaths ?? []),
            ...(result.artifacts?.diffs.map((entry) => entry.path) ?? []),
          ])],
        }
      }

      case 'repo_wiki': {
        const root = ctx.cwd
        const out: string[] = []
        try {
          const pkg = JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8'))
          out.push(`# ${pkg.name ?? path.basename(root)}`)
          if (pkg.description) out.push(`\n${pkg.description}`)
          const scripts = Object.entries(pkg.scripts ?? {})
          if (scripts.length) { out.push('\n## Scripts'); for (const [k, v] of scripts) out.push(`- \`npm run ${k}\`: ${v}`) }
          const deps = Object.keys(pkg.dependencies ?? {})
          if (deps.length) out.push(`\n## Key dependencies\n${deps.slice(0, 24).join(', ')}`)
        } catch { out.push(`# ${path.basename(root)}`) }

        const KNOWN = ['index.html', 'vite.config.ts', 'vite.config.js', 'tsconfig.json', 'next.config.js', 'src/main.tsx', 'src/main.ts', 'src/index.tsx', 'src/App.tsx', 'electron/main.ts', 'server.ts', 'Dockerfile', 'docker-compose.yml', 'AGENTS.md', 'CLAUDE.md', 'README.md']
        const present: string[] = []
        for (const f of KNOWN) { try { await fs.access(path.join(root, f)); present.push(f) } catch { /* absent */ } }
        if (present.length) out.push(`\n## Entry points & config\n${present.map((f) => `- ${f}`).join('\n')}`)

        out.push('\n## Structure')
        const tree: string[] = []
        const walk = async (dir: string, prefix: string, depth: number) => {
          if (depth > 2 || tree.length > 200) return
          let entries: any[] = []
          try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
          const filtered = entries
            .filter((e) => !e.name.startsWith('.') && !SKIP_DIRS.has(e.name))
            .sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))
          for (const e of filtered.slice(0, 40)) {
            tree.push(`${prefix}${e.name}${e.isDirectory() ? '/' : ''}`)
            if (e.isDirectory()) await walk(path.join(dir, e.name), prefix + '  ', depth + 1)
          }
        }
        await walk(root, '', 0)
        out.push(tree.join('\n'))
        return { ok: true, output: out.join('\n'), title: 'Repo map', meta: `${present.length} key files` }
      }

      default:
        return { ok: false, output: `Unknown tool: ${name}` }
    }
  } catch (e: any) {
    return { ok: false, output: e?.message ? String(e.message) : String(e) }
  }
}
