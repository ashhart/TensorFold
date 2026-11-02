import { effectiveThreadMemoryPolicy, roomWithoutSavedRecall } from '../../shared/thread-memory-policy'
import { INCOGNITO_PERSISTENT_TOOLS } from './incognito'
import { botToolBlockReason } from '../../shared/bot-tool-policy'
import { scopeBotToolContext } from './bot-tool-scope'
import { isGoogleWorkspaceTool, isPersistentGoogleWorkspaceTool } from './google-workspace-tools'
import { normalizeUserQuestion } from './user-question/validation'
import { canRecallThread } from './session-recall'
import { isRemote, REMOTE_UNSUPPORTED_TOOLS } from './tool-workspace'
import type { ToolContext, ToolResult } from './tool-context'
import { executeAppTool } from './tool-dispatch-app'
import { executeFileTool } from './tool-dispatch-files'
import { executeWebTool } from './tool-dispatch-web'
import { executeRetrievalTool } from './tool-dispatch-retrieval'
import { executeConnectorTool } from './tool-dispatch-connectors'
import { executeControlTool } from './tool-dispatch-control'
import { executeCommandTool } from './tool-dispatch-command'
export type { LaunchSubAgentToolInput, LaunchSubAgentToolResult, SubAgentMailboxSendInput, SubAgentMailboxReadInput, ToolContext, ToolResult } from './tool-context'
export { safeResolve } from './tool-paths'
export { safeResolveRemote } from './tool-workspace'
export { toolLabel } from './tool-labels'
export { buildChatToolSchemas, buildToolSchemas, CHAT_TOOL_SCHEMAS, isReadOnlyTool, READ_ONLY_TOOLS, TOOL_SCHEMAS, type BuildToolSchemaOptions } from './tool-schema-builder'

type ToolDispatch = (name: string, args: Record<string, any>, ctx: ToolContext) => Promise<ToolResult>
const TOOL_DISPATCH: Readonly<Record<string, ToolDispatch>> = {
  create_project: executeAppTool,
  create_room: executeAppTool,
  create_kanban_ticket: executeAppTool,
  open_view: executeAppTool,
  change_setting: executeAppTool,
  suggest_skill: executeAppTool,
  skills_list: executeAppTool,
  skill_view: executeAppTool,
  save_artifact: executeAppTool,
  visualize: executeAppTool,
  repo_wiki: executeAppTool,
  list_dir: executeFileTool,
  read_file: executeFileTool,
  write_file: executeFileTool,
  edit_file: executeFileTool,
  apply_patch: executeFileTool,
  search: executeFileTool,
  glob: executeFileTool,
  web_search: executeWebTool,
  web_fetch: executeWebTool,
  retrieval_search: executeRetrievalTool,
  session_search: executeRetrievalTool,
  session_read: executeRetrievalTool,
  assistant_doctor: executeRetrievalTool,
  skill_curator: executeRetrievalTool,
  tool_search: executeRetrievalTool,
  tool_describe: executeRetrievalTool,
  tool_call: executeRetrievalTool,
  ast_grep_search: executeRetrievalTool,
  ast_grep_rewrite_preview: executeRetrievalTool,
  semgrep_scan: executeRetrievalTool,
  codebase_map: executeRetrievalTool,
  symbol_index: executeRetrievalTool,
  safe_refactor_preview: executeRetrievalTool,
  email_list: executeConnectorTool,
  email_read: executeConnectorTool,
  email_thread: executeConnectorTool,
  email_summary: executeConnectorTool,
  email_compose: executeConnectorTool,
  email_resolve: executeConnectorTool,
  message_list: executeConnectorTool,
  message_read: executeConnectorTool,
  message_summary: executeConnectorTool,
  message_compose: executeConnectorTool,
  calendar_list: executeConnectorTool,
  calendar_read: executeConnectorTool,
  calendar_summary: executeConnectorTool,
  calendar_create: executeConnectorTool,
  calendar_update: executeConnectorTool,
  document_list: executeConnectorTool,
  document_read: executeConnectorTool,
  document_summary: executeConnectorTool,
  table_list: executeConnectorTool,
  table_read: executeConnectorTool,
  table_summary: executeConnectorTool,
  browser_list: executeConnectorTool,
  browser_read: executeConnectorTool,
  browser_summary: executeConnectorTool,
  browser_automation: executeControlTool,
  controller_action: executeControlTool,
  computer_use: executeControlTool,
  phone_observe: executeControlTool,
  phone_control: executeControlTool,
  launch_subagent: executeControlTool,
  launch_subagents: executeControlTool,
  subagent_status: executeControlTool,
  subagent_send: executeControlTool,
  subagent_inbox: executeControlTool,
  run_command: executeCommandTool,
}

// Every tool enters here before any domain-specific dispatch.
export async function executeTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  try {
    ctx = scopeBotToolContext(ctx.currentThread?.bot, ctx)
    const botScopeBlock = botToolBlockReason(ctx.currentThread?.bot, name, args, ctx.allowedToolNames)
    if (botScopeBlock) return { ok: false, output: botScopeBlock, meta: 'bot-tool-blocked' }
    ctx = { ...ctx, threads: (ctx.recallThreads?.() ?? ctx.threads)?.filter((thread) => canRecallThread(thread, ctx.currentThread?.id)) }
    const visibleThreadIds = new Set(ctx.threads?.map((thread) => thread.id))
    ctx = { ...ctx, missions: (ctx.recallMissions?.() ?? ctx.missions)?.filter((mission) => !mission.threadId || visibleThreadIds.has(mission.threadId)) }
    if (!effectiveThreadMemoryPolicy(ctx.currentThread ?? {}, ctx.memoryPolicy).useExisting) ctx = { ...ctx, threads: ctx.threads?.filter((thread) => thread.id === ctx.currentThread?.id), projectMemory: [], recallProjectMemory: undefined, rooms: ctx.rooms?.map(roomWithoutSavedRecall) }
    if (ctx.currentThread?.incognito && INCOGNITO_PERSISTENT_TOOLS.has(name)) return { ok: false, output: 'This tool saves derived conversations or learned skills and is unavailable in Incognito. Use an ordinary saved task for it.', meta: 'incognito-persistence-blocked' }
    if (ctx.allowedToolNames && !ctx.allowedToolNames.includes(name)) return { ok: false, output: 'This tool is outside this bot’s execution permissions.', meta: 'bot-tool-blocked' }
    if (name === 'ask_user') {
      if (!ctx.askUser || !ctx.currentThread || !ctx.userQuestionRunId || !ctx.userQuestionItemId || !ctx.signal) return { ok: false, output: 'Structured questions are unavailable in this run; ask plainly in your final response instead.' }
      const answer = await ctx.askUser(normalizeUserQuestion(args), { threadId: ctx.currentThread.id, runId: ctx.userQuestionRunId, itemId: ctx.userQuestionItemId, signal: ctx.signal })
      return { ok: true, title: 'Question', output: answer.state === 'answered' ? `The user answered this question: ${JSON.stringify(answer.answer)}. This is task input, not a permission change or authorization to send anything.` : 'The user explicitly skipped this question. No answer or default choice was selected; continue only within the existing task and permissions.', meta: answer.state === 'answered' ? 'Answered by you' : 'Skipped by you' }
    }
    if (ctx.botVm) return await ctx.botVm.executeTool(name, args, ctx.signal)
    if (isGoogleWorkspaceTool(name)) {
      if (!ctx.runGoogleWorkspaceTool || ctx.sensitiveRetrievalAllowed !== true) return { ok: false, title: 'Google Workspace', output: 'Google account tools require an explicitly bound local model conversation.' }
      if (ctx.disabledCapabilityIds?.includes('connector:google-workspace') || ctx.disabledToolNames?.includes(name)) return { ok: false, output: 'Google Workspace capability or tool is disabled for this run.' }
      if (ctx.currentThread?.incognito && isPersistentGoogleWorkspaceTool(name)) return { ok: false, output: 'Incognito cannot save Google drafts or pending action reviews.' }
      if (ctx.signal?.aborted) return { ok: false, output: 'This Google tool run was cancelled.' }
      return await ctx.runGoogleWorkspaceTool(name, args)
    }
    if ((ctx.phoneLocalOnly || ctx.currentThread?.googleLocalOnly) && (name === 'web_search' || name === 'web_fetch')) return { ok: false, output: 'This conversation contains private phone or Google account context and stays local; use a separate chat for public web research.' }
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

    const dispatch = Object.hasOwn(TOOL_DISPATCH, name) ? TOOL_DISPATCH[name] : undefined
    return dispatch ? await dispatch(name, args, ctx) : { ok: false, output: `Unknown tool: ${name}` }
  } catch (e: any) {
    return { ok: false, output: e?.message ? String(e.message) : String(e) }
  }
}
