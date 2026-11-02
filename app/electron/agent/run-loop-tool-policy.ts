// Resolve executable tools and approval policy without weakening captured bot restrictions.

import { botToolBlockReason, isRestrictedBot } from '../../shared/bot-tool-policy'
import { scopeBotToolContext } from './bot-tool-scope'
import { executeTool, type ToolContext, type ToolResult } from './tools'
import { PluginRuntime } from './plugin-runtime'
import { commandApprovalReason } from './command-policy'
import { approvalPolicyScope, externalEffectApprovalReason } from './approval-policy'
import { type ApprovalMode } from './types'

const FULL_ACCESS_ASK_COMMAND_CLASSES = new Set(['destructive-filesystem', 'secret-read', 'system-control', 'network-shell'])

const GIT_PUSH_RE = /(^|[;&|\s])git\s+(?:-\S+\s+)*push\b/

export const MAX_PARALLEL_READ_TOOLS = 4

export const PARALLEL_READ_ONLY_TOOLS = new Set([
  'list_dir', 'read_file', 'search', 'glob',
  'explore', 'delegate_task', 'oracle',
  'repo_wiki', 'web_search', 'web_fetch',
  'ast_grep_search', 'semgrep_scan', 'codebase_map', 'symbol_index', 'retrieval_search',
  'session_search', 'session_read', 'assistant_doctor', 'tool_search', 'tool_describe',
  'subagent_status',
  'email_list', 'email_read', 'email_thread', 'email_summary',
  'message_list', 'message_read', 'message_summary',
  'calendar_list', 'calendar_read', 'calendar_summary',
  'document_list', 'document_read', 'document_summary',
  'table_list', 'table_read', 'table_summary',
  'browser_list', 'browser_read', 'browser_summary',
])

export function pluginToolSchemas(runtime: PluginRuntime, context: Partial<ToolContext> | undefined, enforceConfiguredTrust: boolean, builtinToolNames: ReadonlySet<string> = new Set()): any[] {
  const disabledTools = new Set(context?.disabledToolNames ?? [])
  const disabledCapabilities = new Set(context?.disabledCapabilityIds ?? [])
  const configured = context?.packagedPlugins ?? []
  const out: any[] = []
  const seen = new Set<string>()
  for (const entry of runtime.listToolSchemaEntries()) {
    const name = (entry.schema as any)?.function?.name
    if (!name || disabledTools.has(name) || seen.has(name)) continue
    if (context?.allowedToolNames && !context.allowedToolNames.includes(name)) continue

    // Builtins retain their executor; a colliding plugin schema must not replace their meaning.
    if (builtinToolNames.has(name)) {
      console.warn(`[plugin-runtime] plugin tool "${name}" shadows a builtin tool and is disabled.`)
      continue
    }
    if (!enforceConfiguredTrust) { seen.add(name); out.push(entry.schema as any); continue }
    const plugin = configured.find((candidate) => candidate.id === entry.pluginId || candidate.name === entry.pluginName)
    if (!plugin || plugin.enabled !== true || plugin.trusted !== true) continue
    const capabilityId = `plugin:${String(plugin.name || plugin.id).trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'custom'}`
    if (disabledCapabilities.has(capabilityId)) continue
    seen.add(name)
    out.push(entry.schema as any)
  }
  return out
}

export async function executeRuntimeTool(
  runtime: PluginRuntime,
  name: string,
  args: Record<string, any>,
  context: ToolContext,
  builtinToolNames: ReadonlySet<string>,
): Promise<ToolResult> {
  try { context = scopeBotToolContext(context.currentThread?.bot, context) }
  catch { return { ok: false, output: 'Invalid bot tool scope.' } }
  const botScopeBlock = botToolBlockReason(context.currentThread?.bot, name, args, context.allowedToolNames)
  if (botScopeBlock) return { ok: false, output: botScopeBlock }
  if (builtinToolNames.has(name)) return executeTool(name, args, context)
  if (isRestrictedBot(context.currentThread?.bot)) return { ok: false, output: 'Plugin tools are unavailable to restricted bots.' }
  const pluginTool = runtime.getRegisteredTool(name)
  if (!pluginTool) return executeTool(name, args, context)
  try {
    const raw = await pluginTool.handler(args, { cwd: context.cwd, threadId: context.currentThread?.id, signal: context.signal })
    if (raw && typeof raw === 'object' && typeof raw.ok === 'boolean' && typeof raw.output === 'string') {
      return raw as ToolResult
    }
    return { ok: true, output: typeof raw === 'string' ? raw : JSON.stringify(raw ?? null), title: name }
  } catch (error: any) {
    return { ok: false, output: `Plugin tool ${name} failed: ${error?.message ?? error}`, title: name }
  }
}

export function toolApprovalReasonForMode(tool: string, args: Record<string, any>, approvalMode: ApprovalMode): string | null {
  const reason = tool === 'run_command'
    ? commandApprovalReason(String(args.command ?? ''))
    : externalEffectApprovalReason(tool, args)
  if (approvalMode !== 'auto' || tool !== 'run_command') return reason
  const classification = approvalPolicyScope('tool', tool, args).classification ?? 'unknown'
  if (FULL_ACCESS_ASK_COMMAND_CLASSES.has(classification)) {
    return reason ?? `${classification.replace(/-/g, ' ')} commands ask even in Full access`
  }
  if (GIT_PUSH_RE.test(String(args.command ?? ''))) return 'git push sends code to another machine, so it asks even in Full access'
  return null
}
