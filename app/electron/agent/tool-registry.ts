import { CHAT_TOOL_SCHEMAS, TOOL_SCHEMAS } from './tool-schemas'
import { GOOGLE_WORKSPACE_READ_TOOL_NAMES } from './google-workspace-tools'

export type ToolRegistryScope = 'agent' | 'chat'

export interface ToolSchemaLike {
  type: string
  function?: {
    name?: string
    description?: string
    parameters?: Record<string, unknown>
  }
}

export interface ToolRegistryEntry<T extends ToolSchemaLike = ToolSchemaLike> {
  name: string
  schema: T
  scope: ToolRegistryScope
  source: 'static'
  readOnly: boolean
  externalEffect: boolean
}

export interface BuildStaticToolRegistryOptions {
  scope?: ToolRegistryScope
}

// Tools that never mutate are safe to auto-run regardless of approval mode.
// App-control: open_view (pure navigation) and change_setting (confirm-first in the
// renderer) are safe to auto-run. create_project/create_room/create_kanban_ticket
// mutate app state and stay approval-gated.
export const READ_ONLY_TOOL_NAMES = new Set([
  ...GOOGLE_WORKSPACE_READ_TOOL_NAMES,
  'ask_user',
  'phone_observe',
  'list_dir',
  'read_file',
  'search',
  'glob',
  'update_plan',
  'explore',
  'delegate_task',
  'oracle',
  'repo_wiki',
  'visualize',
  'web_search',
  'web_fetch',
  'ast_grep_search',
  'semgrep_scan',
  'codebase_map',
  'symbol_index',
  'retrieval_search',
  'session_search',
  'session_read',
  'subagent_status',
  // Internal mailbox state is safe in readonly runs, but remains ordered by the
  // execution scheduler because send/read acknowledgements mutate durable state.
  'subagent_send',
  'subagent_inbox',
  'assistant_doctor',
  'skill_curator',
  'save_artifact',
  'tool_search',
  'tool_describe',
  'email_list',
  'email_read',
  'email_thread',
  'email_summary',
  'message_list',
  'message_read',
  'message_summary',
  'calendar_list',
  'calendar_read',
  'calendar_summary',
  'document_list',
  'document_read',
  'document_summary',
  'table_list',
  'table_read',
  'table_summary',
  'browser_list',
  'browser_read',
  'browser_summary',
  'open_view',
  'change_setting',
])

export function buildStaticToolRegistry(options: BuildStaticToolRegistryOptions = {}): ToolRegistryEntry[] {
  const scope = options.scope ?? 'agent'
  const schemas = scope === 'chat' ? CHAT_TOOL_SCHEMAS : TOOL_SCHEMAS
  return schemas.flatMap((schema): ToolRegistryEntry[] => {
    const name = schema.function?.name
    if (!name) return []
    const readOnly = READ_ONLY_TOOL_NAMES.has(name)
    return [{
      name,
      schema,
      scope,
      source: 'static',
      readOnly,
      externalEffect: !readOnly,
    }]
  })
}

export function toolRegistrySchemas<T extends ToolSchemaLike>(entries: ToolRegistryEntry<T>[]): T[] {
  return entries.map((entry) => entry.schema)
}

export function isRegistryReadOnlyTool(name: string, entries: ToolRegistryEntry[] = buildStaticToolRegistry()): boolean {
  if (READ_ONLY_TOOL_NAMES.has(name)) return true
  return entries.find((entry) => entry.name === name)?.readOnly === true
}
