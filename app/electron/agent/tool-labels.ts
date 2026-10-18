import { randomUUID } from 'node:crypto'
import { controllerActionTarget, normalizeControllerActionInput } from './controller-actions'
import { describeComputerAction, normalizeComputerUseInput } from './computer-use/actions'
import type { ConnectorEvent } from './types'

type PersonalMessageConnectorId = Extract<ConnectorEvent['connectorId'], 'telegram' | 'whatsapp' | 'imessage'>

export function formatMarkdownTable(columns: string[], rows: string[][]): string {
  const safeColumns = columns.length ? columns : ['Column 1']
  const escape = (value: string) => String(value ?? '').replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim()
  const header = `| ${safeColumns.map(escape).join(' | ')} |`
  const divider = `| ${safeColumns.map(() => '---').join(' | ')} |`
  const body = rows.length
    ? rows.map((row) => `| ${safeColumns.map((_col, index) => escape(row[index] ?? '')).join(' | ')} |`)
    : [`| ${safeColumns.map(() => '').join(' | ')} |`]
  return [header, divider, ...body].join('\n')
}

export function normalizePersonalMessageConnector(value: unknown): PersonalMessageConnectorId | null {
  const connectorId = String(value || '').trim().toLowerCase()
  return connectorId === 'telegram' || connectorId === 'whatsapp' || connectorId === 'imessage'
    ? connectorId
    : null
}

export function personalMessageDraftEvent(input: { connectorId: PersonalMessageConnectorId; target: string; body: string; threadKey?: string; accountId?: string }): ConnectorEvent {
  const wantsReply = Boolean(input.threadKey)
  const action = wantsReply ? 'message_reply_requested' : 'message_send_requested'
  return {
    id: randomUUID(),
    connectorId: input.connectorId,
    pluginName: connectorName(input.connectorId),
    action,
    direction: 'outbound',
    classification: 'approval',
    priority: 'high',
    trustLevel: 'approve',
    requiresApproval: true,
    approvalId: `approval-${randomUUID().slice(0, 8)}`,
    approvalStatus: 'pending',
    resolvedAt: null,
    resolvedBy: null,
    title: `${connectorName(input.connectorId)} message to ${input.target}`,
    summary: `${wantsReply ? 'Reply request' : 'Send request'} to ${input.target}: ${compactToolText(input.body, 220)}. No external message is sent until approval.`,
    actor: input.target,
    threadKey: input.threadKey || `${input.connectorId}:message:${input.target}`,
    payloadKeys: ['connector_id', 'target', 'body', ...(input.accountId ? ['account_id'] : []), ...(input.threadKey ? ['thread_key'] : [])],
    payload: {
      connectorId: input.connectorId,
      target: input.target,
      body: input.body,
      ...(input.accountId ? { accountId: input.accountId } : {}),
      ...(input.threadKey ? { threadKey: input.threadKey } : {}),
    },
    createdAt: Date.now(),
  }
}

export function connectorName(connectorId: PersonalMessageConnectorId): string {
  if (connectorId === 'telegram') return 'Telegram'
  if (connectorId === 'whatsapp') return 'WhatsApp'
  return 'iMessage'
}

function compactToolText(value: string, max: number): string {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  return text.length <= max ? text : `${text.slice(0, max - 3).trim()}...`
}

export function formatToolNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, '')
}

// Short human label for a tool call, shown on the tool card header.
export function toolLabel(name: string, args: Record<string, any>): { verb: string; target: string } {
  switch (name) {
    case 'list_dir': return { verb: 'List', target: args.path || '.' }
    case 'read_file': return { verb: 'Read', target: args.path || '' }
    case 'write_file': return { verb: 'Write', target: args.path || '' }
    case 'edit_file': return { verb: 'Edit', target: args.path || '' }
    case 'apply_patch': return { verb: 'Patch', target: args.path || 'workspace' }
    case 'web_search': return { verb: 'Web search', target: args.query || '' }
    case 'web_fetch': return { verb: 'Web fetch', target: args.url || '' }
    case 'search': return { verb: 'Search', target: args.query || '' }
    case 'glob': return { verb: 'Glob', target: args.pattern || '' }
    case 'ast_grep_search': return { verb: 'Structural search', target: args.pattern || '' }
    case 'ast_grep_rewrite_preview': return { verb: 'Rewrite preview', target: args.pattern || '' }
    case 'semgrep_scan': return { verb: 'Semgrep scan', target: args.path || 'workspace' }
    case 'codebase_map': return { verb: 'Codebase map', target: args.path || 'workspace' }
    case 'symbol_index': return { verb: 'Symbol index', target: args.query || args.path || 'workspace' }
    case 'safe_refactor_preview': return { verb: 'Safe refactor', target: args.pattern || '' }
    case 'retrieval_search': return { verb: 'Retrieval search', target: args.query || args.path || 'index' }
    case 'email_list': return { verb: 'Email list', target: args.query || (args.unreadOnly ? 'unread' : 'inbox') }
    case 'email_read': return { verb: 'Email read', target: args.id || '' }
    case 'email_thread': return { verb: 'Email thread', target: args.id || '' }
    case 'email_summary': return { verb: 'Email summary', target: args.query || 'inbox' }
    case 'email_compose': return { verb: args.mode === 'send' ? 'Email send' : 'Email draft', target: args.to || args.threadKey || args.inReplyTo || '' }
    case 'email_resolve': return { verb: 'Email resolve', target: args.id || '' }
    case 'message_list': return { verb: 'Message list', target: args.channel || args.query || 'messages' }
    case 'message_read': return { verb: 'Message read', target: args.id || '' }
    case 'message_summary': return { verb: 'Message summary', target: args.channel || args.query || 'messages' }
    case 'message_compose': return { verb: 'Message send', target: `${args.connectorId || 'message'}:${args.target || ''}` }
    case 'calendar_list': return { verb: 'Calendar list', target: args.day || args.query || 'events' }
    case 'calendar_read': return { verb: 'Calendar read', target: args.id || '' }
    case 'calendar_summary': return { verb: 'Calendar summary', target: args.day || 'today' }
    case 'calendar_create': return { verb: 'Calendar create', target: args.title || args.sourceId || '' }
    case 'calendar_update': return { verb: 'Calendar update', target: args.id || args.title || '' }
    case 'document_list': return { verb: 'Document list', target: args.query || args.path || 'documents' }
    case 'document_read': return { verb: 'Document read', target: args.id || '' }
    case 'document_summary': return { verb: 'Document summary', target: args.query || args.path || 'documents' }
    case 'table_list': return { verb: 'Table list', target: args.query || args.path || 'tables' }
    case 'table_read': return { verb: 'Table read', target: args.id || '' }
    case 'table_summary': return { verb: 'Table summary', target: args.query || args.path || 'tables' }
    case 'browser_list': return { verb: 'Browser list', target: args.query || 'browser sources' }
    case 'browser_read': return { verb: 'Browser read', target: args.id || '' }
    case 'browser_summary': return { verb: 'Browser summary', target: args.query || 'browser sources' }
    case 'browser_automation': return { verb: `Browser ${args.action || 'automation'}`, target: args.url || args.selector || args.ref || args.tabId || '' }
    case 'controller_action': {
      // normalizeControllerActionInput throws on malformed/partial args (missing
      // appName, invalid action, own-app target). Labels must never throw — a
      // throw here escapes the per-call handling in loop.ts and aborts the whole
      // turn. Fall back to a defensive label built from raw args on any error.
      try {
        const labelInput = normalizeControllerActionInput(args, { ownAppName: 'TensorFold' })
        return { verb: `Controller ${labelInput.action}`, target: `${labelInput.appName} ${controllerActionTarget(labelInput)}`.trim() }
      } catch {
        const action = String(args.action || '').trim()
        return { verb: action ? `Controller ${action}` : 'Controller action', target: String(args.appName || '') }
      }
    }
    case 'computer_use': {
      // Labels never throw: malformed args fall back to the raw action and app.
      try { return { verb: 'Computer use', target: describeComputerAction(normalizeComputerUseInput(args)) } } catch { return { verb: 'Computer use', target: [args.action, args.app].filter(Boolean).join(' ') } }
    }
    case 'phone_observe': return { verb: 'Read phone screen', target: 'Connected phone for this chat' }
    case 'phone_control': return { verb: 'Phone control', target: `${String(args.action || 'action')}${args.note ? ` · ${String(args.note).slice(0, 100)}` : ''}` }
    case 'launch_subagent': return { verb: 'Launch sub-agent', target: args.name || args.task || '' }
    case 'visualize': return { verb: 'Visualize', target: args.title || args.spec?.title || '' }
    case 'run_command': return { verb: 'Run', target: args.command || '' }
    default:
      if (name.startsWith('mcp__')) return { verb: 'MCP', target: name.split('__').slice(1).join(' / ') }
      return { verb: name, target: '' }
  }
}

export function parseStringList(value: unknown): string[] | undefined {
  if (Array.isArray(value)) {
    const items = value.map((item) => String(item || '').trim()).filter(Boolean)
    return items.length ? items : undefined
  }
  if (typeof value === 'string') {
    const items = value.split(',').map((item) => item.trim()).filter(Boolean)
    return items.length ? items : undefined
  }
  return undefined
}
