import type { AssistantControlSnapshot, ConnectorEvent, EmailThreadResult } from '../../lib/types'
import { sourceName } from './mappings'
import type { AssistantItem, SourceKey } from './types'

export function assistantTaskContext(scope: string, selectedItem: AssistantItem | null, selectedEvent: ConnectorEvent | null | undefined, visibleItems: AssistantItem[], briefing?: AssistantControlSnapshot['briefing']) {
  const lines = [
    '<assistant_source_context>',
    `scope: ${scope}`,
  ]
  if (briefing?.lines.length) {
    lines.push('briefing:')
    lines.push(`- summary: ${briefing.summary}`)
    lines.push(`- source_ids: ${briefing.sourceIds.join(', ') || 'none'}`)
    for (const line of briefing.lines.slice(0, 6)) lines.push(`- ${line}`)
  }
  if (selectedItem) {
    lines.push('selected_item:')
    lines.push(`- source: ${sourceName(selectedItem.source)}`)
    lines.push(`- event_id: ${selectedItem.eventId ?? selectedItem.id}`)
    lines.push(`- source_ids: ${(selectedItem.sourceIds ?? []).join(', ') || 'none'}`)
    for (const source of selectedItem.sources ?? []) {
      lines.push(`- source_ref: ${source.id}${source.path ? ` (${source.path})` : ''}`)
    }
    if (selectedItem.threadKey) lines.push(`- thread_key: ${selectedItem.threadKey}`)
    if (selectedItem.connectorAction) lines.push(`- connector_action: ${selectedItem.connectorAction}`)
    lines.push(`- actor: ${selectedItem.from}`)
    lines.push(`- title: ${selectedItem.title}`)
    lines.push(`- summary: ${selectedItem.detail}`)
    lines.push(`- priority: ${selectedItem.priority}`)
    lines.push(`- suggested_action: ${selectedItem.action}`)
    if (selectedEvent) {
      lines.push(`- trust_level: ${selectedEvent.trustLevel}`)
      lines.push(`- requires_approval: ${selectedEvent.requiresApproval ? 'yes' : 'no'}`)
      lines.push(`- payload_keys: ${selectedEvent.payloadKeys.join(', ') || 'none'}`)
    }
  } else {
    lines.push('visible_signals:')
    for (const item of visibleItems.slice(0, 8)) {
      const ids = item.sourceIds?.length ? ` source_ids=${item.sourceIds.join(',')}` : ''
      lines.push(`- [${sourceName(item.source)}${ids}] ${item.title} from ${item.from}: ${item.detail}`)
    }
  }
  lines.push('</assistant_source_context>')
  return lines.join('\n')
}

export function emailThreadContext(thread: EmailThreadResult) {
  const lines = [
    '<email_thread_context>',
    thread.summary,
  ]
  for (const message of thread.messages.slice(-6)) {
    const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
    lines.push(`- id: ${message.id}`)
    lines.push(`  from: ${message.from || 'Unknown'}`)
    lines.push(`  to: ${message.to || '(not listed)'}`)
    lines.push(`  date: ${when}`)
    lines.push(`  subject: ${message.subject || '(no subject)'}`)
    lines.push(`  body_excerpt: ${message.body || message.snippet}`.slice(0, 900))
  }
  lines.push('</email_thread_context>')
  return lines.join('\n')
}

export function quickActionsFor(source: SourceKey): string[] {
  if (source === 'email') return [
    'Find emails that need replies and draft the responses for review.',
    'Summarize unread email and group it by urgency.',
    'Find commitments I made in email this week.',
  ]
  if (source === 'telegram' || source === 'whatsapp' || source === 'slack') return [
    `Summarize unread ${sourceName(source)} messages and show decisions, blockers, and replies needed.`,
    `Find anything in ${sourceName(source)} that needs my response today.`,
    `Turn the latest ${sourceName(source)} updates into tasks.`,
  ]
  if (source === 'calendar') return [
    'Summarize today and flag anything I need to prepare.',
    'Find meetings that need follow-up notes.',
    'Create a plan around my next two commitments.',
  ]
  if (source === 'docs') return [
    'Summarize the latest documents and highlight decisions needed.',
    'Compare documents against recent communication claims.',
    'Find open questions in the current document set.',
  ]
  if (source === 'browser') return [
    'Summarize saved browser sources and cite the pages.',
    'Find browser pages related to the current blocker.',
    'Turn the explicit browser context into next actions.',
  ]
  return [
    'Check what needs my attention and make a short work plan.',
    'Prepare a morning brief from email, messages, calendar, and documents.',
    'Find anything waiting on me and group it by source.',
  ]
}

export function actionLabel(action: string) {
  if (/repl|draft/i.test(action)) return 'Draft replies'
  if (/summarize|brief/i.test(action)) return 'Summarize'
  if (/commitment|waiting|attention/i.test(action)) return 'Needs me'
  if (/task/i.test(action)) return 'Make tasks'
  if (/compare/i.test(action)) return 'Compare'
  if (/plan/i.test(action)) return 'Plan'
  return 'Use'
}
