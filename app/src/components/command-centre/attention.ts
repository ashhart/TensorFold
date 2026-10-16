import { relativeTime } from '../../lib/format'
import type { AwayBriefItem } from '../../lib/away-brief'
import type { IconName } from '../../lib/icons'
import type {
  ApprovalRule,
  ConnectorAttentionItem,
  ConnectorEvent,
  ConnectorId,
  ConnectorPriority,
  MissionStatus,
  Room,
  Ticket,
} from '../../lib/types'
import { CONNECTOR_ICONS, CONNECTOR_NAMES, TONE_RANK, type ApprovalGroup, type ApprovalRow, type AttentionRow, type AwayBriefRow, type AwayBriefRowDeps, type Tone } from './model'

export function attentionFromConnectorFeed(item: ConnectorAttentionItem): AttentionRow {
  const tone = item.requiresApproval ? 'critical' : priorityTone(item.priority)
  return {
    id: `connector:${item.eventId}`,
    icon: sourceIcon(item.connectorId),
    source: `${connectorName(item.connectorId)} / ${item.actor}`,
    title: item.title,
    detail: item.summary,
    tone,
    age: relativeTime(item.createdAt),
    action: item.action,
    sourceIds: item.sourceIds,
    createdAt: item.createdAt,
  }
}

export function attentionFromConnectorEvent(event: ConnectorEvent): AttentionRow {
  const tone = event.requiresApproval ? 'critical' : priorityTone(event.priority)
  const action = event.requiresApproval
    ? 'Review'
    : event.classification === 'document'
      ? 'Compare'
      : event.classification === 'schedule'
        ? 'Plan'
        : 'Summarize'
  return {
    id: `connector:${event.id}`,
    icon: sourceIcon(event.connectorId),
    source: `${connectorName(event.connectorId)} / ${event.actor || event.pluginName}`,
    title: event.title,
    detail: event.summary,
    tone,
    age: relativeTime(event.createdAt),
    action,
    sourceIds: connectorEventSourceIds(event),
    createdAt: event.createdAt,
  }
}

export function connectorEventSourceIds(event: ConnectorEvent): string[] {
  const ids = [...new Set((event.sources ?? []).map((source) => source.id).filter(Boolean))]
  return ids.length ? ids : [`connector:${event.id}`]
}

export function awayItemToRow(item: AwayBriefItem, deps: AwayBriefRowDeps): AwayBriefRow {
  if (item.kind === 'connector') {
    const connectorId = item.connectorId ?? 'webhook'
    const pendingApproval = item.status === 'pending'
    return {
      id: item.id,
      icon: sourceIcon(connectorId),
      title: item.title,
      detail: item.detail,
      meta: `${connectorName(connectorId)} / ${item.sourceIds[0] ?? item.status ?? 'source'}`,
      tone: item.tone,
      createdAt: item.createdAt,
      actionLabel: pendingApproval ? 'Review' : 'Sources',
      onOpen: pendingApproval
        ? () => document.getElementById('command-centre-approvals')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        : () => deps.setView({ name: 'plugins' }),
    }
  }
  if (item.kind === 'room') {
    return {
      id: item.id,
      icon: 'layers',
      title: item.title,
      detail: item.detail,
      meta: `Room / ${item.sourceIds[0]}`,
      tone: item.tone,
      createdAt: item.createdAt,
      actionLabel: 'Open',
      onOpen: item.roomId ? () => deps.openRoom(item.roomId!) : undefined,
    }
  }
  if (item.kind === 'automation') {
    return {
      id: item.id,
      icon: 'clock',
      title: item.title,
      detail: item.detail,
      meta: `Automation / ${item.sourceIds.at(-1) ?? item.sourceIds[0]}`,
      tone: item.tone,
      createdAt: item.createdAt,
      actionLabel: 'Open',
      onOpen: () => deps.setView({ name: 'automations' }),
    }
  }
  if (item.kind === 'pr_loop') {
    const job = item.jobId ? deps.durableJobs.find((candidate) => candidate.id === item.jobId) : null
    return {
      id: item.id,
      icon: 'branch',
      title: item.title,
      detail: item.detail,
      meta: `PR loop / ${item.sourceIds[0] ?? item.status ?? 'local'}`,
      tone: item.tone,
      createdAt: item.createdAt,
      actionLabel: job ? 'Inspect' : 'Open',
      onOpen: job
        ? () => deps.openDurableJob(job)
        : () => document.getElementById('command-centre-missions')?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
    }
  }
  return {
    id: item.id,
    icon: 'spark',
    title: item.title,
    detail: item.detail,
    meta: `Mission / ${item.sourceIds[0]}`,
    tone: item.tone,
    createdAt: item.createdAt,
    actionLabel: 'Open',
    onOpen: () => document.getElementById('command-centre-missions')?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
  }
}

export function ticketNeedsAttention(ticket: Ticket) {
  return ticket.status === 'review'
    || ticket.status === 'needs_clarification'
    || ticket.priority === 'urgent'
    || (ticket.priority === 'high' && ['ready', 'in_progress'].includes(ticket.status))
}

export function priorityTone(priority: ConnectorPriority): Tone {
  if (priority === 'critical') return 'critical'
  if (priority === 'high') return 'high'
  if (priority === 'low') return 'low'
  return 'normal'
}

export function approvalRiskLabel(tone: Tone): string {
  if (tone === 'critical') return 'Critical risk'
  if (tone === 'high') return 'High risk'
  if (tone === 'good') return 'Review'
  if (tone === 'low') return 'Low risk'
  return 'Normal risk'
}

export function groupApprovalRows(rows: ApprovalRow[]): ApprovalGroup[] {
  const groups = new Map<string, ApprovalGroup>()
  for (const row of rows) {
    const risk = approvalRiskLabel(row.tone)
    const id = `${row.tone}:${row.sourceGroup}`
    const current = groups.get(id)
    if (current) {
      current.rows.push(row)
    } else {
      groups.set(id, {
        id,
        title: `${risk} / ${row.sourceGroup}`,
        detail: '1 waiting',
        tone: row.tone,
        rows: [row],
      })
    }
  }
  return [...groups.values()]
    .map((group) => ({
      ...group,
      detail: `${group.rows.length} waiting`,
      rows: group.rows.sort(sortRows),
    }))
    .sort((a, b) => TONE_RANK[a.tone] - TONE_RANK[b.tone] || b.rows[0].createdAt - a.rows[0].createdAt || a.title.localeCompare(b.title))
}

export function sourceIcon(id: ConnectorId): IconName {
  return CONNECTOR_ICONS[id] ?? 'file'
}

export function connectorName(id: ConnectorId): string {
  return CONNECTOR_NAMES[id] ?? id
}

export function approvalRuleIcon(rule: ApprovalRule): IconName {
  if (rule.scope === 'connector') return 'plug'
  const key = `${rule.legacyTool ?? ''}:${rule.key}`
  if (key.includes('email')) return 'mail'
  if (key.includes('run_command')) return 'terminal'
  if (key.includes('write_file') || key.includes('edit_file') || key.includes('apply_patch')) return 'file'
  if (key.includes('launch_subagent')) return 'branch'
  if (key.includes('mcp__')) return 'plug'
  return 'shield'
}

export function approvalRuleTitle(rule: ApprovalRule): string {
  if (rule.legacyTool) return `${rule.legacyTool} policy`
  if (rule.scope === 'connector') return 'Connector policy'
  const [tool] = rule.key.split(':')
  return `${tool || 'Tool'} policy`
}

export function projectName(projects: { id: string; name: string }[], id: string) {
  return projects.find((p) => p.id === id)?.name ?? 'Project'
}

export function sortRows<T extends { tone: Tone; createdAt: number }>(a: T, b: T) {
  return TONE_RANK[a.tone] - TONE_RANK[b.tone] || b.createdAt - a.createdAt
}

export function threadStatusLabel(status: string) {
  if (status === 'thinking') return 'thinking'
  if (status === 'streaming') return 'streaming'
  if (status === 'tool') return 'tool'
  if (status === 'awaiting') return 'awaiting'
  return 'running'
}

export function roomStatusLabel(state: Room['state']) {
  if (state === 'waiting_user') return 'waiting user'
  if (state === 'waiting_external') return 'waiting external'
  return state
}

export function missionIsActive(status: MissionStatus) {
  return ['queued', 'running', 'waiting_approval', 'paused', 'stalled'].includes(status)
}

export function missionIsTerminal(status: MissionStatus) {
  return ['completed', 'failed', 'cancelled'].includes(status)
}

export function missionStatusLabel(status: MissionStatus) {
  return status.replace(/_/g, ' ')
}
