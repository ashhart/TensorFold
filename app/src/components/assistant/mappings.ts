import type { AssistantControlSnapshot, ConnectorAttentionItem, ConnectorDefinition, ConnectorEvent, ConnectorId } from '../../lib/types'
import type { IconName } from '../../lib/icons'
import { CONNECTOR_ICONS, CONNECTOR_NAMES, FALLBACK_SOURCES, SOURCE_ORDER } from './constants'
import type { AssistantItem, Priority, SourceRow } from './types'

export const sourceIcon = (source: ConnectorId): IconName => CONNECTOR_ICONS[source] ?? 'file'
export const sourceName = (source: ConnectorId): string => CONNECTOR_NAMES[source] ?? source

export function sourceFromConnector(connector: ConnectorDefinition): SourceRow {
  return {
    id: connector.id,
    name: connector.name,
    label: connector.description,
    icon: sourceIcon(connector.id),
    status: connector.status,
    permissionStatus: connector.permissionStatus,
    permissionDetail: connector.permissionDetail,
    error: connector.error,
    unread: connector.unread,
    lastSync: syncLabel(connector.lastSyncAt, connector.status),
    lastActivity: connector.lastEventAt ? timeAgo(connector.lastEventAt) : 'no activity',
  }
}

export function mergeSourceRows(rows: SourceRow[]) {
  const merged = new Map<ConnectorId, SourceRow>()
  for (const source of FALLBACK_SOURCES) merged.set(source.id, source)
  for (const source of rows) merged.set(source.id, source)
  return [...merged.values()].sort((a, b) => sourceSort(a.id) - sourceSort(b.id))
}

export function itemFromEvent(event: ConnectorEvent): AssistantItem {
  const priority: Priority = event.requiresApproval || event.priority === 'critical' ? 'urgent' : event.priority === 'high' ? 'soon' : 'watch'
  const action = event.requiresApproval
    ? 'Review'
    : event.classification === 'document'
      ? 'Compare'
      : event.classification === 'schedule'
        ? 'Plan'
        : 'Summarize'
  return {
    id: event.id,
    source: event.connectorId,
    eventId: event.id,
    threadKey: event.threadKey,
    connectorAction: event.action,
    sourceIds: connectorSourceIds(event),
    sources: event.sources ?? [],
    from: event.actor || event.pluginName,
    title: event.title,
    detail: event.summary,
    priority,
    age: timeAgo(event.createdAt),
    action,
  }
}

export function itemFromAttention(item: ConnectorAttentionItem): AssistantItem {
  const priority: Priority = item.requiresApproval || item.priority === 'critical' ? 'urgent' : item.priority === 'high' ? 'soon' : 'watch'
  return {
    id: item.eventId,
    source: item.connectorId,
    eventId: item.eventId,
    threadKey: item.threadKey,
    connectorAction: item.connectorAction,
    sourceIds: item.sourceIds,
    sources: item.sources,
    from: item.actor,
    title: item.title,
    detail: item.summary,
    priority,
    age: timeAgo(item.createdAt),
    action: item.action,
  }
}

export function connectorSourceIds(event: ConnectorEvent): string[] {
  const ids = [...new Set((event.sources ?? []).map((source) => source.id).filter(Boolean))]
  return ids.length ? ids : [`connector:${event.id}`]
}

export function sourceSort(id: ConnectorId) {
  const index = SOURCE_ORDER.indexOf(id)
  return index === -1 ? 99 : index
}

export function syncLabel(lastSyncAt: number | null, status: ConnectorDefinition['status']) {
  if (lastSyncAt) return `synced ${timeAgo(lastSyncAt)}`
  if (status === 'pending') return 'never synced'
  return 'ready, no sync yet'
}

export function timeAgo(ts: number) {
  const delta = Math.max(0, Date.now() - ts)
  const minutes = Math.floor(delta / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

// Re-export the snapshot briefing type alias for task-context consumers.
export type AssistantBriefing = AssistantControlSnapshot['briefing']
