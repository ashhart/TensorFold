import type { ConnectorDefinition, ConnectorId, ConnectorSourceRef } from '../../lib/types'
import type { IconName } from '../../lib/icons'

export type SourceKey = 'all' | ConnectorId
export type Priority = 'urgent' | 'soon' | 'watch'

export interface AssistantItem {
  id: string
  source: ConnectorId
  eventId?: string
  threadKey?: string
  count?: number
  connectorAction?: string
  sourceIds?: string[]
  sources?: ConnectorSourceRef[]
  from: string
  title: string
  detail: string
  priority: Priority
  age: string
  action: string
}

export interface SourceRow {
  id: ConnectorId
  name: string
  label: string
  icon: IconName
  status: ConnectorDefinition['status']
  permissionStatus: ConnectorDefinition['permissionStatus']
  permissionDetail: string
  error?: string
  unread: number
  lastSync: string
  lastActivity: string
}
