import type { ConnectorId } from '../../lib/types'
import type { IconName } from '../../lib/icons'
import type { Priority, SourceRow } from './types'

export const CONNECTOR_ICONS: Record<ConnectorId, IconName> = {
  email: 'mail',
  slack: 'message',
  discord: 'message',
  mattermost: 'message',
  matrix: 'message',
  googlechat: 'message',
  whatsapp: 'at',
  imessage: 'message',
  docs: 'file',
  browser: 'browser',
  calendar: 'clock',
  telegram: 'send',
  controller: 'monitor',
  webhook: 'plug',
}

export const CONNECTOR_NAMES: Record<ConnectorId, string> = {
  email: 'Email',
  slack: 'Slack',
  discord: 'Discord',
  mattermost: 'Mattermost',
  matrix: 'Matrix',
  googlechat: 'Google Chat',
  whatsapp: 'WhatsApp',
  imessage: 'iMessage',
  docs: 'Docs',
  browser: 'Browser',
  calendar: 'Calendar',
  telegram: 'Telegram',
  controller: 'Controller',
  webhook: 'Webhooks',
}

export const SOURCE_ORDER: ConnectorId[] = ['email', 'telegram', 'slack', 'whatsapp', 'imessage', 'calendar', 'docs', 'browser', 'controller', 'webhook']

export const FALLBACK_SOURCES: SourceRow[] = [
  { id: 'email', name: 'Email', label: 'Personal and work inboxes', icon: 'mail', status: 'mock', permissionStatus: 'granted', permissionDetail: 'Demo email source.', unread: 18, lastSync: 'demo data', lastActivity: 'demo data' },
  { id: 'telegram', name: 'Telegram', label: 'Bot control and room updates', icon: 'send', status: 'pending', permissionStatus: 'not_configured', permissionDetail: 'Telegram is not connected.', unread: 0, lastSync: 'never synced', lastActivity: 'no activity' },
  { id: 'slack', name: 'Slack', label: 'Channels and direct messages', icon: 'message', status: 'pending', permissionStatus: 'not_configured', permissionDetail: 'Slack is not connected.', unread: 7, lastSync: 'never synced', lastActivity: 'no activity' },
  { id: 'whatsapp', name: 'WhatsApp', label: 'Important chats and groups', icon: 'at', status: 'pending', permissionStatus: 'not_configured', permissionDetail: 'WhatsApp is not connected.', unread: 5, lastSync: 'never synced', lastActivity: 'no activity' },
  { id: 'imessage', name: 'iMessage', label: 'Local bridge opt-in only', icon: 'message', status: 'pending', permissionStatus: 'not_configured', permissionDetail: 'iMessage requires an explicit local bridge before messages can enter or leave TensorFold.', unread: 0, lastSync: 'never synced', lastActivity: 'no activity' },
  { id: 'calendar', name: 'Calendar', label: 'Meetings and commitments', icon: 'clock', status: 'pending', permissionStatus: 'not_configured', permissionDetail: 'Calendar is not connected.', unread: 0, lastSync: 'never synced', lastActivity: 'no activity' },
  { id: 'docs', name: 'Documents', label: 'Local files, PDFs, and notes', icon: 'file', status: 'mock', permissionStatus: 'granted', permissionDetail: 'Demo document index.', unread: 3, lastSync: 'indexed sample set', lastActivity: 'demo activity' },
  { id: 'browser', name: 'Browser', label: 'Explicit current-tab and history sources', icon: 'browser', status: 'pending', permissionStatus: 'not_configured', permissionDetail: 'Browser sources require explicit saved tabs or exports.', unread: 0, lastSync: 'never synced', lastActivity: 'no activity' },
]


export const priorityClass: Record<Priority, string> = {
  urgent: 'bg-danger/10 text-danger',
  soon: 'bg-warn/12 text-warn',
  watch: 'bg-ink-5 text-ink-3',
}
