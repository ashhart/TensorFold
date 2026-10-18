/** Display and setup metadata only; availability and permissions come from the host. */
export const CONNECTOR_CATALOG = {
  google: { name: 'Google', tag: 'Gmail & Calendar', description: 'Connect your account to search mail, draft replies and manage your schedule.' },
  email: { name: 'Email', tag: 'Mailbox', description: 'Read selected mailboxes and review outgoing replies.' },
  slack: { name: 'Slack', tag: 'Workspace', description: 'Choose the channels TensorFold may read.' },
  discord: { name: 'Discord', tag: 'Bot', description: 'Connect selected server channels and threads.' },
  telegram: { name: 'Telegram', tag: 'Bot', description: 'Use a bot with explicitly allowed chats and senders.' },
  whatsapp: { name: 'WhatsApp', tag: 'Link your phone', description: 'Scan a Linked Devices QR code; no Business API key needed.' },
  imessage: { name: 'iMessage', tag: 'This Mac', description: 'Allow local access and choose the contacts to sync.' },
  mattermost: { name: 'Mattermost', tag: 'Workspace', description: 'Connect a server and choose channels to sync.' },
  matrix: { name: 'Matrix', tag: 'Homeserver', description: 'Connect selected rooms through your account.' },
  googlechat: { name: 'Google Chat', tag: 'Workspace', description: 'Connect selected Google Chat spaces.' },
  docs: { name: 'Documents', tag: 'Local folders', description: 'Choose the folders the assistant may index.' },
  browser: { name: 'Browser', tag: 'Selected context', description: 'Share a tab or a history export explicitly.' },
  calendar: { name: 'Calendar', tag: 'Schedule', description: 'Read configured calendars and review changes.' },
  controller: { name: 'Controller', tag: 'This Mac', description: 'Use approved local computer actions.' },
  webhook: { name: 'Webhooks', tag: 'Advanced', description: 'Receive verified events through configured routes.' },
} as const

export type ConnectorCatalogId = keyof typeof CONNECTOR_CATALOG
export const CONNECTOR_SETUP_IDS = ['google', 'whatsapp', 'email', 'imessage', 'slack', 'discord', 'telegram', 'mattermost', 'matrix', 'googlechat', 'docs', 'browser'] as const satisfies readonly ConnectorCatalogId[]
export type ConnectorSetupId = typeof CONNECTOR_SETUP_IDS[number]
export function isConnectorSetupId(value: unknown): value is ConnectorSetupId {
  return typeof value === 'string' && (CONNECTOR_SETUP_IDS as readonly string[]).includes(value)
}
