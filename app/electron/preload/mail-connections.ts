import type { MailConnectionsApi } from '../../shared/mail-connections'
export function mailConnectionsPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): MailConnectionsApi {
  return { mailConnections: () => invoke('mailConnections:status'), testMailConnection: target => invoke('mailConnections:test', target) }
}
