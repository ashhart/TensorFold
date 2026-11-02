import type { MailConnectionsApi } from '../../../shared/mail-connections'
export function createMockMailConnectionsApi(): MailConnectionsApi {
  return {
    mailConnections: async () => ({ googleConfigured: false, accounts: [] }),
    testMailConnection: async () => { throw new Error('Mail connection checks require a connected desktop account.') },
  }
}
