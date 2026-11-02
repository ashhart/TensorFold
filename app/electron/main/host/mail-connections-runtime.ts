import { MailConnectionHealthService, type MailHealthRecord } from '../../agent/mail-connection-health'
import type { MailConnectionAccount } from '../../../shared/mail-connections'
import { googleWorkspaceService } from './google-workspace-runtime'
import { microsoftMailService } from './microsoft-mail-runtime'
import { readProtectedJson, writeProtectedJson } from '../../store/write-infra'
import { vontraPath } from '../../store/paths'
import { store } from '../state'

let health: MailConnectionHealthService | undefined
export function mailConnectionsService(): MailConnectionHealthService {
  if (health) return health
  const file = vontraPath('state', 'mail-connection-health.protected.json')
  health = new MailConnectionHealthService({
    async accounts() {
      const [google, microsoft, inbox] = await Promise.all([googleWorkspaceService().snapshot(), microsoftMailService().connections(), googleWorkspaceService().inboxSnapshot()])
      const accounts: MailConnectionAccount[] = google.accounts.map(account => ({
        provider: 'gmail', id: account.id, email: account.email, revision: String(account.revision),
        authorization: account.status === 'connected' ? 'connected' : account.status === 'needs-reauthorization' ? 'reconnect' : 'disconnected',
        read: account.capabilities.includes('gmail.read'), send: account.capabilities.includes('gmail.send'), inbox: inbox.accounts.some(item => item.id === account.id && item.enabled),
      }))
      for (const source of store.settings.email.sources.filter(item => item.kind === 'microsoft_graph')) {
        const account = microsoft.find(item => item.id === source.id && item.revision === source.microsoftRevision && item.email === source.user)
        accounts.push({
          provider: 'microsoft', id: source.id, email: source.user ?? '', revision: `${source.microsoftRevision}:${source.enabled}`,
          authorization: account && source.enabled ? 'connected' : 'disconnected', read: account?.read ?? false, send: account?.send ?? false, inbox: source.enabled, sourceId: source.id
        })
      }
      return { googleConfigured: google.clientConfigured, accounts }
    },
    async probe(account) {
      if (account.provider === 'gmail') {
        await googleWorkspaceService().listMessages(account.id, { limit: 1, query: 'in:inbox' })
      } else {
        const source = store.settings.email.sources.find(item => item.id === account.id && item.enabled && item.kind === 'microsoft_graph')
        if (!source || `${source.microsoftRevision}:${source.enabled}` !== account.revision) throw new Error('Microsoft connection changed before testing.')
        await microsoftMailService().list({ ...source }, 1)
      }
    },
    store: { read: () => readProtectedJson<MailHealthRecord[]>(file, [], false, true), write: records => writeProtectedJson(file, records) },
  })
  return health
}
