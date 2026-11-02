import { bindGmailInboxRun } from './gmail-inbox-run'
import { googleWorkspaceService } from './google-workspace-runtime'
import { runtimeProvider } from './provider-runtime'
import { engineService } from './engine-runtime'
import { requireSecurityRole } from './audit'
import { saveThreads } from '../../store'
import { store } from '../state'
import type { StartRunPayload } from '../../agent/types'

export function prepareGmailInboxRun(payload: StartRunPayload): Promise<void> {
  return bindGmailInboxRun(payload, {
    authorize: () => requireSecurityRole('owner', 'use Gmail context in a local task'),
    selected: (id, revision) => googleWorkspaceService().assertInboxSelected(id, revision),
    thread: id => store.threads.find(thread => thread.id === id),
    provider: thread => {
      const provider = store.settings.providers.find(item => item.id === thread.providerId)
      return provider ? runtimeProvider(provider, thread.providerAccountId) : undefined
    },
    externalResearch: () => engineService()?.hasExternalResearchEnabled() === true,
    persist: () => saveThreads(store.threads),
  })
}
