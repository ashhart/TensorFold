import { registerConversationSearchIpc } from '../../conversation-search/ipc'
import { selectSessionSearchCandidates } from '../../store/session-search-index'
import { blindSearchToken } from '../../store/app-crypto'
import { vontraHomeDir, vontraPath } from '../../store'
import { store } from '../state'
import { handle } from './ipc'
import { requireSecurityRole } from './audit'

export function registerConversationSearchRuntime(): void {
  registerConversationSearchIpc({
    handle,
    authorize: () => requireSecurityRole('operator', 'search local conversations'),
    getThreads: () => store.threads,
    getProfileKey: vontraHomeDir,
    findCandidates: (threads, options) => selectSessionSearchCandidates(vontraPath('state', 'session-search.sqlite'), threads, options, blindSearchToken),
  })
}
