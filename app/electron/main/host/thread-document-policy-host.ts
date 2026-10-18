import type { Project, Settings, Thread } from '../../agent/types'
import { isExternalProvider } from '../../../shared/provider-trust'
import { validateDocumentSourceIds } from '../../../shared/document-collections'
import { vontraHomeDir } from '../../store'
import { saveThreadDocumentCollections } from '../../store/document-collections'
import type { WorkspaceMutationGuard } from './workspace-mutation-guard'

/** Selection is staged durably before publication; failed writes never expose a new live policy. */
export function createThreadDocumentPolicyHost(deps: {
  threads(): Thread[]
  settings(): Settings
  projects(): Project[]
  aborters: Map<string, AbortController>
  guard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  save?: typeof saveThreadDocumentCollections
  authorize?: () => void
  profile?: () => string
}) {
  return async (id: string, input: unknown): Promise<Thread> => {
    const requested = validateDocumentSourceIds(input)
    if (!requested) throw new Error('Choose document collections explicitly; an empty list permits none.')
    const profile = (deps.profile ?? vontraHomeDir)()
    const thread = deps.threads().find(thread => thread.id === id)
    if (!thread || thread.archivedAt) throw new Error('Choose an active chat.')
    const validate = () => {
      deps.authorize?.()
      if ((deps.profile ?? vontraHomeDir)() !== profile || !deps.threads().includes(thread)) throw new Error('The document collection owner changed; try again.')
      if (thread.status === 'running' || deps.aborters.has(id)) throw new Error('Stop this run before changing document collections.')
      const workspace = thread.workspace ?? deps.projects().find(project => project.id === thread.projectId)?.workspace
      const providerId = thread.providerId ?? deps.settings().activeProviderId
      const provider = deps.settings().providers.find(provider => provider.id === providerId)
      if (thread.bot || thread.orchestratorHostId || workspace?.kind === 'ssh' || thread.phoneLocalOnly || thread.googleLocalOnly || provider && isExternalProvider(provider)) throw new Error('Document collections can be selected only for an eligible local chat.')
      if (requested.some(id => !deps.settings().documents.sources.some(source => source.id === id && source.enabled))) throw new Error('Choose enabled document collections configured in this profile.')
    }
    validate()
    const lease = deps.guard.reserveStart({ threadId: id })
    try {
      const accepted = await (deps.save ?? saveThreadDocumentCollections)(deps.threads, id, requested, validate)
      validate()
      return accepted
    } finally { lease.release() }
  }
}
