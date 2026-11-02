import { registerInputHistoryIpc } from '../host/ipc/input-history-ipc'
import { registerMessageQuoteIpc } from '../host/ipc/message-quote-ipc'
import { reviewedRunOutcome } from '../../agent/run-outcome-review'
import type { RunOutcomeReviewInput } from '../../../shared/run-recovery-api'
import { registerReadableExportRuntime } from '../host/readable-export-runtime'
import { registerDocumentLibraryRuntime } from '../host/document-library-runtime'
import { registerConversationSearchRuntime } from '../host/conversation-search-runtime'
import { saveThreads } from '../../store'
import { store } from '../state'
import { handle } from '../host/ipc'
import { requireSecurityRole } from '../host/audit'
import { WorkspaceMutationGuard } from '../host/workspace-mutation-guard'
import { sendToRenderer } from '../host/renderer-events'
import { computerHelperPath } from '../host/computer-use-host'
import { registerComposerDraftIpc } from '../host/ipc/composer-draft-ipc'
import { createAttachmentTextService } from '../host/attachment-text'

interface WorkspaceContentDeps {
  aborters: ReadonlyMap<string, AbortController>
  workspaceMutationGuard: WorkspaceMutationGuard
}

export function registerApplicationWorkspaceContent(deps: WorkspaceContentDeps) {
  const { aborters, workspaceMutationGuard } = deps
  registerComposerDraftIpc()
  handle('runRecovery:review', async (_event, input: RunOutcomeReviewInput) => {
    requireSecurityRole('operator', 'review an uncertain run outcome')
    const lease = workspaceMutationGuard.reserveStart({ threadId: input?.threadId })
    try {
      const thread = store.threads.find(thread => thread.id === input?.threadId && !thread.orchestratorHostId)
      if (!thread) throw new Error('Choose a local conversation.')
      const next = reviewedRunOutcome(thread, input, aborters.has(thread.id))
      requireSecurityRole('operator', 'review an uncertain run outcome')
      await saveThreads(store.threads.map(item => item === thread ? next : item))
      requireSecurityRole('operator', 'review an uncertain run outcome')
      if (!store.threads.includes(thread) || thread.runRecovery?.runId !== input.runId || aborters.has(thread.id)) throw new Error('The run changed; review its current outcome.')
      thread.runRecoveryReviewedId = input.runId
      sendToRenderer('agent:event', { runId: input.runId, threadId: thread.id, type: 'thread_update', patch: { runRecoveryReviewedId: input.runId } })
      return { recovery: thread.runRecovery!, reviewedRunId: input.runId }
    } finally { lease.release() }
  })
  registerConversationSearchRuntime()
  registerInputHistoryIpc({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['input:history'] }) })
  registerMessageQuoteIpc({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['input:quotes'] }) })
  registerReadableExportRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['export:readable'] }) })
  registerDocumentLibraryRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['documents:library'] }) })
  const attachmentTextService = createAttachmentTextService({ helperPath: computerHelperPath })
  handle('attachments:textCapability', () => { requireSecurityRole('operator', 'check local text extraction'); return attachmentTextService.getCapability() })
  handle('attachments:extractText', (_event, input) => { requireSecurityRole('operator', 'read text from a selected file'); return attachmentTextService.extract(input) })

  return attachmentTextService
}
