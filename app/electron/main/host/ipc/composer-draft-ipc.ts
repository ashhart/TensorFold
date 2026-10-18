import { randomUUID } from 'node:crypto'
import type { WebContents } from 'electron'
import { validateComposerDraftScope } from '../../../../shared/composer-drafts'
import type { ComposerDraftScope, ComposerDraftWrite } from '../../../../shared/composer-drafts'
import { readComposerDraft, writeComposerDraft, clearComposerDraft, drainComposerDraftWrites } from '../../../store/composer-drafts'
import { store } from '../../state'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { assistantDraftScope, assistantSourceConfiguration, isAssistantOverviewDraft } from '../../../../shared/assistant-drafts'
import { connectorSourceBinding } from '../../../agent/connector-source-binding'

const pending = new Map<string, { sender: WebContents; resolve: () => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>()
const rendererLifecycle = new WeakMap<WebContents, { everLoaded: boolean }>()

function isInitialBlankUrl(url: string): boolean { return url === '' || url === 'about:blank' }

/** Register immediately after allocating the app window, before any load/navigation. */
export function observeComposerDraftRenderer(sender: WebContents): void {
  if (rendererLifecycle.has(sender)) return
  const lifecycle = { everLoaded: false }
  rendererLifecycle.set(sender, lifecycle)
  const inspect = () => {
    try { if (!isInitialBlankUrl(sender.getURL())) lifecycle.everLoaded = true }
    catch { lifecycle.everLoaded = true }
  }
  inspect()
  // Once a document was admitted, a later blank URL must not bypass the ACK fence.
  sender.on('did-start-navigation', (_event, url, _inPlace, isMainFrame) => {
    if (isMainFrame && !isInitialBlankUrl(url)) lifecycle.everLoaded = true
  })
  sender.on('did-finish-load', inspect)
}

function rendererNeverLoaded(sender: WebContents): boolean {
  const lifecycle = rendererLifecycle.get(sender)
  // An unobserved renderer may have already contained input, so fail closed.
  if (!lifecycle || lifecycle.everLoaded) return false
  try { return isInitialBlankUrl(sender.getURL()) }
  catch { return false }
}

export function validateOwnedDraftScope(scope: ComposerDraftScope): ComposerDraftScope {
  const valid = validateComposerDraftScope(scope)
  if (valid.incognito) throw new Error('Incognito drafts are memory-only.')
  if (valid.surface === 'assistant') {
    if (valid.hostId !== null) throw new Error('Assistant drafts belong to this desktop connector profile.')
    if (isAssistantOverviewDraft(valid, store.settings)) return valid
    const match = store.connectorEvents.some((event) => {
      // Stale cached material from a removed/replaced account cannot authorize its draft.
      for (const ref of event.sources ?? []) {
        if (!ref.sourceId) continue
        const source = assistantSourceConfiguration(store.settings, ref.connectorId, ref.sourceId)
        if (!source || ref.sourceRevision && connectorSourceBinding(source as { id: string }) !== ref.sourceRevision) return false
      }
      try { return assistantDraftScope(event.connectorId, event, store.settings).conversationId === valid.conversationId }
      catch { return false }
    })
    if (!match) throw new Error('Assistant conversation/account is unavailable or changed; keep your text and refresh the source.')
    return valid
  }
  if (valid.hostId !== null && !store.settings.remoteOrchestrators?.some((host) => host.id === valid.hostId)) throw new Error('Draft host is not configured in this desktop profile.')
  const thread = valid.conversationId === null ? null : store.threads.find((candidate) => candidate.id === valid.conversationId && (candidate.orchestratorHostId ?? null) === valid.hostId)
  if (valid.conversationId !== null && valid.hostId === null && !thread) throw new Error('Draft conversation is not available in this desktop profile.')
  if (thread?.incognito) throw new Error('Incognito drafts are memory-only.')
  return valid
}

export function registerComposerDraftIpc(): void {
  handle('composer-drafts:get', async (_event, scope: ComposerDraftScope) => {
    requireSecurityRole('operator', 'read an unfinished message')
    return readComposerDraft(validateOwnedDraftScope(scope))
  })
  handle('composer-drafts:save', async (_event, input: ComposerDraftWrite) => {
    requireSecurityRole('operator', 'save an unfinished message')
    return writeComposerDraft({ ...input, scope: validateOwnedDraftScope(input?.scope) }, () => { validateOwnedDraftScope(input?.scope) })
  })
  handle('composer-drafts:clear', async (_event, input: { scope: ComposerDraftScope; expectedRevision: number }) => {
    requireSecurityRole('operator', 'clear an accepted message draft')
    return clearComposerDraft({ ...input, scope: validateOwnedDraftScope(input?.scope) }, () => { validateOwnedDraftScope(input?.scope) })
  })
  handle('composer-drafts:flushed', (event, input: { requestId: string; error?: string }) => {
    const waiting = pending.get(input?.requestId)
    if (!waiting || waiting.sender !== event.sender) throw new Error('Unknown draft flush request.')
    pending.delete(input.requestId)
    clearTimeout(waiting.timer)
    if (input.error) waiting.reject(new Error('An unfinished message could not be saved. Keep TensorFold open and retry saving it.'))
    else waiting.resolve()
  })
}

/** Call from an awaited close/quit guard before destroying the renderer or its IPC bridge. */
export async function requestRendererDraftFlush(sender: WebContents | null, timeoutMs = 15_000): Promise<void> {
  if (sender && !sender.isDestroyed() && !rendererNeverLoaded(sender)) {
    const requestId = randomUUID()
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(requestId)
        reject(new Error('TensorFold could not confirm that your unfinished message was saved. Keep the window open and try again.'))
      }, timeoutMs)
      pending.set(requestId, { sender, resolve, reject, timer })
      try { sender.send('composer-drafts:flush', requestId) }
      catch { clearTimeout(timer); pending.delete(requestId); reject(new Error('The message window is unavailable; draft saving could not be confirmed.')) }
    })
  }
  await drainComposerDraftWrites()
}
