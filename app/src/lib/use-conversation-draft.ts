import type { MessageQuote } from '../../shared/message-quotes'
import { useCallback, useEffect, useMemo, useSyncExternalStore } from 'react'
import type { SetStateAction } from 'react'
import { composerDraftKey } from '../../shared/composer-drafts'
import type { ComposerDraftPort, ComposerDraftScope } from '../../shared/composer-drafts'
import type { Attachment } from './types'
import { conversationDraftsFor } from './conversation-drafts'
import type { ConversationDraftSendToken } from './conversation-drafts'
import { registerUpdatePreparationBlocker } from './update-safety'
import { forgetComposerFileSources, makeComposerFileSourcesIncognito } from './composer-file-sources'

export interface ComposerDraftFlushBridge {
  onComposerDraftFlush(callback: (requestId: string) => void): () => void
  acknowledgeComposerDraftFlush(input: { requestId: string; error?: string }): Promise<void>
  onComposerDraftReset?(callback: (scope: ComposerDraftScope | null) => void): () => void
}

/** Mount once at App level so even drafts whose composer is offscreen flush before Quit. */
export function useConversationDraftLifecycle(port: ComposerDraftPort | null, bridge: ComposerDraftFlushBridge | null): void {
  const drafts = useMemo(() => conversationDraftsFor(port), [port])
  useEffect(() => {
    const removeBlocker = registerUpdatePreparationBlocker(() => drafts.pendingReason())
    const removeFlush = bridge?.onComposerDraftFlush((requestId) => {
      void drafts.flushAll().then(
        () => bridge.acknowledgeComposerDraftFlush({ requestId }),
        () => bridge.acknowledgeComposerDraftFlush({ requestId, error: 'An unfinished message could not be saved. Keep TensorFold open and retry saving it.' }),
      ).catch(() => {})
    })
    const removeReset = bridge?.onComposerDraftReset?.((scope) => {
      if (!scope?.incognito) forgetComposerFileSources(scope ? composerDraftKey(scope) : undefined)
      if (scope?.incognito) { makeComposerFileSourcesIncognito(composerDraftKey({ ...scope, incognito: false }), composerDraftKey({ ...scope, incognito: true })); drafts.makeIncognito(scope) }
      else drafts.forget(scope)
    })
    return () => { removeBlocker(); removeFlush?.(); removeReset?.() }
  }, [drafts, bridge])
}

export function useConversationDraft(scope: ComposerDraftScope, port: ComposerDraftPort | null) {
  const key = composerDraftKey(scope)
  const stableScope = useMemo(() => ({ ...scope }), [key])
  const drafts = useMemo(() => conversationDraftsFor(port), [port])
  const subscribe = useCallback((listener: () => void) => drafts.subscribe(stableScope, listener), [drafts, stableScope])
  const getSnapshot = useCallback(() => drafts.get(stableScope), [drafts, stableScope])
  const draft = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  // Navigation flushes the old captured scope. The store retains it even if the component
  // unmounts while the new chat hydrates, and errors stay attached to that old chat.
  useEffect(() => () => { void drafts.flush(stableScope).catch(() => {}) }, [drafts, stableScope])
  const setText = useCallback((value: SetStateAction<string>) => {
    const current = drafts.get(stableScope).text
    drafts.update(stableScope, { text: typeof value === 'function' ? value(current) : value })
  }, [drafts, stableScope])
  const setAttachments = useCallback((value: SetStateAction<Attachment[]>) => {
    const current = drafts.get(stableScope).attachments
    drafts.update(stableScope, { attachments: typeof value === 'function' ? value(current) : value })
  }, [drafts, stableScope])
  const setQuotes = useCallback((value: SetStateAction<MessageQuote[]>) => {
    const current = drafts.get(stableScope).quotes ?? []
    drafts.update(stableScope, { quotes: typeof value === 'function' ? value(current) : value })
  }, [drafts, stableScope])
  return {
    ...draft, setText, setAttachments, setQuotes,
    capture: useCallback(() => drafts.capture(stableScope), [drafts, stableScope]),
    clearAccepted: useCallback((token: ConversationDraftSendToken) => drafts.clearAccepted(token), [drafts]),
    flush: useCallback(() => drafts.flush(stableScope), [drafts, stableScope]),
    retry: useCallback(() => drafts.retry(stableScope), [drafts, stableScope]),
    /** Async file selection can compare this token before merging processed results. */
    scopeKey: key,
  }
}
