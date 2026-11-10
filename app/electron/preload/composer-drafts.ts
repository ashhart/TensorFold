import type { ComposerDraftScope, ComposerDraftWrite } from '../../shared/composer-drafts'

type Invoke = (channel: string, ...args: unknown[]) => Promise<any>
type Subscribe = <T>(channel: string, listener: (event: unknown, payload: T) => void) => () => void

export function composerDraftsPreload(invoke: Invoke, subscribe: Subscribe) {
  return {
    getComposerDraft: (scope: ComposerDraftScope) => invoke('composer-drafts:get', scope),
    saveComposerDraft: (input: ComposerDraftWrite) => invoke('composer-drafts:save', input),
    clearComposerDraft: (input: { scope: ComposerDraftScope; expectedRevision: number }) => invoke('composer-drafts:clear', input),
    onComposerDraftFlush: (cb: (requestId: string) => void) => subscribe<string>('composer-drafts:flush', (_e, requestId) => cb(requestId)),
    onComposerDraftReset: (cb: (scope: ComposerDraftScope | null) => void) => subscribe<ComposerDraftScope | null>('composer-drafts:reset', (_e, scope) => cb(scope)),
    acknowledgeComposerDraftFlush: (input: { requestId: string; error?: string }): Promise<void> => invoke('composer-drafts:flushed', input),
  }
}
