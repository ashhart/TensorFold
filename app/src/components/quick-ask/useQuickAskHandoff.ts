import { useEffect } from 'react'
import { useStore } from '../../lib/store'
import { conversationDraftsFor } from '../../lib/conversation-drafts'
import { desktopDraftPort } from '../../lib/desktop-drafts'

export function useQuickAskHandoff(ready: boolean): void {
  useEffect(() => {
    const launcher = window.tensorfoldQuickAskLauncher
    if (!launcher || !ready) return
    let mounted = true
    const accept = async () => {
      const next = await launcher.consumeHandoff()
      if (!mounted || !next) return
      const { thread, draft } = next
      if (draft && (draft.text || draft.attachments.length)) {
        const scope = { surface: 'chat' as const, conversationId: thread.id, hostId: thread.orchestratorHostId ?? null, incognito: false }
        const drafts = conversationDraftsFor(desktopDraftPort)
        drafts.update(scope, { text: draft.text, attachments: draft.attachments })
        await drafts.flush(scope)
      }
      const state = useStore.getState()
      state.applyThreadUpdate(thread); state.openThread(thread.id)
    }
    const receive = () => { void accept().catch(() => useStore.getState().setNotice({ kind: 'error', message: 'The Ask TensorFold conversation was saved, but its draft could not be opened. Reopen Ask TensorFold to recover it.' })) }
    const off = launcher.onHandoff(receive)
    receive()
    return () => { mounted = false; off() }
  }, [ready])
}
