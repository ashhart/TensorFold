import type { quickComplete } from '../../agent/openai'
import type { AgentEvent, Provider, Thread } from '../../agent/types'
import { ensureTensorFoldStudioStylePrompt } from '../../agent/prompt-style'

export function createThreadTitleGenerator(deps: {
  completeTitle: typeof quickComplete
  reserveOperation(): { release(): void }
  isCurrentThread(thread: Thread): boolean
  persistThreads(): void
  emitToRenderer(event: AgentEvent): void
}) {
  return async function maybeAutoTitle(thread: Thread, provider: Provider, model: string, userText: string): Promise<void> {
    if (thread.googleLocalOnly) return
    let reservation: { release(): void } | null = null
    try {
      reservation = deps.reserveOperation()
      const initialTitle = thread.title
      if (!deps.isCurrentThread(thread) || !['New task', 'New chat'].includes(initialTitle)) return
      if (provider?.transport === 'codex-cli' || provider?.transport === 'claude-cli') {
        const clean = userText.replace(/\s+/g, ' ').trim().split(/\s+/).slice(0, 5).join(' ').slice(0, 48)
        if (clean) {
          thread.title = clean
          deps.persistThreads()
          deps.emitToRenderer({ runId: '', threadId: thread.id, type: 'title', title: clean })
        }
        return
      }
      const title = await deps.completeTitle({
        provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model,
        messages: [
          { role: 'system', content: ensureTensorFoldStudioStylePrompt('Reply with a 2-5 word title (Title Case, no quotes, no period) for this conversation.') },
          { role: 'user', content: userText.slice(0, 500) },
        ],
        approvalMode: 'readonly',
        assertRequestAllowed: () => { if (thread.googleLocalOnly) throw new Error('Google account chats do not use automatic title models.') },
      })
      const clean = title.replace(/^["']|["'.]+$/g, '').slice(0, 48)
      if (clean && !thread.googleLocalOnly && deps.isCurrentThread(thread) && thread.title === initialTitle) {
        thread.title = clean
        deps.persistThreads()
        deps.emitToRenderer({ runId: '', threadId: thread.id, type: 'title', title: clean })
      }
    } catch { /* titling is best-effort */ } finally {
      reservation?.release()
    }
  }
}
