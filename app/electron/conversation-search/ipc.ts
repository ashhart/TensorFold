import type { ConversationSearchRequest } from '../../shared/conversation-search'
import { conversationSearchMatchIsCurrent, createConversationSearchService, localSearchThread, type ConversationSearchDependencies } from './service'

export interface ConversationSearchSender {
  once?(event: 'destroyed', listener: () => void): unknown
}

export interface ConversationSearchIpcDependencies<Event extends { sender: ConversationSearchSender }> extends ConversationSearchDependencies {
  /** The application's trusted-frame handle(), never ipcMain.handle directly. */
  handle(channel: string, listener: (event: Event, ...args: any[]) => unknown): void
  authorize(event: Event): void | Promise<void>
  /** A current namespace identity; changing profiles invalidates all older bindings. */
  getProfileKey(): string
}

/** Pure registration with injected host authority; intentionally unavailable to remote routes. */
export function registerConversationSearchIpc<Event extends { sender: ConversationSearchSender }>(deps: ConversationSearchIpcDependencies<Event>) {
  const bindings = new WeakMap<ConversationSearchSender, { threadId: string; profileKey: string }>()
  const observed = new WeakSet<ConversationSearchSender>()
  const service = createConversationSearchService(deps)
  const authorize = async (event: Event) => {
    try { await deps.authorize(event) } catch (error) { bindings.delete(event.sender); throw error }
  }
  deps.handle('conversation:search:set-active-thread', async (event, threadId: unknown) => {
    await authorize(event)
    // A refused navigation must also discard the previous privileged binding.
    bindings.delete(event.sender)
    if (threadId === null) return
    if (typeof threadId !== 'string' || !threadId || threadId.length > 256) throw new Error('Invalid active conversation.')
    const current = deps.getThreads().find((thread) => thread.id === threadId && localSearchThread(thread))
    if (!current) throw new Error('The active conversation is unavailable.')
    bindings.set(event.sender, { threadId, profileKey: deps.getProfileKey() })
    if (!observed.has(event.sender)) {
      observed.add(event.sender)
      event.sender.once?.('destroyed', () => bindings.delete(event.sender))
    }
  })
  deps.handle('conversation:search', async (event, request: ConversationSearchRequest) => {
    await authorize(event)
    const profileKey = deps.getProfileKey()
    const binding = bindings.get(event.sender)
    const allowedThreadId = binding?.profileKey === profileKey ? binding.threadId : null
    if (!allowedThreadId) bindings.delete(event.sender)
    const result = await service.conversationSearch(request, { allowedThreadId })
    await authorize(event)
    if (deps.getProfileKey() !== profileKey) throw new Error('Search profile changed; search again.')
    if (result.scope === 'thread' && (bindings.get(event.sender) !== binding
      || !deps.getThreads().some((thread) => thread.id === allowedThreadId && localSearchThread(thread)))) {
      throw new Error('Current conversation changed; search again.')
    }
    const live = deps.getThreads()
    if (!result.matches.every((match) => conversationSearchMatchIsCurrent(match, request, live))) throw new Error('Conversation changed; search again.')
    return result
  })
  return { clearActiveThread: (sender: ConversationSearchSender) => bindings.delete(sender) }
}
