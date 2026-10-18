import type { ConversationSearchApi } from '../../shared/conversation-search'

export function conversationSearchPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): ConversationSearchApi {
  return {
    conversationSearch: request => invoke('conversation:search', request),
    conversationSearchSetActiveThread: id => invoke('conversation:search:set-active-thread', id),
  }
}
