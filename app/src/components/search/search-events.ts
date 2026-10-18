import type { ConversationSearchMatch } from '../../../shared/conversation-search'

export const OPEN_CONVERSATION_SEARCH = 'tensorfold:open-conversation-search'
export const REVEAL_CONVERSATION_MATCH = 'tensorfold:reveal-conversation-match'
export interface SearchOpenRequest { scope: 'all' | 'thread'; threadId?: string; query?: string }
export interface SearchRevealRequest { threadId: string; itemId: string; nonce: number; expiresAt: number; origin?: 'search' | 'outline' | 'quote' }
let requested: SearchOpenRequest | null = null
let target: SearchRevealRequest | null = null
let nonce = 0

export function requestConversationSearch(input: SearchOpenRequest = { scope: 'all' }): void {
  requested = { ...input }
  window.dispatchEvent(new Event(OPEN_CONVERSATION_SEARCH))
}
export function consumeConversationSearchRequest(): SearchOpenRequest | null { const value = requested; requested = null; return value }
export function revealConversationMatch(match: ConversationSearchMatch): void {
  if (!match.itemId) return
  revealConversationItem({ threadId: match.threadId, itemId: match.itemId }, 'search')
}
export function revealConversationItem(item: { threadId: string; itemId: string }, origin: 'search' | 'outline' | 'quote' = 'outline'): void {
  if (!item.threadId || !item.itemId) return
  target = { ...item, origin, nonce: ++nonce, expiresAt: Date.now() + 15_000 }
  window.dispatchEvent(new Event(REVEAL_CONVERSATION_MATCH))
}
export function consumeConversationSearchTarget(threadId: string): SearchRevealRequest | null {
  if (target && target.expiresAt < Date.now()) target = null
  if (target?.threadId !== threadId) return null
  const value = target; target = null; return value
}
