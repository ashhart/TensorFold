/** Filters use current conversation metadata, not guessed historical model use. */
export interface ConversationSearchFilters {
  /** Inclusive updated timestamp, in epoch milliseconds. */
  updatedFrom?: number
  /** Exclusive updated timestamp, in epoch milliseconds. */
  updatedBefore?: number
  providerId?: string
  model?: string
  /** A file explicitly attached to an active user message, including legacy image parts. */
  hasAttachments?: boolean
}

export interface ConversationSearchRequest {
  scope: 'all' | 'thread'
  query: string
  threadId?: string | null
  projectId?: string | null
  includeArchived?: boolean
  limit?: number
  filters?: ConversationSearchFilters
}

/** Offsets are UTF-16 ranges in the original display excerpt, with an exclusive end. */
export interface ConversationSearchHighlight { start: number; end: number }

export interface ConversationSearchMatch {
  threadId: string
  title: string
  kind: 'task' | 'chat'
  projectId: string | null
  updatedAt: number
  /** Null for title/history matches, which open without pretending to have a timeline anchor. */
  itemId: string | null
  source: 'title' | 'timeline' | 'history'
  /** Active saved API history ordinal, never an inferred visible timeline position. */
  messageIndex?: number
  excerpt: string
  highlights: ConversationSearchHighlight[]
  score: number
}

export interface ConversationSearchResult {
  query: string
  scope: 'all' | 'thread'
  matches: ConversationSearchMatch[]
  /** Additional verified matches in the bounded scan. */
  omitted: number
  /** More indexed candidates or live text may exist outside the scan budget. */
  truncated: boolean
}

export interface ConversationSearchApi {
  conversationSearch(request: ConversationSearchRequest): Promise<ConversationSearchResult>
  conversationSearchSetActiveThread(threadId: string | null): Promise<void>
}
