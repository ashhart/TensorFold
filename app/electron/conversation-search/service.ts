import { conversationMatchesFilters, normalizeConversationSearchFilters } from '../../shared/conversation-search-filters'
import type { ConversationSearchMatch, ConversationSearchRequest, ConversationSearchResult } from '../../shared/conversation-search'
import type { SessionSearchOptions } from '../agent/session-recall'
import type { Thread, TimelineItem } from '../agent/types'
import { contentText } from '../agent/content'
import { matchingTerms, MAX_SEARCH_QUERY_CHARS, normalizeSearchText, searchExcerpt, searchTerms } from './text'

const MAX_THREADS = 1_000
const MAX_ITEMS = 10_000
const MAX_SCAN_CHARS = 2_000_000
const MAX_ITEM_CHARS = 128_000

export interface ConversationSearchDependencies {
  /** Must return only the current profile's live, host-owned threads. */
  getThreads(): readonly Thread[]
  /** Hashed candidates are hints only, checked again against the permitted live objects. */
  findCandidates?(threads: Thread[], options: SessionSearchOptions): { threads: Thread[]; truncated: boolean }
}

export interface ConversationSearchAccess {
  /** Supplied by the host binding, never copied from request.threadId. */
  allowedThreadId?: string | null
}

export function localSearchThread(thread: Thread): boolean {
  return !thread.orchestratorHostId
}

export function validateConversationSearchRequest(input: unknown): ConversationSearchRequest {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Invalid conversation search request.')
  const value = input as Record<string, unknown>
  if (value.scope !== 'all' && value.scope !== 'thread') throw new Error('Invalid conversation search scope.')
  if (typeof value.query !== 'string' || value.query.length > MAX_SEARCH_QUERY_CHARS) throw new Error('Search query must be at most 512 characters.')
  for (const key of ['threadId', 'projectId']) {
    if (value[key] != null && (typeof value[key] !== 'string' || (value[key] as string).length > 256)) throw new Error('Invalid conversation search filter.')
  }
  if (value.includeArchived !== undefined && typeof value.includeArchived !== 'boolean') throw new Error('Invalid conversation search filter.')
  if (value.limit !== undefined && (typeof value.limit !== 'number' || !Number.isFinite(value.limit))) throw new Error('Invalid conversation search limit.')
  const threadId = typeof value.threadId === 'string' ? value.threadId.trim() : null
  if ((value.scope === 'thread' && !threadId) || (value.scope === 'all' && threadId)) throw new Error('A thread search requires an explicit current conversation.')
  return {
    scope: value.scope,
    query: value.query.trim().replace(/\s+/g, ' '),
    threadId,
    projectId: typeof value.projectId === 'string' ? value.projectId.trim() || null : null,
    includeArchived: value.includeArchived === true,
    filters: normalizeConversationSearchFilters(value.filters),
    limit: value.limit === undefined ? 50 : Math.max(1, Math.min(100, Math.floor(value.limit as number))),
  }
}

function timelineText(item: TimelineItem): string {
  return [item.title, item.tool, item.text, item.command, item.meta, item.output].filter(Boolean).join('\n')
}

/** Revalidate after an asynchronous authority check, before any computed excerpts leave main. */
export function conversationSearchMatchIsCurrent(match: ConversationSearchMatch, request: ConversationSearchRequest, threads: readonly Thread[]): boolean {
  const thread = threads.find((candidate) => candidate.id === match.threadId && localSearchThread(candidate))
  if (!thread || !conversationMatchesFilters(thread, request.filters) || (request.scope === 'all' && (thread.incognito || thread.phoneLocalOnly || thread.googleLocalOnly || thread.bot
    || (!request.includeArchived && thread.archivedAt))) || (request.projectId && thread.projectId !== request.projectId)) return false
  let text = ''
  if (match.source === 'title') text = thread.title || ''
  else if (match.source === 'timeline') {
    const item = thread.timeline.find((candidate) => candidate.id === match.itemId)
    if (!item) return false
    text = timelineText(item)
  } else {
    const message = thread.apiMessages[match.messageIndex ?? -1]
    if (!message || (message.role !== 'user' && message.role !== 'assistant')) return false
    text = contentText(message.content)
  }
  const terms = searchTerms(request.query)
  const current = searchExcerpt(text.slice(0, MAX_ITEM_CHARS), terms)
  return matchingTerms(text, terms).length === terms.length && current.excerpt === match.excerpt
    && JSON.stringify(current.highlights) === JSON.stringify(match.highlights)
}

/** Main-owned lexical search; it never invokes a provider, agent, tool or persistent write. */
export function createConversationSearchService(deps: ConversationSearchDependencies) {
  return {
    async conversationSearch(input: ConversationSearchRequest, access: ConversationSearchAccess = {}): Promise<ConversationSearchResult> {
      const request = validateConversationSearchRequest(input)
      const result: ConversationSearchResult = { query: request.query, scope: request.scope, matches: [], omitted: 0, truncated: false }
      const live = deps.getThreads()
      let visible: Thread[]
      if (request.scope === 'thread') {
        if (!access.allowedThreadId || access.allowedThreadId !== request.threadId) throw new Error('Current conversation search is not authorized.')
        const current = live.find((thread) => thread.id === access.allowedThreadId && localSearchThread(thread))
        if (!current) throw new Error('The current conversation is no longer available.')
        // Explicit current-chat scan is ephemeral, including an incognito/current guarded chat.
        visible = [current]
      } else {
        visible = live.filter((thread) => localSearchThread(thread) && !thread.incognito && !thread.phoneLocalOnly && !thread.googleLocalOnly && !thread.bot)
      }
      visible = visible.filter((thread) => conversationMatchesFilters(thread, request.filters) && (!request.projectId || thread.projectId === request.projectId)
        && (request.scope === 'thread' || request.includeArchived || !thread.archivedAt))
      const terms = searchTerms(request.query)
      if (!terms.length) return result
      if (request.scope === 'all' && deps.findCandidates) {
        const permitted = new Set(visible)
        const narrowed = deps.findCandidates(visible, { ...request, includeCurrent: true })
        visible = narrowed.threads.filter((thread) => permitted.has(thread))
        result.truncated = narrowed.truncated
      }
      if (visible.length > MAX_THREADS) result.truncated = true
      visible = visible.slice(0, MAX_THREADS)
      let items = 0, scannedChars = 0
      const matches: Array<ConversationSearchMatch & { order: number }> = []
      const addMatch = (thread: Thread, text: string, itemId: string | null, order: number, messageIndex?: number) => {
        const remaining = MAX_SCAN_CHARS - scannedChars
        const bounded = text.slice(0, Math.min(MAX_ITEM_CHARS, remaining))
        if (bounded.length < text.length) result.truncated = true
        scannedChars += bounded.length
        if (!bounded || matchingTerms(bounded, terms).length !== terms.length) return
        const phrase = normalizeSearchText(bounded).includes(normalizeSearchText(request.query))
        matches.push({
          threadId: thread.id, title: (thread.title || 'Untitled conversation').slice(0, 240), kind: thread.kind,
          projectId: thread.projectId ?? null, updatedAt: thread.updatedAt || thread.createdAt || 0,
          itemId, source: messageIndex !== undefined ? 'history' : itemId === null ? 'title' : 'timeline',
          ...(messageIndex !== undefined ? { messageIndex } : {}),
          ...searchExcerpt(bounded, terms), score: terms.length + (phrase ? 4 : 0) + (itemId === null ? 1 : 0), order,
        })
      }
      scan: for (const thread of visible) {
        if (request.scope === 'all') addMatch(thread, thread.title || '', null, -1)
        const displayed = new Set<string>()
        // Only the active timeline yields anchors; superseded edit ledgers and API-only text
        // are never guessed onto a current item, even when the words happen to be identical.
        for (let index = 0; index < (thread.timeline ?? []).length; index++) {
          if (items >= MAX_ITEMS || scannedChars >= MAX_SCAN_CHARS) { result.truncated = true; break scan }
          const item = thread.timeline[index]
          items++
          if (item.kind === 'user' || item.kind === 'assistant') displayed.add(`${item.kind}\0${item.text ?? ''}`)
          if (item.id) addMatch(thread, timelineText(item), item.id, index)
        }
        // Legacy active user/assistant text can lack a display row after import/compaction.
        // Return an honest history-only result, never system/tool context or a guessed anchor.
        for (let index = 0; index < (thread.apiMessages ?? []).length; index++) {
          if (items >= MAX_ITEMS || scannedChars >= MAX_SCAN_CHARS) { result.truncated = true; break scan }
          const message = thread.apiMessages[index]
          items++
          if (message.role !== 'user' && message.role !== 'assistant') continue
          const text = contentText(message.content)
          if (!displayed.has(`${message.role}\0${text}`)) addMatch(thread, text, null, thread.timeline.length + index, index)
        }
      }
      matches.sort(request.scope === 'thread'
        ? (a, b) => a.order - b.order
        : (a, b) => b.score - a.score || b.updatedAt - a.updatedAt || a.threadId.localeCompare(b.threadId) || a.order - b.order)
      result.matches = matches.slice(0, request.limit).map(({ order: _order, ...match }) => match)
      result.omitted = Math.max(0, matches.length - result.matches.length)
      return result
    },
  }
}
