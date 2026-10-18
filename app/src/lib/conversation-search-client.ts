import { normalizeConversationSearchFilters } from '../../shared/conversation-search-filters'
import type { ConversationSearchApi, ConversationSearchHighlight, ConversationSearchRequest, ConversationSearchResult } from '../../shared/conversation-search'

export interface ConversationSearchState {
  request: ConversationSearchRequest | null
  result: ConversationSearchResult | null
  busy: boolean
  error: string
}

export function searchDisplaySegments(text: string, highlights: ConversationSearchHighlight[]): Array<{ text: string; match: boolean }> {
  const ranges = highlights.map(({ start, end }) => ({ start: Math.max(0, Math.min(text.length, start)), end: Math.max(0, Math.min(text.length, end)) }))
    .filter(({ start, end }) => Number.isSafeInteger(start) && Number.isSafeInteger(end) && end > start).sort((a, b) => a.start - b.start)
  const merged: ConversationSearchHighlight[] = []
  for (const range of ranges) {
    const last = merged.at(-1)
    if (last && range.start <= last.end) last.end = Math.max(last.end, range.end)
    else merged.push({ ...range })
  }
  const segments: Array<{ text: string; match: boolean }> = []
  let offset = 0
  for (const range of merged) {
    if (range.start > offset) segments.push({ text: text.slice(offset, range.start), match: false })
    segments.push({ text: text.slice(range.start, range.end), match: true }); offset = range.end
  }
  if (offset < text.length) segments.push({ text: text.slice(offset), match: false })
  return segments
}

function validResult(value: ConversationSearchResult, request: ConversationSearchRequest): boolean {
  if (!value || value.scope !== request.scope || value.query !== request.query || !Array.isArray(value.matches) || value.matches.length > 100 || !Number.isSafeInteger(value.omitted) || value.omitted < 0 || typeof value.truncated !== 'boolean') return false
  return value.matches.every((match) => typeof match.threadId === 'string' && match.threadId.length > 0 && match.threadId.length <= 256
    && (request.scope !== 'thread' || match.threadId === request.threadId)
    && typeof match.title === 'string' && match.title.length <= 4096 && ['task', 'chat'].includes(match.kind)
    && (match.projectId === null || typeof match.projectId === 'string') && Number.isFinite(match.updatedAt)
    && (match.itemId === null || typeof match.itemId === 'string' && match.itemId.length <= 256)
    && ['title', 'timeline', 'history'].includes(match.source) && typeof match.excerpt === 'string' && match.excerpt.length <= 4096
    && Array.isArray(match.highlights) && match.highlights.length <= 100 && match.highlights.every((range) => Number.isSafeInteger(range.start) && Number.isSafeInteger(range.end) && range.start >= 0 && range.end > range.start && range.end <= match.excerpt.length))
}

/** IPC may finish after a query, scope or view changes; old results never own the UI. */
export function createConversationSearchClient(port: Pick<ConversationSearchApi, 'conversationSearch'>) {
  let epoch = 0
  let state: ConversationSearchState = { request: null, result: null, busy: false, error: '' }
  const listeners = new Set<(value: ConversationSearchState) => void>()
  const publish = (next: ConversationSearchState) => { state = next; listeners.forEach((listener) => listener(state)) }
  return {
    snapshot: () => state,
    subscribe(listener: (value: ConversationSearchState) => void) { listeners.add(listener); listener(state); return () => { listeners.delete(listener) } },
    clear() { epoch++; publish({ request: null, result: null, busy: false, error: '' }) },
    async search(input: ConversationSearchRequest) {
      const own = ++epoch, request = { ...input, query: input.query.trim().replace(/\s+/g, ' ') }
      if (!request.query) { publish({ request, result: null, busy: false, error: '' }); return }
      if (request.query.length > 512) { publish({ request, result: null, busy: false, error: 'Use a search query of at most 512 characters.' }); return }
      try { request.filters = normalizeConversationSearchFilters(input.filters) } catch { publish({ request, result: null, busy: false, error: 'Check the search dates and filters.' }); return }
      publish({ request, result: null, busy: true, error: '' })
      try {
        const result = await port.conversationSearch(request)
        if (own !== epoch) return
        if (!validResult(result, request)) throw new Error('Invalid local search result.')
        publish({ request, result, busy: false, error: '' })
      } catch {
        if (own === epoch) publish({ request, result: null, busy: false, error: 'Search could not finish. Your conversation and draft are unchanged; retry the search.' })
      }
    },
  }
}
