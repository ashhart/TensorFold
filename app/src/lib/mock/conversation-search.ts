import { conversationMatchesFilters, normalizeConversationSearchFilters } from '../../../shared/conversation-search-filters'
import { matchingTerms, searchExcerpt, searchTerms } from '../../../shared/conversation-search-text'
import type { ConversationSearchApi, ConversationSearchMatch } from '../../../shared/conversation-search'
import type { Thread } from '../types'

/** Synthetic browser data only; production search is owned by main. */
export function createMockConversationSearchApi(threads: () => Thread[]): ConversationSearchApi {
  let active: string | null = null
  return {
    async conversationSearchSetActiveThread(id) { active = id },
    async conversationSearch(request) {
      const filters = normalizeConversationSearchFilters(request.filters)
      const query = request.query.trim().replace(/\s+/g, ' ')
      if (request.scope === 'thread' && (!active || active !== request.threadId)) throw new Error('Choose the current conversation first.')
      const visible = threads().filter(thread => !thread.orchestratorHostId && conversationMatchesFilters(thread, filters)
        && (request.scope === 'thread' ? thread.id === active : !thread.incognito && !thread.phoneLocalOnly && !thread.googleLocalOnly && !thread.bot && (request.includeArchived || !thread.archivedAt))
        && (!request.projectId || thread.projectId === request.projectId))
      const matches: ConversationSearchMatch[] = []
      const terms = searchTerms(query)
      if (terms.length) for (const thread of visible.slice(0, 1000)) {
        const candidates = [{ id: null as string | null, source: 'title' as const, text: thread.title }, ...thread.timeline.map(item => ({ id: item.id, source: 'timeline' as const, text: [item.title, item.text, item.output, item.command, item.meta].filter(Boolean).join('\n') }))]
        for (const item of candidates.slice(0, 10_000)) {
          if (matchingTerms(item.text, terms).length !== terms.length) continue
          const excerpt = searchExcerpt(item.text, terms)
          matches.push({ threadId: thread.id, title: thread.title, kind: thread.kind, projectId: thread.projectId ?? null, updatedAt: thread.updatedAt, itemId: item.id, source: item.source, ...excerpt, score: 1 })
        }
      }
      const limit = Math.max(1, Math.min(100, request.limit ?? 50))
      return { query, scope: request.scope, matches: matches.slice(0, limit), omitted: Math.max(0, matches.length - limit), truncated: visible.length > 1000 || visible.some(thread => thread.timeline.length > 9999) }
    },
  }
}
