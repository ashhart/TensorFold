import { conversationSearchDateBound, normalizeConversationSearchFilters } from '../../../shared/conversation-search-filters'
import type { ConversationSearchFilters } from '../../../shared/conversation-search'

export interface SearchFilterDraft { from: string; to: string; providerId: string; model: string; attachments: boolean }
export const emptySearchFilterDraft = (): SearchFilterDraft => ({ from: '', to: '', providerId: '', model: '', attachments: false })
export function searchFiltersFromDraft(draft: SearchFilterDraft): ConversationSearchFilters | undefined {
  return normalizeConversationSearchFilters({
    ...(draft.from ? { updatedFrom: conversationSearchDateBound(draft.from) } : {}),
    ...(draft.to ? { updatedBefore: conversationSearchDateBound(draft.to, true) } : {}),
    ...(draft.providerId ? { providerId: draft.providerId } : {}),
    ...(draft.model ? { model: draft.model } : {}),
    ...(draft.attachments ? { hasAttachments: true } : {}),
  })
}
