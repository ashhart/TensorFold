import type { ConversationSearchFilters } from './conversation-search'

/** No display text is written to a derived index for these filters. */
export interface SearchFilterConversation {
  updatedAt?: number
  createdAt?: number
  providerId?: string | null
  model?: string | null
  timeline?: readonly { kind: string; attachments?: readonly unknown[] }[]
  apiMessages?: readonly { role: string; content: unknown }[]
}

export function normalizeConversationSearchFilters(input: unknown): ConversationSearchFilters | undefined {
  if (input === undefined) return undefined
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Invalid conversation search filters.')
  const value = input as Record<string, unknown>
  if (Object.keys(value).some(key => !['updatedFrom', 'updatedBefore', 'providerId', 'model', 'hasAttachments'].includes(key))) throw new Error('Invalid conversation search filters.')
  const filters: ConversationSearchFilters = {}
  for (const key of ['updatedFrom', 'updatedBefore'] as const) {
    const number = value[key]
    if (number === undefined) continue
    if (typeof number !== 'number' || !Number.isSafeInteger(number) || number < 0 || number > 8_640_000_000_000_000) throw new Error('Invalid conversation search date.')
    filters[key] = number
  }
  if (filters.updatedFrom !== undefined && filters.updatedBefore !== undefined && filters.updatedFrom >= filters.updatedBefore) throw new Error('The start date must not be after the end date.')
  for (const key of ['providerId', 'model'] as const) {
    const text = value[key]
    if (text === undefined) continue
    if (typeof text !== 'string' || !text || text.length > 512 || /[\u0000-\u001f\u007f]/.test(text)) throw new Error('Invalid conversation search model filter.')
    // Model and provider identities are exact; trimming can select a different identity.
    filters[key] = text
  }
  if (value.hasAttachments !== undefined) {
    if (typeof value.hasAttachments !== 'boolean') throw new Error('Invalid conversation search attachment filter.')
    if (value.hasAttachments) filters.hasAttachments = true
  }
  return Object.keys(filters).length ? filters : undefined
}

export function conversationHasAttachments(thread: SearchFilterConversation): boolean {
  return !!thread.timeline?.some(item => item.kind === 'user' && !!item.attachments?.length)
    || !!thread.apiMessages?.some(message => message.role === 'user' && Array.isArray(message.content)
      && message.content.some(part => part && typeof part === 'object' && (part.type === 'image_url' || part.type === 'input_image' || part.type === 'input_file' || part.type === 'file')))
}

export function conversationMatchesFilters(thread: SearchFilterConversation, filters?: ConversationSearchFilters): boolean {
  if (!filters) return true
  const updated = thread.updatedAt || thread.createdAt || 0
  return (filters.updatedFrom === undefined || updated >= filters.updatedFrom)
    && (filters.updatedBefore === undefined || updated < filters.updatedBefore)
    && (filters.providerId === undefined || thread.providerId === filters.providerId)
    && (filters.model === undefined || thread.model === filters.model)
    && (!filters.hasAttachments || conversationHasAttachments(thread))
}

export function conversationSearchFilterKey(filters?: ConversationSearchFilters): string {
  return JSON.stringify([filters?.updatedFrom ?? null, filters?.updatedBefore ?? null, filters?.providerId ?? null, filters?.model ?? null, !!filters?.hasAttachments])
}

/** Date inputs are local calendar dates; the inclusive end becomes the next local midnight. */
export function conversationSearchDateBound(day: string, end = false): number | undefined {
  if (!day) return undefined
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error('Choose a valid calendar date.')
  const [year, month, date] = day.split('-').map(Number)
  const value = new Date(0)
  value.setHours(0, 0, 0, 0); value.setFullYear(year, month - 1, date)
  if (value.getFullYear() !== year || value.getMonth() !== month - 1 || value.getDate() !== date || year < 1970) throw new Error('Choose a valid calendar date.')
  if (end) value.setDate(value.getDate() + 1)
  return value.getTime()
}
