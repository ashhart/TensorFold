import { createHash, randomUUID } from 'node:crypto'
import { MAX_QUOTE_TEXT, validateMessageQuotes } from '../../shared/message-quotes'
import type { MessageQuote, PrepareMessageQuote } from '../../shared/message-quotes'
import type { Thread, TimelineItem } from './types'
export function quoteSourceRevision(item: TimelineItem): string { return createHash('sha256').update(item.text ?? '').digest('hex') }
function source(thread: Thread, id: string): TimelineItem {
  if (thread.orchestratorHostId) throw new Error('Quote messages on the conversation’s owning host.')
  const item = thread.timeline.find(item => item.id === id)
  if (!item || !['user','assistant'].includes(item.kind) || !item.text?.trim()) throw new Error('This quoted message is no longer available.')
  return item
}
export function prepareMessageQuote(thread: Thread, input: PrepareMessageQuote): MessageQuote {
  if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).some(key => !['threadId', 'itemId', 'selection'].includes(key)) || input.threadId !== thread.id || typeof input.itemId !== 'string' || input.itemId.length > 160) throw new Error('Choose a message in this conversation.')
  const item = source(thread, input.itemId), original = item.text!
  const selected = input.selection === undefined ? original : input.selection
  if (typeof selected !== 'string' || !selected.trim() || selected.length > MAX_QUOTE_TEXT || !original.includes(selected)) throw new Error('Select a passage of at most 8,000 characters from this message.')
  return { id: randomUUID(), sourceItemId: item.id, sourceRevision: quoteSourceRevision(item), sourceText: selected, text: selected }
}
export function resolveMessageQuotes(thread: Thread, input: unknown): MessageQuote[] {
  const quotes = validateMessageQuotes(input)
  for (const quote of quotes) {
    const item = source(thread, quote.sourceItemId)
    if (quote.sourceRevision !== quoteSourceRevision(item) || !item.text!.includes(quote.sourceText)) throw new Error('A quoted source changed; refresh or remove the quote before sending.')
  }
  return quotes
}
