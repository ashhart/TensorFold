import { validateMessageQuotes } from '../../shared/message-quotes'
import type { MessageQuote } from '../../shared/message-quotes'
import type { TimelineItem } from '../agent/types'

export function readableQuotations(item: TimelineItem): MessageQuote[] {
  const quotes = validateMessageQuotes(item.quotedMessages)
  if (quotes.length && item.kind !== 'user') throw new Error('Only user messages can contain saved quotations.')
  return quotes
}

/** Export only the visible quoted text; an edited quotation never reveals its hidden original. */
export function readableMessageText(item: TimelineItem): string {
  const message = item.text || '[Message has no text. See omitted attachments.]'
  const quotes = readableQuotations(item)
  if (!quotes.length) return message
  const blocks = quotes.map(quote => `Quoted passage · source message: ${quote.sourceItemId} · ${quote.text === quote.sourceText ? 'Original quotation' : 'User-edited quotation'}\n${quote.text}`)
  return `${blocks.join('\n\n')}\n\nMessage:\n${message}`
}
