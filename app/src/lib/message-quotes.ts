import { validateMessageQuotes } from '../../shared/message-quotes'
import type { MessageQuote, PrepareMessageQuote } from '../../shared/message-quotes'
import { api } from './ipc'
import { useStore } from './store'
import type { Thread } from './types'

export const MESSAGE_QUOTE_EVENT = 'tensorfold:message-quote'
export interface MessageQuoteEvent { threadId: string; quote: MessageQuote }

function quoteOwner(thread: Thread): string {
  return JSON.stringify([thread.id, thread.conversationRevision ?? 0, thread.providerId ?? null, thread.providerAccountId ?? null,
    thread.model ?? null, thread.projectId ?? null, thread.cwd ?? '', thread.workspace ?? null, thread.engine ?? 'native',
    !!thread.incognito, !!thread.phoneLocalOnly, !!thread.googleLocalOnly, thread.bot ?? null])
}
async function sourceRevision(text: string): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('This passage could not be quoted.')
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')
}

/** A late preparation cannot insert an old source after a chat, owner or revision changes. */
export async function requestMessageQuote(input: PrepareMessageQuote): Promise<void> {
  const request = { threadId: input.threadId, itemId: input.itemId, ...(input.selection === undefined ? {} : { selection: input.selection }) }
  const initial = useStore.getState().threads.find(row => row.id === request.threadId)
  const source = initial?.timeline.find(row => row.id === request.itemId && (row.kind === 'user' || row.kind === 'assistant'))
  if (!initial || !source?.text?.trim()) return
  const originalText = source.text, originalKind = source.kind, owner = quoteOwner(initial)
  const current = () => {
    const state = useStore.getState(), view = state.view
    const thread = state.threads.find(row => row.id === request.threadId)
    const item = thread?.timeline.find(row => row.id === request.itemId)
    return view.name === 'thread' && view.id === request.threadId && thread && !thread.orchestratorHostId && !thread.archivedAt
      && quoteOwner(thread) === owner && item?.kind === originalKind && item.text === originalText
  }
  if (!current()) return
  try {
    const [prepared, revision] = await Promise.all([api.prepareMessageQuote(request), sourceRevision(originalText)])
    if (!current()) return
    const [quote] = validateMessageQuotes([prepared])
    if (quote.sourceItemId !== request.itemId || quote.sourceRevision !== revision || quote.sourceText !== (request.selection ?? originalText) || quote.text !== quote.sourceText) throw new Error('This passage could not be quoted.')
    window.dispatchEvent(new CustomEvent<MessageQuoteEvent>(MESSAGE_QUOTE_EVENT, { detail: { threadId: request.threadId, quote } }))
  } catch (error) {
    if (current()) useStore.getState().setNotice({ kind: 'error', message: error instanceof Error ? error.message : 'This passage could not be quoted.' })
  }
}
