import { useEffect, useRef, useState, type RefObject } from 'react'
import type { MessageQuote } from '../../../shared/message-quotes'
import { MAX_QUOTE_TEXT } from '../../../shared/message-quotes'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { revealConversationItem } from '../search/search-events'

/** Selection must begin and end in this content node, never in another message or action. */
export function ownedMessageSelection(content: HTMLElement | null): string | undefined {
  const selection = typeof window.getSelection === 'function' ? window.getSelection() : null
  if (!content || !selection || selection.isCollapsed || selection.rangeCount !== 1 || !selection.anchorNode || !selection.focusNode
    || !content.contains(selection.anchorNode) || !content.contains(selection.focusNode)) return undefined
  const range = selection.getRangeAt(0)
  if (!content.contains(range.commonAncestorContainer)) return undefined
  const text = selection.toString()
  return text.trim() ? text.slice(0, MAX_QUOTE_TEXT + 1) : undefined
}

export function MessageQuoteAction({ contentRef, onQuote, label }: { contentRef: RefObject<HTMLElement>; onQuote(selection?: string): void; label: string }) {
  const tx = useAuthoredText()
  const captured = useRef<{ value: string | undefined } | null>(null)
  return <button type="button" aria-label={label} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[12px] text-ink-2 hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent"
    onMouseDown={event => { if (event.button === 0) captured.current = { value: ownedMessageSelection(contentRef.current) } }}
    onClick={() => { const value = captured.current ? captured.current.value : ownedMessageSelection(contentRef.current); captured.current = null; onQuote(value) }}>{tx('Quote')}</button>
}

const editingOwners = new WeakMap<(editing: boolean) => void, symbol>()

export interface DraftMessageQuotesProps {
  quotes: readonly MessageQuote[]
  threadId: string
  onChange?(id: string, text: string): void
  onRemove?(id: string): void
  disabled?: boolean
  onEditingChange?(editing: boolean): void
}

export function DraftMessageQuotes({ quotes, threadId, onChange, onRemove, disabled = false, onEditingChange }: DraftMessageQuotesProps) {
  const tx = useAuthoredText()
  const [edit, setEdit] = useState<{ id: string; threadId: string; text: string; expected: string } | null>(null)
  const live = useRef({ quotes, threadId, disabled, onChange, onRemove, onEditingChange })
  live.current = { quotes, threadId, disabled, onChange, onRemove, onEditingChange }
  const editOwner = useRef(Symbol('quote-editor'))
  const reportEditing = (editing: boolean, callback = live.current.onEditingChange) => {
    if (!callback) return
    if (editing) { editingOwners.set(callback, editOwner.current); callback(true) }
    else if (editingOwners.get(callback) === editOwner.current) { editingOwners.delete(callback); callback(false) }
  }
  const ownEdit = edit?.threadId === threadId && quotes.some(quote => quote.id === edit.id && quote.text === edit.expected) ? edit : null
  useEffect(() => { reportEditing(!!ownEdit); return () => reportEditing(false, onEditingChange) }, [!!ownEdit, threadId, onEditingChange])
  const buttonClass = 'rounded px-2 py-1 text-[11px] text-ink-2 hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40'
  if (!quotes.length) return null
  return <section aria-label={tx('Quoted passages')} className="scroll-thin max-h-64 space-y-2 overflow-y-auto rounded-lg border border-line bg-card p-2">
    <p className="text-[11px] text-ink-3">{tx('Quotes are context for your next message; editing a quote does not change its source.')}</p>
    {quotes.map(quote => <div key={quote.id} className="min-w-0 rounded border border-line/70 bg-canvas p-2">
      <div className="flex flex-wrap items-center justify-between gap-2"><span className="text-[11px] font-medium text-ink-2">{tx('Quoted passage')}{quote.text !== quote.sourceText && <span className="ml-2 text-warn">{tx('Edited quotation')}</span>}</span><button type="button" disabled={disabled} className={buttonClass} onClick={() => {
        if (live.current.threadId === threadId && !live.current.disabled && live.current.quotes.some(item => item.id === quote.id && item.sourceItemId === quote.sourceItemId)) revealConversationItem({ threadId, itemId: quote.sourceItemId }, 'quote')
      }}>{tx('Open source message')}</button></div>
      {ownEdit?.id === quote.id ? <div className="mt-2 space-y-2"><textarea aria-label={tx('Quoted text')} autoFocus disabled={disabled} maxLength={MAX_QUOTE_TEXT} value={ownEdit.text} onChange={event => { if (!live.current.disabled && live.current.threadId === threadId) setEdit({ ...ownEdit, text: event.target.value }) }} className="min-h-20 w-full resize-y rounded border border-line bg-card p-2 text-[12px] leading-relaxed text-ink focus-visible:outline-2 focus-visible:outline-accent" />
        {(!ownEdit.text.trim() || ownEdit.text.length > MAX_QUOTE_TEXT) && <p role="alert" className="text-[11px] text-danger">{tx('The quoted text must contain 1 to 8,000 characters.')}</p>}
        <div className="flex flex-wrap justify-end gap-2"><button type="button" disabled={disabled} className={buttonClass} onClick={() => { if (!live.current.disabled && live.current.threadId === threadId) { reportEditing(false); setEdit(null) } }}>{tx('Cancel quotation edit')}</button><button type="button" disabled={disabled || !ownEdit.text.trim() || ownEdit.text.length > MAX_QUOTE_TEXT || !onChange} className={buttonClass} onClick={() => {
          const current = live.current
          if (current.disabled || current.threadId !== threadId || !ownEdit.text.trim() || ownEdit.text.length > MAX_QUOTE_TEXT || !current.quotes.some(item => item.id === quote.id && item.text === ownEdit.expected)) return
          current.onChange?.(quote.id, ownEdit.text); reportEditing(false); setEdit(null)
        }}>{tx('Save quotation')}</button></div></div> : <>
        <blockquote dir="auto" className="mt-2 max-h-28 overflow-y-auto whitespace-pre-wrap break-words border-l-2 border-accent/40 pl-2 text-[12px] leading-relaxed text-ink-2">{quote.text}</blockquote>
        <div className="mt-1 flex flex-wrap justify-end gap-1">{onChange && <button type="button" disabled={disabled} className={buttonClass} onClick={() => {
          if (!live.current.disabled && live.current.threadId === threadId && live.current.quotes.some(item => item.id === quote.id && item.text === quote.text)) { reportEditing(true); setEdit({ id: quote.id, threadId, text: quote.text, expected: quote.text }) }
        }}>{tx('Edit quoted passage')}</button>}{onRemove && <button type="button" disabled={disabled} className={buttonClass} onClick={() => { if (!live.current.disabled && live.current.threadId === threadId && live.current.quotes.some(item => item.id === quote.id)) { live.current.onRemove?.(quote.id); if (edit?.id === quote.id) { reportEditing(false); setEdit(null) } } }}>{tx('Remove quoted passage')}</button>}</div>
      </>}
    </div>)}
  </section>
}

export function SentMessageQuotes({ quotes, threadId }: { quotes: readonly MessageQuote[]; threadId: string }) {
  const tx = useAuthoredText()
  if (!quotes.length) return null
  return <section aria-label={tx('Quoted passages')} className="w-full max-w-[42rem] space-y-2">{quotes.map(quote => <div key={quote.id} className="rounded-lg border border-line bg-card p-2.5">
    <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-ink-3"><span>{tx('Quoted passage')}{quote.text !== quote.sourceText && <span className="ml-2 text-warn">{tx('Edited quotation')}</span>}</span><button type="button" className="rounded px-2 py-1 text-accent focus-visible:outline-2 focus-visible:outline-accent" onClick={() => revealConversationItem({ threadId, itemId: quote.sourceItemId }, 'quote')}>{tx('Open source message')}</button></div>
    <blockquote dir="auto" className="mt-1 max-h-40 overflow-y-auto whitespace-pre-wrap break-words border-l-2 border-accent/40 pl-2 text-[12px] leading-relaxed text-ink-2">{quote.text}</blockquote>
  </div>)}</section>
}
