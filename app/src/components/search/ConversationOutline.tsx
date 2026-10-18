import { createPortal } from 'react-dom'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { TimelineItem } from '../../lib/types'
import type { ThreadAnnotations } from '../../lib/thread-annotations'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useAccessibleDialog } from '../navigation/useAccessibleDialog'
import { buildConversationOutline, outlinePage, type OutlineKind } from './conversation-outline'
import { revealConversationItem } from './search-events'

export function ConversationOutline({ threadId, timeline, annotations, activeAssistantId }: { threadId: string; timeline: readonly TimelineItem[]; annotations: ThreadAnnotations; activeAssistantId?: string }) {
  const tx = useAuthoredText()
  const [open, setOpen] = useState(false), [query, setQuery] = useState(''), [kind, setKind] = useState<'all' | OutlineKind>('all'), [page, setPage] = useState(0)
  const live = useRef({ threadId, timeline, open, mounted: true })
  live.current = { threadId, timeline, open, mounted: live.current.mounted }
  useEffect(() => { live.current.mounted = true; return () => { live.current.mounted = false } }, [])
  const panel = useRef<HTMLDivElement>(null), input = useRef<HTMLInputElement>(null)
  const close = () => setOpen(false)
  useAccessibleDialog({ open, onClose: close, dialogRef: panel, initialFocusRef: input })
  const outline = useMemo(() => open ? buildConversationOutline(timeline, annotations, { activeAssistantId }) : { entries: [], truncated: false }, [open, timeline, annotations, activeAssistantId])
  const visible = useMemo(() => outlinePage(outline.entries, { query, kind, page }), [outline, query, kind, page])
  useEffect(() => { setOpen(false); setQuery(''); setKind('all'); setPage(0) }, [threadId])
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || !event.shiftKey || event.key.toLowerCase() !== 'o') return
      if (document.querySelector('[aria-modal="true"]')) return
      event.preventDefault(); setOpen(value => !value)
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [])
  const labels = { message: tx('Your message'), answer: tx('Answer'), outcome: tx('Recorded outcome'), pin: tx('Pinned decision') }
  return <>
    <button type="button" aria-haspopup="dialog" aria-expanded={open} title={tx('Conversation outline (⌘⇧O / Ctrl+Shift+O)')} className="rounded-lg px-2 py-1 text-[12px] text-ink-2 hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent" onClick={() => setOpen(true)}>{tx('Outline')}</button>
    {open && createPortal(<div className="fixed inset-0 z-[130] flex items-start justify-center bg-canvas/85 p-3 sm:p-6" onMouseDown={event => { if (event.target === event.currentTarget) close() }}>
      <div ref={panel} role="dialog" aria-modal="true" aria-labelledby="conversation-outline-title" tabIndex={-1} className="flex max-h-full w-full max-w-[640px] flex-col rounded-xl border border-line bg-canvas shadow-pop outline-none">
        <header className="flex items-center justify-between gap-3 border-b border-line p-3"><h2 id="conversation-outline-title" className="text-[15px] font-medium text-ink">{tx('Conversation outline')}</h2><button type="button" className="rounded px-2 py-1 text-[12px] text-ink-2 focus-visible:outline-2 focus-visible:outline-accent" onClick={close}>{tx('Close outline')}</button></header>
        <div className="space-y-2 border-b border-line p-3"><input ref={input} aria-label={tx('Filter outline text')} maxLength={256} value={query} onChange={event => { setQuery(event.target.value); setPage(0) }} placeholder={tx('Find a message, outcome or decision…')} className="w-full rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink focus-visible:outline-2 focus-visible:outline-accent" /><label className="flex items-center gap-2 text-[12px] text-ink-2">{tx('Show')}<select aria-label={tx('Outline entry type')} value={kind} onChange={event => { setKind(event.target.value as 'all' | OutlineKind); setPage(0) }} className="min-w-0 rounded border border-line bg-card px-2 py-1"><option value="all">{tx('All entries')}</option>{(Object.keys(labels) as OutlineKind[]).map(key => <option key={key} value={key}>{labels[key]}</option>)}</select></label><p className="text-[11px] text-ink-3">{tx('Messages, final answers, file changes and failed work are shown from the active conversation. Pin a message to mark a decision; no summary is sent to a model.')}</p></div>
        <nav aria-label={tx('Conversation outline entries')} className="scroll-thin min-h-0 space-y-1 overflow-y-auto p-3" onKeyDown={event => {
          if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
          const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button[data-outline-item]')]
          const position = buttons.indexOf(document.activeElement as HTMLButtonElement)
          if (position < 0 || !buttons.length) return
          event.preventDefault()
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (position + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length
          buttons[next]?.focus()
        }}>
          {visible.entries.map(entry => <button key={entry.itemId} data-outline-item type="button" className="block w-full rounded-lg border border-line/60 bg-card px-3 py-2 text-left hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent" onClick={() => {
            if (!live.current.mounted || !live.current.open || live.current.threadId !== threadId || !live.current.timeline.some(item => item.id === entry.itemId)) return
            close(); revealConversationItem({ threadId, itemId: entry.itemId }, 'outline')
          }}><span className="block text-[10px] font-medium text-accent">{labels[entry.kind]}</span><span dir="auto" className="mt-1 block break-words text-[13px] text-ink-2">{entry.label || tx('Untitled message')}</span></button>)}
          {!visible.total && <p role="status" className="p-5 text-center text-[13px] text-ink-3">{tx('No outline entries match these filters.')}</p>}
        </nav>
        <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-line p-3 text-[12px] text-ink-2"><span role="status">{tx('Outline entries: {count}', { count: visible.total })}{visible.pages > 1 ? tx(' · Page {page} of {pages}', { page: visible.page + 1, pages: visible.pages }) : ''}</span><div className="flex gap-2"><button type="button" disabled={visible.page === 0} onClick={() => setPage(visible.page - 1)} className="rounded px-2 py-1 disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-accent">{tx('Previous entries')}</button><button type="button" disabled={visible.page + 1 >= visible.pages} onClick={() => setPage(visible.page + 1)} className="rounded px-2 py-1 disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-accent">{tx('Next entries')}</button></div>{outline.truncated && <p className="w-full text-warn">{tx('The outline reached its scan limit. Use Find in chat to locate other messages.')}</p>}</footer>
      </div>
    </div>, document.body)}
  </>
}
