import { useRef, useState } from 'react'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import { Icon } from '../../lib/icons'
import { EmptyState } from './EmptyState'
import { sourceName } from './mappings'
import type { AssistantItem, SourceKey, SourceRow } from './types'

export function TriageRail({ items, selectedItemId, onSelect, selectedSource = 'all', sources = [], onSourceChange, pendingThreadKeys = [] }: {
  items: AssistantItem[]
  selectedItemId: string | null
  onSelect: (id: string) => void
  selectedSource?: SourceKey
  sources?: SourceRow[]
  onSourceChange?: (source: SourceKey) => void
  pendingThreadKeys?: string[]
}) {
  const tx = useAuthoredText()
  const [query, setQuery] = useState('')
  const [needsReview, setNeedsReview] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)
  const pending = new Set(pendingThreadKeys)
  const term = query.trim().toLocaleLowerCase()
  const visible = items.filter((item) => (!needsReview || pending.has(item.threadKey ?? item.id)) && (!term || [item.from, item.title, item.detail, sourceName(item.source)].some((value) => value.toLocaleLowerCase().includes(term))))
  const selectedVisible = visible.some((item) => item.id === selectedItemId)
  return (
    <aside aria-label={tx('Assistant conversations')} className="flex w-full shrink-0 flex-col border-b border-line bg-card-2 md:w-[248px] md:border-b-0 md:border-r xl:w-[288px]">
      <div className="grid grid-cols-2 items-center gap-2 px-3 py-2 md:block md:space-y-2.5 md:py-3">
        <div className="hidden items-center justify-between gap-2 md:flex">
          <h3 className="text-[13px] font-semibold text-ink">{tx('Conversations')}</h3>
          <span className="rounded-full bg-surface px-2 text-[12px] tabular-nums text-ink-3">{items.length}</span>
        </div>
        {onSourceChange && <select aria-label={tx('Filter inbox by source')} value={selectedSource} onChange={(event) => { onSourceChange(event.target.value as SourceKey); setQuery('') }} className="w-full rounded-lg border border-line bg-card px-2.5 py-1.5 text-[12px] text-ink outline-none focus:border-accent">
          <option value="all">{tx('All sources')}</option>
          {sources.map((source) => <option key={source.id} value={source.id}>{source.name}</option>)}
        </select>}
        <label className="hidden items-center gap-2 rounded-lg border border-line bg-card px-2.5 focus-within:border-accent md:flex">
          <Icon name="search" size={13} className="shrink-0 text-ink-3" />
          <input aria-label={tx('Search inbox conversations')} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tx('Search conversations')} className="h-8 min-w-0 flex-1 bg-transparent text-[12px] text-ink outline-none placeholder:text-ink-3" />
          {query && <button type="button" aria-label={tx('Clear conversation search')} onClick={() => setQuery('')} className="rounded text-ink-3 focus-visible:outline-2 focus-visible:outline-accent"><Icon name="x" size={12} /></button>}
        </label>
        <div className="hidden gap-1 rounded-lg bg-surface p-0.5 md:flex" aria-label={tx('Conversation filter')}>
          <button type="button" aria-pressed={!needsReview} onClick={() => setNeedsReview(false)} className={clsx('flex-1 rounded-md px-2 py-1 text-[12px] focus-visible:outline-2 focus-visible:outline-accent', !needsReview ? 'bg-card font-medium text-ink shadow-sm' : 'text-ink-3')}>{tx('All')}</button>
          <button type="button" aria-pressed={needsReview} onClick={() => setNeedsReview(true)} className={clsx('flex-1 rounded-md px-2 py-1 text-[12px] focus-visible:outline-2 focus-visible:outline-accent', needsReview ? 'bg-card font-medium text-ink shadow-sm' : 'text-ink-3')}>{tx('Needs review')}</button>
        </div>
        <select aria-label={tx('Choose a conversation')} value={selectedItemId ?? ''} onChange={(event) => onSelect(event.target.value)} className="w-full rounded-lg border border-line bg-card px-2 py-2 text-[13px] text-ink md:hidden">
          {!items.length && <option value="">{tx('No conversations')}</option>}
          {items.map((item) => <option key={item.id} value={item.id}>{item.from} · {item.title}</option>)}
        </select>
      </div>
      <div ref={listRef} className="scroll-thin hidden min-h-0 flex-1 overflow-y-auto border-t border-line p-1.5 md:block">
        {visible.length === 0 ? <EmptyState icon={term ? 'search' : 'check'} title={term ? tx('No matching conversations') : needsReview ? tx('Nothing awaiting review') : tx('No conversations yet')} detail={term ? tx('Try a sender, subject, or message preview.') : needsReview ? tx('Pending approvals appear here when an action needs your decision.') : tx('Synced messages and documents will appear here.')} /> : visible.map((item, index) => (
          <button key={item.id} type="button" data-testid="assistant-conversation" data-thread-index={index} aria-current={selectedItemId === item.id ? 'true' : undefined} tabIndex={selectedItemId === item.id || (!selectedVisible && index === 0) ? 0 : -1}
            onClick={() => onSelect(item.id)}
            onKeyDown={(event) => {
              const next = event.key === 'ArrowDown' ? Math.min(index + 1, visible.length - 1) : event.key === 'ArrowUp' ? Math.max(0, index - 1) : event.key === 'Home' ? 0 : event.key === 'End' ? visible.length - 1 : null
              if (next === null) return
              event.preventDefault(); onSelect(visible[next].id)
              listRef.current?.querySelector<HTMLButtonElement>(`[data-thread-index="${next}"]`)?.focus()
            }}
            className={clsx('mb-1 flex w-full items-start gap-2.5 rounded-xl border px-2.5 py-3 text-left transition-colors focus-visible:outline-2 focus-visible:outline-accent', selectedItemId === item.id ? 'border-accent/20 bg-accent-soft' : 'border-transparent hover:bg-surface-hover')}>
            <BrandIcon brand={connectorBrand(item.source)} size={25} className="mt-0.5 shrink-0 rounded-lg" />
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-1.5"><span className="truncate text-[12px] font-medium text-ink-2">{item.from}</span><span className="ml-auto shrink-0 text-[11px] text-ink-3">{item.age}</span></span>
              <span className="mt-1 block truncate text-[13px] font-medium leading-snug text-ink">{item.title || item.detail}</span>
              <span className="mt-1 block truncate text-[12px] leading-relaxed text-ink-3">{item.detail}</span>
              <span className="mt-2 flex items-center gap-1.5 text-[11px] text-ink-3">
                {pending.has(item.threadKey ?? item.id) ? <span className="rounded bg-warn/15 px-1.5 py-0.5 font-medium text-warn">{tx('Needs review')}</span> : item.priority === 'urgent' || item.priority === 'soon' ? <span className="rounded bg-surface px-1.5 py-0.5">{tx('Priority')}</span> : null}
                <span>{sourceName(item.source)}</span>{(item.count ?? 0) > 1 && <span className="ml-auto tabular-nums">{tx('Messages: {count}', { count: item.count })}</span>}
              </span>
            </span>
          </button>
        ))}
      </div>
    </aside>
  )
}
