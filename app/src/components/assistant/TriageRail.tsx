import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import { EmptyState } from './EmptyState'
import { PriorityPill } from './PriorityPill'
import { sourceName } from './mappings'
import type { AssistantItem } from './types'

// Triage rail: pick a source thread to open as a conversation.
export function TriageRail({ items, selectedItemId, onSelect }: {
  items: AssistantItem[]
  selectedItemId: string | null
  onSelect: (id: string) => void
}) {
  const tx = useAuthoredText()
  return (
    <div className="hidden w-[280px] shrink-0 flex-col border-r border-line md:flex">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <div className="text-[13px] font-medium text-ink-2">{tx("Threads")}</div>
        <span className="text-[12px] text-ink-4">{items.length}</span>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        {items.length === 0 ? (
          <EmptyState icon="check" title={tx("Clear for now")} detail="New email, messages, documents, and approvals land here." />
        ) : items.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className={clsx('flex w-full items-start gap-2.5 px-3 py-2.5 text-left transition-colors hover:bg-surface-hover', selectedItemId === item.id && 'bg-surface')}
          >
            <BrandIcon brand={connectorBrand(item.source)} size={26} className="mt-0.5" />
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-1.5">
                <span className="truncate text-[13px] font-medium text-ink-2">{item.from}</span>
                {item.count && item.count > 1 ? (
                  <span className="shrink-0 rounded-full bg-ink-5 px-1.5 text-[11px] font-medium tabular-nums text-ink-3">{item.count > 99 ? '99+' : item.count}</span>
                ) : null}
                <span className="ml-auto shrink-0 text-[12px] text-ink-4">{item.age}</span>
              </span>
              {/* Show the latest message preview; fall back to the title when it adds
                  info beyond the sender (e.g. an email subject differs from the from). */}
              <span className="mt-0.5 block truncate text-[13px] leading-snug text-ink">
                {item.title && item.title !== item.from ? item.title : (item.detail || item.title)}
              </span>
              <span className="mt-1 flex items-center gap-1.5">
                <PriorityPill priority={item.priority} />
                <span className="truncate text-[12px] text-ink-3">{sourceName(item.source)}</span>
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
