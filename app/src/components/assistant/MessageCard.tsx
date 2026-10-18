import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { BrandMark } from '../BrandMark'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import type { BubbleRow, MessageDeliveryStatus } from './conversation'

// Discord-style message card: avatar + name + timestamp on a header line,
// content below, all LEFT-aligned (not left/right bubbles). TensorFold's replies get
// an "APP"/bot badge. Threaded replies collapse into a "N replies" preview you
// click to expand. This replaces the iMessage-style bubble renderer.

export interface MessageCardItem {
  id: string
  author: string
  text: string
  time?: number
  connectorId?: string
  isBot?: boolean
  isFromMe?: boolean
  status?: MessageDeliveryStatus
  replyToId?: string | null
  threadKey?: string | null
  sort: number
}

function avatarFor(item: MessageCardItem): { brand: string | null; initials: string } {
  if (item.connectorId) {
    const brand = connectorBrand(item.connectorId)
    if (brand && brand !== 'custom') return { brand, initials: '' }
  }
  const initials = item.author.split(/\s+/).slice(0, 2).map((w) => w[0]?.toUpperCase() ?? '').join('') || '?'
  return { brand: null, initials }
}

function timeLabel(ts?: number): string {
  if (!ts) return ''
  const d = new Date(ts)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  return sameDay ? authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Today at {value0}", { value0: String(time) }) : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`
}

interface DraftActions {
  onApprove?: (id: string) => void
  onEdit?: (id: string) => void
  onDiscard?: (id: string) => void
  onRedraft?: (id: string, how: string) => void
}

const REDRAFTS: { label: string; how: string }[] = [
  { get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Shorter") }, how: 'shorter and more to the point' },
  { get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Warmer") }, how: 'warmer and friendlier' },
  { get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Firmer") }, how: 'firmer and more direct' },
]

function MessageCard({ item, onApprove, onEdit, onDiscard, onRedraft }: { item: MessageCardItem } & DraftActions) {
  const tx = useAuthoredText()
  const { brand, initials } = avatarFor(item)
  const isTensorFoldStudio = item.isBot || ['TensorFold', 'Vontra'].includes(item.author)
  return (
    <div className="group flex items-start gap-3 px-2 py-2 hover:bg-surface">
      {/* Avatar: brand tile for source messages, initials for people, folded-lane glyph for TensorFold. */}
      <span className="mt-0.5 shrink-0">
        {isTensorFoldStudio ? (
          <BrandMark size={36} tile />
        ) : brand ? (
          <BrandIcon brand={brand as any} size={36} className="rounded-full" />
        ) : (
          <span className="flex h-9 w-9 items-center justify-center rounded-full bg-surface text-[13px] font-semibold text-ink-2">{initials}</span>
        )}
      </span>
      <div className="min-w-0 flex-1">
        {/* Header: name + APP badge + timestamp. */}
        <div className="flex items-center gap-1.5">
          <span className={clsx('text-[14px] font-semibold', isTensorFoldStudio ? 'text-accent' : 'text-ink')}>{isTensorFoldStudio ? "TensorFold" : item.author}</span>
          {isTensorFoldStudio && <span className="rounded bg-accent-soft px-1 py-0.5 text-[11px] font-bold uppercase tracking-wide text-accent">APP</span>}
          {item.isFromMe && !isTensorFoldStudio && <span className="rounded bg-surface px-1 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-ink-3">{tx("You")}</span>}
          <span className="text-[12px] text-ink-4">{timeLabel(item.time)}</span>
        </div>
        {/* Content. */}
        <div className="mt-0.5 whitespace-pre-wrap break-words text-[14px] leading-relaxed text-ink-2">{item.text}</div>
        {item.status && item.status !== 'draft' && <div role="status" className="mt-1 text-[12px] text-ink-2">{item.status === 'sent' ? tx("Delivered") : item.status === 'approved' ? tx("Approved, no delivery receipt") : item.status === 'sending' ? tx("Sending…") : item.status === 'uncertain' ? tx("Delivery uncertain; check the destination before sending again") : tx("Delivery failed")}</div>}
        {/* Draft approval affordance (TensorFold drafts awaiting the user). */}
        {isTensorFoldStudio && item.status === 'draft' && (
          <div className="mt-1.5 flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5">
              <button onClick={() => onApprove?.(item.id)} className="rounded-lg bg-accent px-2.5 py-1 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong">{tx("Review message")}</button>
              <button onClick={() => onEdit?.(item.id)} className="rounded-lg border border-line px-2.5 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:bg-surface-hover">{tx("Edit")}</button>
              <button onClick={() => onDiscard?.(item.id)} className="rounded-lg px-2 py-1 text-[13px] text-ink-3 transition-colors hover:text-danger">{tx("Discard")}</button>
            </div>
            {/* One-click redraft variants: ask the assistant to re-write this draft. */}
            {onRedraft && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[12px] text-ink-4">{tx("Redraft:")}</span>
                {REDRAFTS.map((r) => (
                  <button key={r.label} onClick={() => onRedraft(item.id, r.how)} className="rounded-md border border-line px-1.5 py-0.5 text-[12px] text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink">{r.label}</button>
                ))}
              </div>
            )}
          </div>
        )}
        {item.status === 'sent' && <div className="mt-1 flex items-center gap-1 text-[12px] text-ink-4"><Icon name="check" size={11} /> {tx("Sent")}</div>}
      </div>
    </div>
  )
}

// Group flat message list into threads: a root message + its replies (by replyToId
// or threadKey). Roots render fully; reply chains collapse into "N replies" previews.
function groupThreads(items: MessageCardItem[]): { root: MessageCardItem; replies: MessageCardItem[] }[] {
  const byId = new Map(items.map((i) => [i.id, i]))
  const roots: MessageCardItem[] = []
  const repliesOf = new Map<string, MessageCardItem[]>()
  for (const item of items) {
    const parentId = item.replyToId ?? null
    const parent = parentId ? byId.get(parentId) ?? null : null
    if (parent && parentId) {
      const arr = repliesOf.get(parentId) ?? []
      arr.push(item); repliesOf.set(parentId, arr)
    } else if (item.threadKey && items.some((other) => other.replyToId === item.id)) {
      roots.push(item)
    } else {
      roots.push(item)
    }
  }
  return roots.map((root) => ({ root, replies: (repliesOf.get(root.id) ?? []).sort((a, b) => a.sort - b.sort) }))
}

export function MessageCardView({ items, emptyTitle, emptyDetail, onApprove, onEdit, onDiscard, onRedraft }: {
  items: MessageCardItem[]
  emptyTitle: string
  emptyDetail: string
} & DraftActions) {
  const tx = useAuthoredText()
  const endRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }
  useEffect(() => {
    if (stick.current) endRef.current?.scrollIntoView({ block: 'end' })
  }, [items.length])

  if (items.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-surface text-ink-3"><Icon name="message" size={20} /></span>
        <div className="mt-3 text-[14px] font-medium text-ink-2">{emptyTitle}</div>
        <div className="mt-1 max-w-[340px] text-[13px] leading-relaxed text-ink-3">{emptyDetail}</div>
      </div>
    )
  }

  const threads = groupThreads(items)
  return (
    <div ref={scrollRef} onScroll={onScroll} className="scroll-thin min-h-0 flex-1 overflow-y-auto py-3 select-text">
      <div className="mx-auto flex w-full max-w-[760px] flex-col">
        {threads.map(({ root, replies }) => (
          <div key={root.id}>
            <MessageCard item={root} onApprove={onApprove} onEdit={onEdit} onDiscard={onDiscard} onRedraft={onRedraft} />
            {replies.length > 0 && (
              replies.length <= 2 || expanded.has(root.id) ? (
                // Show all replies (few, or user expanded).
                <div className="ml-12 border-l-2 border-line pl-2">
                  {replies.map((r) => <MessageCard key={r.id} item={r} onApprove={onApprove} onEdit={onEdit} onDiscard={onDiscard} onRedraft={onRedraft} />)}
                </div>
              ) : (
                // Collapsed preview: "N replies — click to expand".
                <div className="ml-12 py-1">
                  <button
                    onClick={() => setExpanded((s) => new Set(s).add(root.id))}
                    className="rounded-lg px-2 py-1 text-[13px] font-medium text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink"
                  >
                    {replies.length} {tx("replies — click to expand")}</button>
                </div>
              )
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}

// Convert the shared BubbleRow (eventToBubble output) into the card-item shape.
export function bubbleToCard(row: BubbleRow): MessageCardItem {
  return {
    id: row.id,
    author: row.author,
    text: row.text,
    time: row.time,
    connectorId: row.source,
    isBot: row.direction === 'in' && ['TensorFold', 'Vontra'].includes(row.author),
    isFromMe: row.direction === 'out',
    status: row.status,
    sort: row.sort,
  }
}
