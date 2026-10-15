import { useEffect, useRef } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import type { BubbleRow } from './conversation'

// A single message bubble in the unified Assistant conversation surface.
// Outgoing (your reply / instruction) = iMessage blue, right-aligned.
// Incoming (source message / TensorFold's draft) = neutral gray, left-aligned.
export function Bubble({ row, onApprove, onEdit, onDiscard }: {
  row: BubbleRow
  onApprove?: (id: string) => void
  onEdit?: (id: string) => void
  onDiscard?: (id: string) => void
}) {
  const out = row.direction === 'out'
  return (
    <div className={clsx('flex w-full fade-in', out ? 'justify-end' : 'justify-start')}>
      <div className={clsx('flex max-w-[78%] flex-col', out ? 'items-end' : 'items-start')}>
        {/* Author + timestamp header */}
        <div className={clsx('mb-1 flex items-center gap-1.5 px-1 text-[12px] text-ink-4', out ? 'flex-row-reverse' : 'flex-row')}>
          <span className="font-medium text-ink-3">{row.author}</span>
          {row.time && <span>{timeLabel(row.time)}</span>}
          {row.source && <span className="text-ink-5">· {row.source}</span>}
        </div>
        {/* The bubble body */}
        <div
          className={clsx(
            'whitespace-pre-wrap px-3.5 py-2.5 text-[14px] leading-relaxed shadow-card',
            out
              ? 'rounded-3xl rounded-br-md bg-bubble-out text-bubble-out-ink'
              : 'rounded-3xl rounded-bl-md border border-line bg-bubble-in text-bubble-in-ink',
          )}
        >
          {row.text}
        </div>
        {/* Status + approval affordances for an agent draft awaiting the user. */}
        {out && row.status === 'draft' && (
          <div className="mt-1.5 flex items-center gap-1.5 pr-1">
            <button onClick={() => onApprove?.(row.id)} className="rounded-lg bg-accent px-2.5 py-1 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong">Approve</button>
            <button onClick={() => onEdit?.(row.id)} className="rounded-lg border border-line px-2.5 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:bg-surface-hover">Edit</button>
            <button onClick={() => onDiscard?.(row.id)} className="rounded-lg px-2 py-1 text-[13px] text-ink-3 transition-colors hover:text-danger">Discard</button>
          </div>
        )}
        {out && row.status === 'sent' && (
          <div className="mt-1 flex items-center gap-1 pr-1 text-[12px] text-ink-4"><Icon name="check" size={11} /> Sent</div>
        )}
        {out && row.status === 'failed' && (
          <div className="mt-1 pr-1 text-[12px] text-danger">Failed to send</div>
        )}
      </div>
    </div>
  )
}

function timeLabel(ts: number) {
  const d = new Date(ts)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  return sameDay ? time : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`
}

// The scrollable conversation column. Auto-sticks to the bottom as new bubbles
// arrive unless the user has scrolled up to read history.
export function ConversationView({ rows, emptyTitle, emptyDetail, onApprove, onEdit, onDiscard }: {
  rows: BubbleRow[]
  emptyTitle: string
  emptyDetail: string
  onApprove?: (id: string) => void
  onEdit?: (id: string) => void
  onDiscard?: (id: string) => void
}) {
  const endRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Track whether the user is near the bottom; only auto-scroll when they are.
  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  useEffect(() => {
    if (stick.current) endRef.current?.scrollIntoView({ block: 'end' })
  }, [rows.length])

  if (rows.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-surface text-ink-3"><Icon name="message" size={20} /></span>
        <div className="mt-3 text-[14px] font-medium text-ink-2">{emptyTitle}</div>
        <div className="mt-1 max-w-[340px] text-[13px] leading-relaxed text-ink-3">{emptyDetail}</div>
      </div>
    )
  }

  return (
    <div ref={scrollRef} onScroll={onScroll} className="scroll-thin min-h-0 flex-1 overflow-y-auto px-6 py-6 select-text">
      <div className="mx-auto flex w-full max-w-[720px] flex-col gap-4">
        {rows.map((row) => (
          <Bubble key={row.id} row={row} onApprove={onApprove} onEdit={onEdit} onDiscard={onDiscard} />
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}
