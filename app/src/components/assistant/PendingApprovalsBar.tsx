import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { resolveInboxApproval } from './gmail-inbox-client'
import { Icon } from '../../lib/icons'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import type { ConnectorEvent } from '../../lib/types'
import { selectPendingApprovals } from './pending-approvals'
import { OutboundApprovalDialog } from './OutboundApprovalDialog'

// Keep pending decisions visible without allowing a large queue to displace the conversation.
export function PendingApprovalsBar({
  events,
  onResolved,
  onOpen,
}: {
  events: ConnectorEvent[]
  onResolved: () => void
  onOpen?: (threadKey: string) => void
}) {
  const tx = useAuthoredText()
  const queue = selectPendingApprovals(events)
  const [busy, setBusy] = useState<string | null>(null)
  const [reviewId, setReviewId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(false)
  if (queue.length === 0) return null

  const resolve = async (id: string, decision: 'approve' | 'deny', reviewDigest?: string) => {
    if (busy) return
    setBusy(id)
    setError('')
    try {
      const event = events.find(item => item.id === id)
      if (!event) throw new Error('Refresh this draft before reviewing it.')
      await resolveInboxApproval(event, decision, reviewDigest)
      setReviewId(null)
      onResolved()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : tx("This decision could not be saved."))
      onResolved()
    } finally {
      setBusy(null)
    }
  }

  return (
    <section aria-label={tx('Pending approvals')} className="shrink-0 overflow-hidden border-b border-warn/25 bg-warn/[0.06]">
      {reviewId && events.find((event) => event.id === reviewId) && <OutboundApprovalDialog event={events.find((event) => event.id === reviewId)!} busy={busy === reviewId} error={error} onConfirm={(digest) => void resolve(reviewId, 'approve', digest)} onClose={() => setReviewId(null)} />}
      {error && !reviewId && <p role="alert" className="px-4 py-3 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
      <div className="flex flex-wrap items-center gap-2 px-4 py-2 sm:px-6">
        <Icon name="shield" size={14} className="text-warn" />
        <span className="text-[13px] font-semibold text-ink">{tx('Awaiting your review')}</span>
        <span className="rounded-full bg-warn/20 px-1.5 text-[12px] font-medium tabular-nums text-ink-2">{queue.length}</span>
        <span className="ml-auto text-[12px] text-ink-3">{tx('Actions wait for your decision')}</span>
      </div>
      <div className="scroll-thin max-h-[220px] divide-y divide-warn/10 overflow-y-auto">
        {(expanded ? queue : queue.slice(0, 2)).map((item) => (
          <div key={item.id} className="flex flex-wrap items-center gap-2 px-4 py-2 sm:px-6">
            <BrandIcon brand={connectorBrand(item.connectorId)} size={24} className="shrink-0 rounded-[6px]" />
            <button
              type="button"
              onClick={() => onOpen?.(item.threadKey)}
              className="min-w-0 flex-1 rounded text-left focus-visible:outline-2 focus-visible:outline-accent"
              title={tx("Open this conversation")}
            >
              <span className="flex items-center gap-1.5">
                <span className="truncate text-[13px] font-medium text-ink">{item.title || item.actor}</span>
                {item.urgent && <span className="shrink-0 rounded bg-warn/20 px-1 text-[11px] font-semibold uppercase tracking-wide text-warn">{tx("urgent")}</span>}
              </span>
              <span className="mt-0.5 block truncate text-[13px] text-ink-3">{item.summary || item.actor}</span>
            </button>
            <button
              type="button"
              disabled={!!busy}
              onClick={() => { const event = events.find((candidate) => candidate.id === item.id); if (event?.outbound || event?.outboundError || /send_requested|reply_requested|automation_result_ready/.test(event?.action ?? '')) { setError(''); setReviewId(item.id) } else void resolve(item.id, 'approve') }}
              className="shrink-0 rounded-lg bg-accent px-3 py-1.5 text-[12px] font-medium text-accent-ink transition-colors hover:bg-accent-strong focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50"
            >
              {events.some((event) => event.id === item.id && (event.outbound || event.outboundError || /send_requested|reply_requested|automation_result_ready/.test(event.action))) ? tx('Review message') : tx('Approve action')}</button>
            <button
              type="button"
              disabled={!!busy}
              onClick={() => resolve(item.id, 'deny')}
              className="shrink-0 rounded-lg px-2.5 py-1.5 text-[12px] text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50"
            >
              {tx("Reject")}</button>
          </div>
        ))}
      </div>
      {queue.length > 2 && <button type="button" onClick={() => setExpanded(!expanded)} aria-expanded={expanded} className="w-full border-t border-warn/15 py-2 text-[12px] font-medium text-ink-2 hover:bg-warn/10 focus-visible:outline-2 focus-visible:outline-accent">{expanded ? tx('Show fewer approvals') : tx('Show all approvals ({count})', { count: queue.length })}</button>}
    </section>
  )
}
