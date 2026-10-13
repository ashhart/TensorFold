import { useState } from 'react'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import type { ConnectorEvent } from '../../lib/types'
import { selectPendingApprovals } from './pending-approvals'

// Consolidated approve-from-the-channel bar: every outbound action blocked on your
// decision, across all connectors, in one urgent-first strip. Approve/Reject resolve
// through the same connector approval path the per-thread draft buttons use; clicking
// the row jumps to that conversation. Renders nothing when the queue is empty.
export function PendingApprovalsBar({
  events,
  onResolved,
  onOpen,
}: {
  events: ConnectorEvent[]
  onResolved: () => void
  onOpen?: (threadKey: string) => void
}) {
  const queue = selectPendingApprovals(events)
  const [busy, setBusy] = useState<string | null>(null)
  if (queue.length === 0) return null

  const resolve = async (id: string, decision: 'approve' | 'deny') => {
    setBusy(id)
    try {
      await api.resolveConnectorApproval(id, decision)
      onResolved()
    } catch {
      /* best effort — refresh still reconciles state */
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="mx-8 mt-4 overflow-hidden rounded-xl border border-warn/30 bg-warn/[0.06] shadow-card fade-in">
      <div className="flex items-center gap-2 border-b border-warn/20 px-4 py-2">
        <Icon name="shield" size={14} className="text-warn" />
        <span className="text-[13px] font-medium text-ink">Needs your approval</span>
        <span className="rounded-full bg-warn/20 px-1.5 text-[12px] font-medium tabular-nums text-ink-2">{queue.length}</span>
        <span className="ml-auto text-[12px] text-ink-3">across every channel</span>
      </div>
      <div className="divide-y divide-line">
        {queue.map((item) => (
          <div key={item.id} className="flex items-center gap-3 px-4 py-2.5">
            <BrandIcon brand={connectorBrand(item.connectorId)} size={24} className="shrink-0 rounded-[6px]" />
            <button
              onClick={() => onOpen?.(item.threadKey)}
              className="min-w-0 flex-1 text-left"
              title="Open this conversation"
            >
              <span className="flex items-center gap-1.5">
                <span className="truncate text-[13px] font-medium text-ink">{item.title || item.actor}</span>
                {item.urgent && <span className="shrink-0 rounded bg-warn/20 px-1 text-[11px] font-semibold uppercase tracking-wide text-warn">urgent</span>}
              </span>
              <span className="mt-0.5 block truncate text-[13px] text-ink-3">{item.summary || item.actor}</span>
            </button>
            <button
              disabled={busy === item.id}
              onClick={() => resolve(item.id, 'approve')}
              className="shrink-0 rounded-md bg-positive/15 px-2.5 py-1 text-[13px] font-medium text-positive transition-colors hover:bg-positive/25 disabled:opacity-50"
            >
              Approve
            </button>
            <button
              disabled={busy === item.id}
              onClick={() => resolve(item.id, 'deny')}
              className="shrink-0 rounded-md border border-line px-2.5 py-1 text-[13px] text-ink-3 transition-colors hover:text-danger disabled:opacity-50"
            >
              Reject
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
