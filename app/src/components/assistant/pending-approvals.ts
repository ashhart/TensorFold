// Cross-channel "needs your approval" queue. The Assistant surfaces per-thread drafts,
// but a blocked outbound action shouldn't require hunting the right thread to clear it.
// selectPendingApprovals folds every connector event awaiting a decision into one
// urgent-first list so the consolidated bar can show Approve/Reject inline.
import type { ConnectorEvent } from '../../lib/types'

export interface PendingApproval {
  id: string
  connectorId: ConnectorEvent['connectorId']
  title: string
  summary: string
  actor: string
  threadKey: string
  createdAt: number
  urgent: boolean
}

// An event needs me when it explicitly asks for approval and hasn't been resolved.
// approvalStatus is optional/nullable on older events — treat unset as still-pending
// rather than silently dropping it from the queue.
export function isAwaitingApproval(event: ConnectorEvent): boolean {
  if (!event.requiresApproval) return false
  const status = event.approvalStatus
  return status == null || status === 'pending'
}

const URGENT_PRIORITIES = new Set(['critical', 'high'])

export function selectPendingApprovals(events: ConnectorEvent[]): PendingApproval[] {
  return events
    .filter(isAwaitingApproval)
    .map((event) => ({
      id: event.id,
      connectorId: event.connectorId,
      title: event.title,
      summary: event.summary,
      actor: event.actor,
      threadKey: event.threadKey,
      createdAt: event.createdAt,
      urgent: URGENT_PRIORITIES.has(event.priority),
    }))
    // Urgent first, then newest first — the most pressing decision sits at the top.
    .sort((a, b) => (Number(b.urgent) - Number(a.urgent)) || (b.createdAt - a.createdAt))
}
