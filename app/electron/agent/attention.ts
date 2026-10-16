// Unified "your turn" attention model. One place that decides what currently needs
// the USER (not the agent): pending tool approvals and connector drafts awaiting a
// send. Drives the needs-me count/badge and the OS-notification ping. Pure + testable;
// the host supplies the live arrays and the native-notification sink.

export type AttentionKind = 'approval' | 'connector_draft'

export interface AttentionItem {
  id: string
  kind: AttentionKind
  title: string
  detail: string
  threadId?: string | null
}

interface JobLike {
  id: string
  pendingApprovals?: { callId: string; tool?: string; threadId?: string | null }[]
}
interface ConnectorEventLike {
  id: string
  requiresApproval?: boolean
  approvalStatus?: string | null
  title?: string
  actor?: string
  pluginName?: string
  threadKey?: string | null
}

/**
 * Everything that needs the user right now: each pending tool approval (the agent is
 * blocked waiting on a yes/no) and each connector draft still awaiting a send.
 */
export function computeNeedsMe(state: { jobs?: JobLike[]; connectorEvents?: ConnectorEventLike[] }): AttentionItem[] {
  const items: AttentionItem[] = []
  for (const job of state.jobs ?? []) {
    for (const a of job.pendingApprovals ?? []) {
      items.push({
        id: `approval:${a.callId}`,
        kind: 'approval',
        title: 'Approval needed',
        detail: a.tool ? `Run ${a.tool}?` : 'A tool action needs your approval.',
        threadId: a.threadId ?? null,
      })
    }
  }
  for (const e of state.connectorEvents ?? []) {
    // A draft the agent prepared but hasn't sent — approvalStatus not yet resolved.
    if (e.requiresApproval && e.approvalStatus !== 'approved' && e.approvalStatus !== 'denied') {
      items.push({
        id: `draft:${e.id}`,
        kind: 'connector_draft',
        title: 'Reply waiting to send',
        detail: e.title || `${e.actor || e.pluginName || 'A draft'} is ready to send.`,
        threadId: e.threadKey ?? null,
      })
    }
  }
  return items
}

export function needsMeCount(state: { jobs?: JobLike[]; connectorEvents?: ConnectorEventLike[] }): number {
  return computeNeedsMe(state).length
}

/**
 * Which needs-me items are NEW since the last computation — so the OS notification
 * fires once per item, not on every state tick. `seen` is the set of ids already
 * notified; returns the newly-appeared items (caller adds their ids to `seen`).
 */
export function newAttention(current: AttentionItem[], seen: Set<string>): AttentionItem[] {
  return current.filter((item) => !seen.has(item.id))
}

/** Reconcile `seen` with what's still pending so resolved items can re-notify later. */
export function pruneSeen(seen: Set<string>, current: AttentionItem[]): Set<string> {
  const live = new Set(current.map((i) => i.id))
  return new Set([...seen].filter((id) => live.has(id)))
}

/** Title + body for a native notification. Body is kept short for the OS surface. */
export function attentionNotification(item: AttentionItem): { title: string; body: string } {
  return { title: `TensorFold · ${item.title}`, body: item.detail.slice(0, 140) }
}
