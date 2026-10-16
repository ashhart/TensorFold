// Pure mapping from connector events + assistant timeline items to bubble rows for
// the unified Assistant conversation surface ("both together"): incoming source
// messages and TensorFold's drafts render as gray bubbles (left); your replies and
// instructions render as blue bubbles (right). Kept framework-free so it can be
// unit-tested directly.

export type BubbleDirection = 'in' | 'out'

export interface BubbleRow {
  id: string
  direction: BubbleDirection
  author: string
  text: string
  time?: number
  source?: string
  // Outbound bubbles that are agent drafts awaiting approval carry a status the
  // ConversationView renders as an Approve/Edit/Discard affordance.
  status?: 'draft' | 'sent' | 'failed'
  /** Stable sort key (createdAt or a synthesized index) for ordering the column. */
  sort: number
}

// Outbound-looking actions: the agent composed/sent/replied, or the user emitted
// an instruction. Everything else is an inbound source message (gray, left).
const OUTBOUND_ACTIONS = /\b(send|sent|reply|replied|draft|forwarded|outbound|deliver)/i
const OUTBOUND_DIRECTIONS = new Set(['outbound', 'internal'])

/** Decide bubble side from a connector event's direction/action. */
export function eventDirection(input: {
  direction?: string
  action?: string
  isFromMe?: boolean
}): BubbleDirection {
  if (input.isFromMe) return 'out'
  if (input.direction && OUTBOUND_DIRECTIONS.has(input.direction)) return 'out'
  if (input.action && OUTBOUND_ACTIONS.test(input.action)) return 'out'
  return 'in'
}

/**
 * Map a connector event (email/whatsapp/imessage/...) to a bubble row.
 * `isFromMe` is the strongest signal (e.g. iMessage chat.db `is_from_me`); fall
 * back to the event direction/action classification the connector spine already
 * computes.
 */
export function eventToBubble(event: {
  id: string
  actor?: string
  title?: string
  summary?: string
  detail?: string
  text?: string
  action?: string
  direction?: string
  connectorId?: string
  threadKey?: string
  createdAt?: number
  isFromMe?: boolean
  approvalStatus?: string
}): BubbleRow {
  const direction = eventDirection(event)
  const text = (event.text ?? event.summary ?? event.detail ?? event.title ?? '').trim()
  const status: BubbleRow['status'] | undefined =
    direction === 'out'
      ? event.approvalStatus === 'approved'
        ? 'sent'
        : event.approvalStatus === 'denied'
          ? 'failed'
          : event.action && OUTBOUND_ACTIONS.test(event.action) && event.approvalStatus !== 'approved'
            ? 'draft'
            : undefined
      : undefined
  return {
    id: event.id,
    direction,
    author: event.actor || (direction === 'out' ? 'You' : 'Incoming'),
    text,
    time: event.createdAt,
    source: event.connectorId,
    status,
    sort: event.createdAt ?? 0,
  }
}

/** Map a timeline item from an agent task (the TensorFold chat) into a bubble. */
export function timelineItemToBubble(item: {
  id: string
  kind: string
  text?: string
  state?: string
  createdAt?: number
}): BubbleRow | null {
  // user messages and live "Guide" cues are your voice -> blue (out).
  if (item.kind === 'user' || item.kind === 'cue') {
    return { id: item.id, direction: 'out', author: 'You', text: (item.text || '').trim(), status: item.kind === 'cue' ? 'sent' : undefined, sort: item.createdAt ?? 0 }
  }
  // assistant replies are TensorFold's voice -> gray (in).
  if (item.kind === 'assistant') {
    return { id: item.id, direction: 'in', author: 'TensorFold', text: (item.text || '').trim(), sort: item.createdAt ?? 0 }
  }
  // Tool/notice/reasoning/error rows are not conversational — skip them so the
  // bubble column reads as a chat, not a transcript.
  return null
}

/** Merge + stable-sort a mixed list of source bubbles and agent-chat bubbles. */
export function mergeBubbles(rows: BubbleRow[]): BubbleRow[] {
  return [...rows].sort((a, b) => (a.sort || 0) - (b.sort || 0) || (a.id < b.id ? -1 : 1))
}

/**
 * Collapse connector events into one row PER CONVERSATION (threadKey), so e.g. an
 * iMessage chat with 24 messages is a single thread — not 24 rows. The representative
 * is the newest event in the conversation; `count` is the number of messages in it.
 * Events with no threadKey stand alone (keyed by id). Threads are returned newest-first.
 */
export function groupEventsIntoThreads<T extends { id: string; threadKey?: string; createdAt?: number }>(
  events: T[],
): Array<{ event: T; count: number }> {
  const groups = new Map<string, T[]>()
  for (const e of events) {
    const key = (e.threadKey && e.threadKey.trim()) || e.id
    const arr = groups.get(key)
    if (arr) arr.push(e)
    else groups.set(key, [e])
  }
  const reps = [...groups.values()].map((arr) => {
    const newest = arr.reduce((a, b) => ((b.createdAt ?? 0) > (a.createdAt ?? 0) ? b : a))
    return { event: newest, count: arr.length }
  })
  return reps.sort((a, b) => (b.event.createdAt ?? 0) - (a.event.createdAt ?? 0))
}

// Combine pre-sorted (newest-first) items that share a threadKey into ONE thread
// row — fold the rest into its count and key the row by threadKey so selection is
// stable as new messages land. Items without a threadKey pass through untouched.
// Used for the Assistant "attention" list, which emits one item per message (e.g.
// iMessage/Slack), so a conversation would otherwise show as many separate rows.
export function combineItemsByThread<T extends { id: string; threadKey?: string; count?: number }>(items: T[]): T[] {
  const byThread = new Map<string, T>()
  const out: T[] = []
  for (const item of items) {
    const key = item.threadKey?.trim()
    if (!key) { out.push(item); continue }
    const existing = byThread.get(key)
    if (existing) {
      existing.count = (existing.count ?? 1) + (item.count ?? 1)
    } else {
      const entry = { ...item, id: key, count: item.count ?? 1 }
      byThread.set(key, entry)
      out.push(entry)
    }
  }
  return out
}
