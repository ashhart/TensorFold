import type { ArchiveSnapshot, Room, RoomEventItem, Thread, TimelineItem } from '../../types'
import { projectedHostId, projectionOriginsMatch } from './proxy-state'

export function mergeTimelineItems(base: TimelineItem[] = [], live: TimelineItem[] = []): TimelineItem[] {
  const order: string[] = []
  const byId = new Map<string, TimelineItem>()
  const put = (item: TimelineItem) => {
    if (!byId.has(item.id)) order.push(item.id)
    const existing = byId.get(item.id)
    if (!existing) {
      byId.set(item.id, { ...item })
      return
    }
    const merged = { ...existing, ...item }
    for (const field of ['text', 'output'] as const) {
      const a = (existing as any)[field]
      const b = (item as any)[field]
      if (typeof a === 'string' && typeof b === 'string') (merged as any)[field] = b.length >= a.length ? b : a
    }
    byId.set(item.id, merged)
  }
  base.forEach(put)
  live.forEach(put)
  return order.map((id) => byId.get(id)!).filter(Boolean)
}

// Reconcile a server room snapshot (returned by api.updateRoom) with the live in-store room.
// While `await api.updateRoom(...)` is in flight the orchestrator can stream `event`/`final`/`status`
// items into the current room via applyRoomEvent; the awaited snapshot predates them. Replacing the
// room wholesale with the snapshot silently drops that streamed activity, so union the events by id
// and preserve the live state/finalText for any field the caller's patch did not explicitly set.
export function mergeRoomSnapshot(current: Room, server: Room, patch: Partial<Room>): Room {
  const byId = new Map<string, RoomEventItem>()
  const order: string[] = []
  const put = (item: RoomEventItem) => {
    if (!byId.has(item.id)) order.push(item.id)
    byId.set(item.id, item)
  }
  // Server events first (authoritative ordering for persisted items), then any live-only items.
  server.events.forEach(put)
  current.events.forEach(put)
  const events = order.map((id) => byId.get(id)!)
  return {
    ...server,
    events,
    state: 'state' in patch ? server.state : current.state,
    finalText: 'finalText' in patch ? server.finalText : (current.finalText || server.finalText),
  }
}

export function mergeThreadRecord(base: Thread, live: Thread): Thread {
  const updatedAt = Math.max(base.updatedAt ?? 0, live.updatedAt ?? 0)
  const newer = (live.updatedAt ?? 0) >= (base.updatedAt ?? 0)
  const baseConversationRevision = base.conversationRevision ?? 0
  const liveConversationRevision = live.conversationRevision ?? 0
  const authoritativeConversation = baseConversationRevision > liveConversationRevision
  const staleConversation = baseConversationRevision < liveConversationRevision
  return {
    ...(newer ? base : live),
    ...(newer ? live : base),
    updatedAt,
    conversationRevision: Math.max(baseConversationRevision, liveConversationRevision),
    // A checkpoint revert deliberately SHORTENS both arrays. Accept that exact host
    // conversation only when its monotonic epoch advances; ordinary same-epoch snapshots
    // still union with live deltas so an in-flight persistence reply cannot erase streaming.
    timeline: authoritativeConversation ? base.timeline : staleConversation ? live.timeline : mergeTimelineItems(base.timeline, live.timeline),
    apiMessages: authoritativeConversation
      ? base.apiMessages
      : staleConversation
        ? live.apiMessages
      : (live.apiMessages?.length ?? 0) >= (base.apiMessages?.length ?? 0) ? live.apiMessages : base.apiMessages,
  }
}

export function mergeThreadLists(loaded: Thread[], live: Thread[]): Thread[] {
  const byId = new Map<string, Thread>()
  for (const thread of loaded) byId.set(thread.id, thread)
  for (const thread of live) {
    const existing = byId.get(thread.id)
    if (existing && !projectionOriginsMatch(projectedHostId(existing), projectedHostId(thread))) continue
    byId.set(thread.id, existing ? mergeThreadRecord(existing, thread) : thread)
  }
  return [...byId.values()].sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))
}

function archiveOriginHostId(item: { orchestratorHostId?: string | null; orchestratorId?: string | null }): string | null {
  return item.orchestratorHostId ?? item.orchestratorId ?? null
}

export { archiveOriginHostId }

function mergeArchiveRecords<T extends { id: string; updatedAt?: number | null; orchestratorHostId?: string | null }>(groups: T[][]): T[] {
  const byOrigin = new Map<string, T>()
  for (const item of groups.flat()) {
    const key = `${archiveOriginHostId(item as T & { orchestratorId?: string | null }) ?? 'local'}:${item.id}`
    const existing = byOrigin.get(key)
    if (!existing || (item.updatedAt ?? 0) >= (existing.updatedAt ?? 0)) byOrigin.set(key, item)
  }
  return [...byOrigin.values()].sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))
}

export function mergeArchiveSnapshots(snapshots: ArchiveSnapshot[]): ArchiveSnapshot {
  return {
    projects: mergeArchiveRecords(snapshots.map((snapshot) => snapshot.projects)),
    threads: mergeArchiveRecords(snapshots.map((snapshot) => snapshot.threads)),
    rooms: mergeArchiveRecords(snapshots.map((snapshot) => snapshot.rooms)),
    tickets: mergeArchiveRecords(snapshots.map((snapshot) => snapshot.tickets)),
  }
}
