import type { AgentEvent, RemoteControlEvent } from './types'

export const DEFAULT_REMOTE_STREAM_BUFFER_BYTES = 512 * 1024

export interface RemoteStreamGap {
  reason: 'backpressure'
  droppedEvents: number
  droppedBytes: number
  fromEventId: string
  toEventId: string
  recovery: 'snapshot'
}

interface BufferedEvent {
  event: RemoteControlEvent
  bytes: number
}

/**
 * Bounded per-client queue for a blocked SSE response. Only leading streamed text deltas
 * may be discarded. Ordered lifecycle/mutation events are never overtaken; if one alone
 * would exceed the cap the caller must close the client and let it re-baseline.
 */
export class KeepTailRemoteEventBuffer {
  private readonly entries: BufferedEvent[] = []
  private bytes = 0
  private gap: RemoteStreamGap | null = null

  constructor(private readonly maxBytes = DEFAULT_REMOTE_STREAM_BUFFER_BYTES) {}

  enqueue(event: RemoteControlEvent): { overflow: boolean } {
    const buffered = { event, bytes: remoteEventWireBytes(event) }
    this.entries.push(buffered)
    this.bytes += buffered.bytes

    while (this.bytes > this.maxBytes) {
      const head = this.entries[0]
      // Keep at least the newest event. An individually oversized event must trigger a
      // reconnect/snapshot instead of leaving an unbounded queue or silently truncating JSON.
      if (!head || this.entries.length === 1 || !isRemoteItemDelta(head.event)) return { overflow: true }
      this.entries.shift()
      this.bytes -= head.bytes
      this.recordDrop(head)
    }
    return { overflow: false }
  }

  shift(): RemoteControlEvent | null {
    const head = this.entries.shift()
    if (!head) return null
    this.bytes -= head.bytes
    return head.event
  }

  takeGapEvent(createdAt = Date.now()): RemoteControlEvent | null {
    const gap = this.gap
    this.gap = null
    return gap ? { id: gap.toEventId, kind: 'stream_gap', createdAt, data: gap } : null
  }

  get length(): number { return this.entries.length }
  get byteLength(): number { return this.bytes }
  get hasGap(): boolean { return this.gap !== null }

  clear(): void {
    this.entries.length = 0
    this.bytes = 0
    this.gap = null
  }

  private recordDrop(entry: BufferedEvent): void {
    if (!this.gap) {
      this.gap = {
        reason: 'backpressure',
        droppedEvents: 1,
        droppedBytes: entry.bytes,
        fromEventId: entry.event.id,
        toEventId: entry.event.id,
        recovery: 'snapshot',
      }
      return
    }
    this.gap.droppedEvents++
    this.gap.droppedBytes += entry.bytes
    this.gap.toEventId = entry.event.id
  }
}

export function isRemoteItemDelta(event: Pick<RemoteControlEvent, 'kind' | 'data'>): boolean {
  if (event.kind !== 'agent' || !event.data || typeof event.data !== 'object') return false
  return (event.data as Partial<AgentEvent>).type === 'item_delta'
}

export function remoteEventWireBytes(event: RemoteControlEvent): number {
  return Buffer.byteLength(JSON.stringify(event), 'utf8') + 48
}
