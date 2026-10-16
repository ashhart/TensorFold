// Host-side event spine assembly (VST-034): restore the spine from its journal on boot,
// tail every new fact back into the journal, compact on a threshold, and map live
// AgentEvents into compact spine records. Pure factory — the filesystem is injected, so
// tests run on an in-memory fs and main-app passes node:fs/promises.

import { createEventSpine, type CommandReceipt, type EventSpine, type ReadSinceOptions, type ReadSinceResult, type SpineCommand, type SpineEvent, type SpineEventInput } from './event-spine'
import { readSpineJournal, rewriteSpineJournal, spineJournalWriter, type SpineJournalFs } from './event-spine-journal'
import type { AgentEvent } from './types'

export interface EventSpineHostDeps {
  journalFile: string
  fs: SpineJournalFs
  onError?: (message: string) => void
  now?: () => number
  // Check for compaction every N records (default 2000); always keep the newest N events
  // (default 500) so fresh subscribers can catch up without an immediate snapshot.
  compactEvery?: number
  keepRecent?: number
}

export interface EventSpineHost {
  readonly spine: EventSpine
  record<TPayload>(input: SpineEventInput<TPayload>): SpineEvent<TPayload>
  executeCommand(command: SpineCommand, handler: () => Array<SpineEventInput> | SpineEventInput | void): CommandReceipt
  advanceCursor(projector: string, sequence: number): number
  readSince(since: number, options?: ReadSinceOptions): ReadSinceResult
  receipt(commandId: string): CommandReceipt | null
  recordCount(): number
  /** Clear live replay/idempotency state and durably rewrite the journal before returning. */
  purge(): Promise<void>
  // Await all queued journal work (graceful shutdown + tests).
  flush(): Promise<void>
}

export async function createEventSpineHost(deps: EventSpineHostDeps): Promise<EventSpineHost> {
  const restored = await readSpineJournal(deps.fs, deps.journalFile)
  const spine = createEventSpine({ now: deps.now }, restored)
  const writer = spineJournalWriter(deps.fs, deps.journalFile, deps.onError)
  spine.subscribe(writer.onEvent)

  const compactEvery = deps.compactEvery ?? 2000
  const keepRecent = deps.keepRecent ?? 500
  let sinceCompactCheck = 0
  const maybeCompact = () => {
    if (++sinceCompactCheck < compactEvery) return
    sinceCompactCheck = 0
    const target = spine.headSequence - keepRecent
    if (target <= spine.compactionWatermark) return
    const before = spine.compactionWatermark
    const watermark = spine.compactThrough(target)
    if (watermark === before) return
    // The rewrite runs on the writer chain so it can never race a concurrent append onto
    // the file it is about to replace.
    writer.run(() => rewriteSpineJournal(deps.fs, deps.journalFile, spine.snapshotState()))
  }

  return {
    spine,
    record(input) {
      const event = spine.append(input)
      maybeCompact()
      return event
    },
    executeCommand(command, handler) {
      const receipt = spine.execute(command, handler)
      writer.onReceipt(receipt)
      maybeCompact()
      return receipt
    },
    advanceCursor(projector, sequence) {
      const cursor = spine.advance(projector, sequence)
      writer.onCursor(projector, cursor)
      return cursor
    },
    readSince: (since, options) => spine.readSince(since, options),
    receipt: (commandId) => spine.receipt(commandId),
    recordCount: () => spine.recordCount(),
    async purge() {
      spine.clearAll()
      // Queue the empty snapshot behind every earlier append and await the whole writer
      // chain. A later append queues after this rewrite, so no pre-purge event can race back
      // into the journal after the security control reports success.
      writer.run(() => rewriteSpineJournal(deps.fs, deps.journalFile, spine.snapshotState()))
      await writer.flush()
    },
    flush: () => writer.flush(),
  }
}

// Map a live AgentEvent to a compact spine record, or null for events that don't belong in
// the ordered log. Deliberately lean: `item_delta` (per-token streaming) is skipped, and
// payloads carry identifiers/states — never message text or tool args/output. The full
// conversational record already lives in the thread journal; the spine is the ORDERING and
// receipt layer, not a second transcript.
export function spineInputForAgentEvent(event: AgentEvent): SpineEventInput | null {
  const base = { aggregate: 'thread' as const, streamId: event.threadId, correlationId: event.runId }
  switch (event.type) {
    case 'status':
      return { ...base, type: 'agent.status', payload: { status: event.status } }
    case 'item_add':
      return { ...base, type: 'agent.item-added', payload: { itemId: event.item.id, kind: event.item.kind, ...(event.item.tool ? { tool: event.item.tool } : {}) } }
    case 'item_update':
      // Only state transitions are structural; text/output patches are streaming noise.
      if (event.patch.state == null) return null
      return { ...base, type: 'agent.item-state', payload: { itemId: event.id, state: event.patch.state } }
    case 'approval_request':
      return { ...base, type: 'agent.approval-requested', payload: { callId: event.callId, itemId: event.itemId } }
    case 'title':
      // The title is derived from conversation text and already lives in protected threads.json.
      // The ordering spine only needs to say that a title changed; persisting it here would create
      // a second plaintext copy in event-spine.jsonl.
      return { ...base, type: 'agent.titled', payload: { changed: true } }
    case 'plan':
      return { ...base, type: 'agent.plan-updated', payload: { steps: event.plan.length, goalStatus: event.goalStatus ?? null } }
    case 'thread_update':
      return { ...base, type: 'agent.thread-updated', payload: { fields: Object.keys(event.patch) } }
    case 'done':
      return { ...base, type: 'agent.run-done', payload: {} }
    case 'error':
      // Provider/tool errors frequently embed prompts, paths, response bodies, or credentials.
      // Keep the structural failure fact while leaving the detailed message in protected thread
      // state and the redacted audit pipeline.
      return { ...base, type: 'agent.run-errored', payload: { failed: true } }
    default:
      return null
  }
}
