// Run: jiti src/lib/spine-projector.test.ts
//
// Renderer spine projector (VST-034/041): persisted-cursor catch-up, paging, baseline
// silence, digest wording, compaction truncation, unreachable-host quietness.
import assert from 'node:assert/strict'
import { catchUpFromSpine, summarizeSpineEvents } from './spine-projector'
import type { ReadSinceResult, SpineEvent } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('spine-projector tests:')

const ev = (sequence: number, type: string): SpineEvent => ({
  eventId: `e${sequence}`,
  sequence,
  aggregate: 'thread',
  streamId: 't1',
  streamVersion: sequence,
  type,
  payload: {},
  occurredAt: sequence,
  commandId: null,
  causationEventId: null,
  correlationId: null,
  actor: null,
})

function harness(pages: ReadSinceResult[], startCursor = 0) {
  let cursor = startCursor
  let advanced: number | null = null
  const reads: number[] = []
  return {
    deps: {
      readSince: async ({ since }: { since: number }) => {
        reads.push(since)
        return pages.shift() ?? { events: [], hasMore: false, snapshotRequired: false }
      },
      advanceCursor: async ({ sequence }: { projector: string; sequence: number }) => { advanced = sequence; return sequence },
      loadCursor: () => cursor,
      saveCursor: (sequence: number) => { cursor = sequence },
    },
    cursorNow: () => cursor,
    advancedTo: () => advanced,
    reads,
  }
}

{
  assert.equal(summarizeSpineEvents([]), null)
  assert.equal(
    summarizeSpineEvents([ev(1, 'agent.run-done'), ev(2, 'agent.run-done'), ev(3, 'agent.run-errored'), ev(4, 'approval.resolved'), ev(5, 'job.retried')]),
    'While disconnected: 2 runs finished, 1 failed, 1 approval resolved, 1 job change.',
  )
  assert.equal(summarizeSpineEvents([ev(1, 'agent.status')]), null, 'streaming noise never becomes news')
  ok('summarize: counts the notable, ignores the noise')
}

{
  // First run: baseline only — cursor advances, NO summary spam of old history.
  const h = harness([{ events: [ev(1, 'agent.run-done'), ev(2, 'agent.run-errored')], hasMore: false, snapshotRequired: false }])
  const result = await catchUpFromSpine(h.deps)
  assert.equal(result, null, 'baseline is silent')
  assert.equal(h.cursorNow(), 2, 'but the cursor is established')
  assert.equal(h.advancedTo(), 2, 'and mirrored server-side')
  ok('first run: silent baseline, cursor persisted')
}

{
  // Catch-up from a real cursor: pages until !hasMore, digests, advances.
  const h = harness([
    { events: [ev(11, 'agent.run-done'), ev(12, 'thread.created')], hasMore: true, snapshotRequired: false },
    { events: [ev(13, 'approval.resolved')], hasMore: false, snapshotRequired: false },
  ], 10)
  const result = await catchUpFromSpine(h.deps)
  assert.ok(result)
  assert.equal(result!.fromSequence, 10)
  assert.equal(result!.toSequence, 13)
  assert.equal(result!.eventCount, 3)
  assert.match(result!.summary ?? '', /1 run finished, 1 approval resolved, 1 new thread/)
  assert.deepEqual(h.reads, [10, 12], 'pages resume from the last seen sequence')
  assert.equal(h.cursorNow(), 13)
  ok('catch-up: pages, digests, persists the new cursor')
}

{
  // Nothing new: cursor untouched, null result.
  const h = harness([{ events: [], hasMore: false, snapshotRequired: false }], 20)
  assert.equal(await catchUpFromSpine(h.deps), null)
  assert.equal(h.cursorNow(), 20)
  ok('no news: quiet, cursor unchanged')
}

{
  // Cursor below the compaction watermark: truncated flag surfaces (state was re-baselined).
  const h = harness([{ events: [ev(101, 'agent.run-done')], hasMore: false, snapshotRequired: true }], 5)
  const result = await catchUpFromSpine(h.deps)
  assert.equal(result?.truncated, true)
  ok('compacted gap: reported as truncated, never silently complete')
}

{
  // Unreachable host / no spine: stay quiet, keep the cursor for next time.
  const h = harness([], 30)
  h.deps.readSince = async () => { throw new Error('offline') }
  assert.equal(await catchUpFromSpine(h.deps), null)
  assert.equal(h.cursorNow(), 30)
  ok('offline: no digest, cursor preserved for the next attempt')
}

console.log(`  ${passed} passed`)
