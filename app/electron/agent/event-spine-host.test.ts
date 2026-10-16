// Run: jiti electron/agent/event-spine-host.test.ts
import assert from 'node:assert/strict'
import { createEventSpineHost, spineInputForAgentEvent } from './event-spine-host'
import { parseSpineJournal } from './event-spine-journal'
import type { AgentEvent } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('event-spine-host tests:')

function memFs() {
  const files = new Map<string, string>()
  return {
    files,
    async appendFile(file: string, data: string) { files.set(file, (files.get(file) ?? '') + data) },
    async readFile(file: string) {
      const content = files.get(file)
      if (content == null) throw new Error('ENOENT')
      return content
    },
    async writeFile(file: string, data: string) { files.set(file, data) },
    async rename(from: string, to: string) {
      const content = files.get(from)
      if (content == null) throw new Error('ENOENT')
      files.set(to, content)
      files.delete(from)
    },
  }
}

const fixedNow = () => 1_700_000_000_000

// --- boot → record → reboot: the journal is the source of truth ---
{
  const fs = memFs()
  const first = await createEventSpineHost({ journalFile: '/spine.jsonl', fs, now: fixedNow })
  first.record({ aggregate: 'thread', streamId: 'A', type: 'agent.status', payload: { status: 'thinking' } })
  first.executeCommand({ commandId: 'c1', aggregate: 'thread', streamId: 'A', kind: 'thread.create' }, () => (
    { aggregate: 'thread', streamId: 'A', type: 'thread.created', payload: {} }
  ))
  first.advanceCursor('renderer', 2)
  await first.flush()

  const second = await createEventSpineHost({ journalFile: '/spine.jsonl', fs, now: fixedNow })
  assert.equal(second.spine.headSequence, 2, 'events restored across a restart')
  assert.equal(second.spine.cursor('renderer'), 2, 'cursors restored')
  assert.equal(second.receipt('c1')?.status, 'accepted', 'receipts restored — the same command replayed after restart stays exactly-once')
  let reran = 0
  second.executeCommand({ commandId: 'c1', aggregate: 'thread', streamId: 'A', kind: 'thread.create' }, () => { reran++; return [] })
  assert.equal(reran, 0, 'the restored receipt short-circuits the handler')
  assert.equal(second.record({ aggregate: 'thread', streamId: 'A', type: 't', payload: {} }).sequence, 3, 'sequences continue after restore')
  await second.flush()
  ok('host: boot→record→reboot keeps events, cursors, receipts, exactly-once')
}

// --- threshold compaction rewrites the journal without losing racing appends ---
{
  const fs = memFs()
  const host = await createEventSpineHost({ journalFile: '/spine.jsonl', fs, now: fixedNow, compactEvery: 10, keepRecent: 2 })
  // A projector cursor is what allows compaction (the watermark is bounded by the slowest cursor).
  for (let i = 1; i <= 9; i++) host.record({ aggregate: 'thread', streamId: 'A', type: `t${i}`, payload: {} })
  host.advanceCursor('renderer', 9)
  // The 10th record trips the compaction check; an 11th lands right behind the rewrite.
  host.record({ aggregate: 'thread', streamId: 'A', type: 't10', payload: {} })
  host.record({ aggregate: 'thread', streamId: 'A', type: 't11', payload: {} })
  await host.flush()

  const state = parseSpineJournal(fs.files.get('/spine.jsonl') ?? '')
  assert.ok((state.compactedThrough ?? 0) > 0, 'compaction happened')
  const sequences = (state.events ?? []).map((event) => event.sequence)
  assert.ok(sequences.includes(11), 'the append racing the rewrite survived (serialized through the writer chain)')
  assert.ok(!sequences.includes(1), 'old events are gone from the journal')

  const rebooted = await createEventSpineHost({ journalFile: '/spine.jsonl', fs, now: fixedNow })
  assert.equal(rebooted.spine.headSequence, 11, 'head survives compaction + reboot')
  assert.equal(rebooted.readSince(0).snapshotRequired, true, 'pre-watermark readers are told to re-baseline')
  ok('host: threshold compaction rewrites atomically, racing appends survive')
}

// --- compaction never outruns a slow projector ---
{
  const fs = memFs()
  const host = await createEventSpineHost({ journalFile: '/spine.jsonl', fs, now: fixedNow, compactEvery: 5, keepRecent: 1 })
  host.advanceCursor('slow', 2)
  for (let i = 1; i <= 12; i++) host.record({ aggregate: 'job', streamId: 'j', type: `t${i}`, payload: {} })
  await host.flush()
  assert.equal(host.spine.compactionWatermark, 2, 'the slow projector pins the watermark')
  assert.equal(host.readSince(2).events.length, 10, 'everything past the slow cursor is still replayable')
  ok('host: the slowest cursor pins compaction')
}

// --- security purge: live state + journal are erased without cursor ambiguity ---
{
  const fs = memFs()
  const host = await createEventSpineHost({ journalFile: '/spine.jsonl', fs, now: fixedNow })
  host.record({ aggregate: 'thread', streamId: 'secret-thread', type: 'agent.status', payload: { marker: 'SECRET-SPINE' } })
  host.executeCommand({ commandId: 'secret-command', aggregate: 'thread', streamId: 'secret-thread', kind: 'thread.update' }, () => (
    { aggregate: 'thread', streamId: 'secret-thread', type: 'thread.updated', payload: { marker: 'SECRET-RECEIPT' } }
  ))
  host.advanceCursor('secret-projector', 2)
  await host.flush()
  assert.equal(host.recordCount(), 4, 'precondition: two events + receipt + cursor are retained')

  await host.purge()
  assert.equal(host.recordCount(), 0, 'purge removes live events, receipts, and cursors')
  assert.equal(host.receipt('secret-command'), null, 'exactly-once receipt is forgotten')
  assert.equal(host.readSince(0).snapshotRequired, true, 'pre-purge clients are forced to re-baseline')
  const journal = fs.files.get('/spine.jsonl') ?? ''
  assert.doesNotMatch(journal, /SECRET-SPINE|SECRET-RECEIPT|secret-thread|secret-command|secret-projector/, 'durable journal contains no purged identifiers or payloads')

  const next = host.record({ aggregate: 'thread', streamId: 'fresh', type: 'thread.created', payload: {} })
  assert.equal(next.sequence, 3, 'post-purge events do not reuse pre-purge sequence numbers')
  await host.flush()
  const rebooted = await createEventSpineHost({ journalFile: '/spine.jsonl', fs, now: fixedNow })
  assert.equal(rebooted.readSince(0).snapshotRequired, true)
  assert.deepEqual(rebooted.readSince(2).events.map((event) => event.streamId), ['fresh'])
  ok('host purge: live + durable replay state erased, old cursors re-baseline')
}

// --- AgentEvent mapping: structural whitelist, lean payloads ---
{
  const base = { runId: 'r1', threadId: 'T1' }
  const added = spineInputForAgentEvent({ ...base, type: 'item_add', item: { id: 'i1', kind: 'tool', tool: 'run_command', args: { command: 'rm -rf /' }, text: 'SECRET transcript' } as any } as AgentEvent)!
  assert.equal(added.type, 'agent.item-added')
  assert.deepEqual(added.payload, { itemId: 'i1', kind: 'tool', tool: 'run_command' }, 'no args/text leak into the spine')
  assert.equal(added.streamId, 'T1')
  assert.equal(added.correlationId, 'r1', 'runId rides as correlation')

  assert.equal(spineInputForAgentEvent({ ...base, type: 'item_delta', id: 'i1', field: 'text', chunk: 'streaming…' } as AgentEvent), null, 'per-token deltas never hit the spine')
  assert.equal(spineInputForAgentEvent({ ...base, type: 'item_update', id: 'i1', patch: { output: 'big blob' } } as AgentEvent), null, 'output patches are streaming noise')
  const stateChange = spineInputForAgentEvent({ ...base, type: 'item_update', id: 'i1', patch: { state: 'ok', output: 'big blob' } } as AgentEvent)!
  assert.deepEqual(stateChange.payload, { itemId: 'i1', state: 'ok' }, 'state transitions are recorded WITHOUT the output')

  const secretTitle = spineInputForAgentEvent({ ...base, type: 'title', title: 'SECRET TITLE FROM PROMPT' } as AgentEvent)!
  assert.deepEqual(secretTitle.payload, { changed: true }, 'derived thread titles are not duplicated into the plaintext spine')
  const err = spineInputForAgentEvent({ ...base, type: 'error', message: 'SECRET ERROR RESPONSE BODY' } as AgentEvent)!
  assert.deepEqual(err.payload, { failed: true }, 'error detail is not duplicated into the plaintext spine')
  assert.doesNotMatch(JSON.stringify([secretTitle, err]), /SECRET TITLE|SECRET ERROR/, 'serialized spine inputs contain no title/error secret')
  assert.equal(spineInputForAgentEvent({ ...base, type: 'done' } as AgentEvent)?.type, 'agent.run-done')
  assert.equal(spineInputForAgentEvent({ ...base, type: 'approval_request', callId: 'c', itemId: 'i' } as AgentEvent)?.type, 'agent.approval-requested')
  const planned = spineInputForAgentEvent({ ...base, type: 'plan', plan: [{}, {}, {}] as any, goalStatus: 'active' } as AgentEvent)!
  assert.deepEqual(planned.payload, { steps: 3, goalStatus: 'active' })
  ok('spineInputForAgentEvent: whitelist + lean payloads (no text/args/output)')
}

console.log(`\n${passed} passed`)
