import assert from 'node:assert/strict'
import {
  distillProceduralNotes,
  formatProceduralNotesForPrompt,
  normalizeReason,
  type ProceduralNote,
  type Trajectory,
} from './procedural-distiller.ts'
import type { OrchestrationJournalEvent } from '../../electron/agent/orchestration-journal.ts'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

let seqCounter = 0
function evt(type: string, payload: Record<string, unknown>, ts = 1000): OrchestrationJournalEvent {
  return { seq: ++seqCounter, type, actor: 'room', ts, payload }
}

function trajectory(subject: string, events: OrchestrationJournalEvent[]): Trajectory {
  return { subject, events }
}

function byId(notes: ProceduralNote[], id: string): ProceduralNote | undefined {
  return notes.find((n) => n.id === id)
}

// --- normalizeReason -------------------------------------------------------

{
  const a = normalizeReason('run 7 failed: exit 1 in /tmp/abc123def with sha 9f8e7d6c5b4a')
  const b = normalizeReason('run 9 failed: exit 1 in /tmp/other456 with sha 1a2b3c4d5e6f')
  assert.equal(a, b, 'volatile run/path/number/sha tokens strip to the same normalized reason')
  assert.ok(a.includes('failed') && a.includes('exit'), 'stable words survive normalization')
  ok('normalizeReason collapses volatile tokens so recurrences cluster')
}

{
  // A reason that is ONLY volatile tokens must not normalize to empty (falls back to raw).
  const n = normalizeReason('abc123def456789')
  assert.ok(n.length > 0, 'all-volatile reason falls back to a non-empty string')
  ok('normalizeReason never returns empty for a non-empty input')
}

// --- clean trajectories yield no notes -------------------------------------

{
  const notes = distillProceduralNotes([
    trajectory('room_ok', [
      evt('started', { state: 'running' }),
      evt('turn', { seat: 'planner', tick: 1 }),
      evt('finalized', { state: 'complete', seat: 'planner' }),
    ]),
  ])
  assert.equal(notes.length, 0, 'a clean successful run distills to zero notes')
  ok('clean success produces no procedural notes')
}

// --- a single failure yields one low-confidence note -----------------------

{
  const notes = distillProceduralNotes([
    trajectory('room_1', [
      evt('started', { state: 'running' }),
      evt('finalized', { state: 'failed', reason: 'error', error: 'tests need env DATABASE_URL' }),
    ]),
  ])
  assert.equal(notes.length, 1, 'one failure -> one note')
  assert.equal(notes[0].category, 'failure')
  assert.equal(notes[0].occurrences, 1)
  assert.equal(notes[0].confidence, 'low')
  assert.ok(notes[0].text.includes('DATABASE_URL'), 'note text carries the extracted reason')
  assert.deepEqual(notes[0].subjects, ['room_1'])
  ok('single failure -> one low-confidence note citing the reason')
}

// --- recurrence raises confidence and clusters across subjects -------------

{
  const mkFail = (subject: string, ts: number) =>
    trajectory(subject, [
      // Same underlying failure; the only differences are volatile (ms + run number),
      // which normalization strips so all four cluster into one note.
      evt('finalized', { state: 'failed', error: `migration timed out after ${ts}ms on run ${ts}` }, ts),
    ])
  const notes = distillProceduralNotes([
    mkFail('mission_a', 100),
    mkFail('mission_b', 200),
    mkFail('mission_c', 300),
    mkFail('mission_d', 400),
  ])
  assert.equal(notes.length, 1, 'same normalized failure across 4 subjects clusters to one note')
  const note = notes[0]
  assert.equal(note.occurrences, 4)
  assert.equal(note.confidence, 'high', '4 occurrences -> high confidence')
  assert.equal(note.subjects.length, 4, 'all four subjects recorded')
  assert.ok(note.text.toLowerCase().includes('repeatedly'), 'recurring note uses pattern framing')
  assert.ok(note.lastTs === 400 && note.firstSeq < note.lastSeq, 'provenance spans first..last event')
  ok('recurrence clusters across subjects and raises confidence to high')
}

// --- stall, retry, incomplete categories -----------------------------------

{
  const notes = distillProceduralNotes([
    trajectory('room_x', [
      evt('seat_stalled', { seat: 'deploy', note: 'no progress: waiting on network' }),
    ]),
    trajectory('mission_y', [
      evt('requeued', { action: 'continue', reason: 'verifier requested another pass' }),
    ]),
    trajectory('room_z', [
      evt('finalized', { state: 'idle', reason: 'no_final', aborted: false }),
    ]),
    trajectory('room_w', [
      evt('finalized', { state: 'idle', aborted: true }),
    ]),
  ])
  const cats = new Set(notes.map((n) => n.category))
  assert.ok(cats.has('stall'), 'seat_stalled -> stall note')
  assert.ok(cats.has('retry'), 'requeued-with-reason -> retry note')
  assert.ok(cats.has('incomplete'), 'no_final / aborted -> incomplete note')
  const stall = notes.find((n) => n.category === 'stall')!
  assert.ok(stall.subjects[0].includes('#deploy'), 'stall note is attributed to the seat')
  ok('stall / retry / incomplete signals each produce their category')
}

// --- requeue without a reason is not a lesson ------------------------------

{
  const notes = distillProceduralNotes([
    trajectory('mission_q', [evt('requeued', { action: 'continue' })]),
  ])
  assert.equal(notes.length, 0, 'a plain requeue with no reason produces no note')
  ok('reasonless requeue is ignored (not every retry is a lesson)')
}

// --- minOccurrences filter + ranking ---------------------------------------

{
  const notes = distillProceduralNotes(
    [
      trajectory('m1', [evt('finalized', { state: 'failed', error: 'flaky once' }, 10)]),
      trajectory('m2', [evt('finalized', { state: 'failed', error: 'persistent boom' }, 20)]),
      trajectory('m3', [evt('finalized', { state: 'failed', error: 'persistent boom' }, 30)]),
    ],
    { minOccurrences: 2 },
  )
  assert.equal(notes.length, 1, 'minOccurrences drops the one-off failure')
  assert.equal(notes[0].occurrences, 2)
  ok('minOccurrences filters out one-off flakes')
}

{
  // Ranking: more-recurrent first regardless of recency.
  const notes = distillProceduralNotes([
    trajectory('r1', [evt('finalized', { state: 'failed', error: 'rare late failure' }, 9999)]),
    trajectory('r2', [evt('finalized', { state: 'failed', error: 'common failure' }, 10)]),
    trajectory('r3', [evt('finalized', { state: 'failed', error: 'common failure' }, 20)]),
  ])
  assert.equal(notes[0].occurrences, 2, 'the recurrent note ranks above the more-recent one-off')
  ok('notes rank by recurrence before recency')
}

// --- stable ids across distills --------------------------------------------

{
  const build = () =>
    distillProceduralNotes([
      trajectory('s1', [evt('finalized', { state: 'failed', error: 'stable reason here' }, 5)]),
    ])[0]
  seqCounter = 500
  const first = build()
  seqCounter = 900
  const second = build()
  assert.equal(first.id, second.id, 'same category+reason yields a stable id across distills')
  assert.ok(byId([first], first.id), 'id lookup works')
  ok('note ids are stable and content-derived')
}

// --- prompt formatting -----------------------------------------------------

{
  const notes = distillProceduralNotes([
    trajectory('p1', [evt('finalized', { state: 'failed', error: 'boom' }, 1)]),
    trajectory('p2', [evt('finalized', { state: 'failed', error: 'boom' }, 2)]),
  ])
  const block = formatProceduralNotesForPrompt(notes)
  assert.ok(block.startsWith('- ['), 'prompt block is a bulleted list')
  assert.ok(block.includes('failure') && block.includes('seen 2'), 'block cites category + recurrence count')
  ok('formatProceduralNotesForPrompt renders a compact cited block')
}

// --- robustness: empty / malformed input -----------------------------------

{
  assert.deepEqual(distillProceduralNotes([]), [], 'empty input -> empty output')
  assert.deepEqual(
    distillProceduralNotes([{ subject: '', events: [] }, { subject: 'x', events: [] }]),
    [],
    'subjectless / eventless trajectories produce nothing',
  )
  ok('handles empty and malformed input without throwing')
}

console.log(`\n${passed} procedural-distiller assertions passed`)
