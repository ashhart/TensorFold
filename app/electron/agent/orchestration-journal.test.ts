import assert from 'node:assert/strict'
import {
  appendJournalEvent,
  readJournal,
  parseJournal,
  journalPathFor,
  missionSnapshotFromJournal,
  roomSnapshotFromJournal,
  recordMissionEvent,
  recordRoomEvent,
  enableOrchestrationJournal,
  isOrchestrationJournalEnabled,
  quiesceOrchestrationJournal,
  MAX_ORCHESTRATION_SEQ_CACHE_ENTRIES,
  _resetOrchestrationJournalState,
  _orchestrationJournalCacheSizes,
  type OrchestrationJournalFsDeps,
  type MissionJournalEvent,
  type RoomJournalEvent,
} from './orchestration-journal'

// In-memory fs fake: satisfies OrchestrationJournalFsDeps with no real disk, so the
// append/replay round-trip is exercised purely. mkdir is a no-op; append concatenates;
// readFile throws ENOENT-style for an absent file so the "absent reads empty" path runs.
function memFs(): OrchestrationJournalFsDeps & { files: Map<string, string>; appendCalls: number } {
  const files = new Map<string, string>()
  return {
    files,
    appendCalls: 0,
    async mkdir() { return undefined },
    async appendFile(file, data) {
      this.appendCalls++
      files.set(file, (files.get(file) ?? '') + data)
    },
    async readFile(file) {
      const value = files.get(file)
      if (value === undefined) throw new Error(`ENOENT: ${file}`)
      return value
    },
  }
}

async function main() {
  const baseDir = '/virtual/journals/missions'

  // --- journalPathFor shape + traversal guard ---
  assert.equal(journalPathFor('/base', 'abc'), '/base/abc.ndjson')
  assert.throws(() => journalPathFor('/base', '../escape'), /unsafe id/)
  assert.throws(() => journalPathFor('/base', 'a/b'), /unsafe id/)
  assert.throws(() => journalPathFor('/base', ''), /unsafe id/)

  // --- append N events, read back in seq order, seq strictly monotonic from 1 ---
  _resetOrchestrationJournalState()
  const fs = memFs()
  const N = 8
  for (let i = 0; i < N; i++) {
    const { seq } = await appendJournalEvent(
      baseDir,
      'm1',
      { type: 'status_changed', actor: 'supervisor', ts: 1000 + i, payload: { i } },
      fs,
    )
    assert.equal(seq, i + 1, 'sequential appends count up from 1')
  }
  assert.equal(fs.appendCalls, N, 'one append per event')
  const entries = await readJournal(baseDir, 'm1', fs)
  assert.equal(entries.length, N)
  for (let i = 0; i < N; i++) {
    assert.equal(entries[i].seq, i + 1)
    assert.equal(entries[i].type, 'status_changed')
    assert.equal(entries[i].actor, 'supervisor')
    assert.equal(entries[i].ts, 1000 + i)
    assert.equal((entries[i].payload as any).i, i)
    if (i > 0) assert.ok(entries[i].seq > entries[i - 1].seq, 'seq strictly monotonic')
  }

  // optional fields round-trip only when provided
  _resetOrchestrationJournalState()
  const fs2 = memFs()
  await appendJournalEvent(baseDir, 'm-opt', { type: 'created', actor: 'system', ts: 1 }, fs2)
  await appendJournalEvent(baseDir, 'm-opt', { type: 'checkpoint', actor: 'supervisor', ts: 2, payload: { label: 'p1' }, causationId: 'c-1' }, fs2)
  const optEntries = await readJournal(baseDir, 'm-opt', fs2)
  assert.equal(optEntries[0].payload, undefined, 'absent payload stays absent')
  assert.equal(optEntries[0].causationId, undefined, 'absent causationId stays absent')
  assert.equal((optEntries[1].payload as any).label, 'p1')
  assert.equal(optEntries[1].causationId, 'c-1', 'causationId round-trips')

  // --- concurrent appends to the same id do not interleave / fork the seq ---
  _resetOrchestrationJournalState()
  const fs3 = memFs()
  const M = 40
  const results = await Promise.all(
    Array.from({ length: M }, (_, i) => appendJournalEvent(baseDir, 'concurrent', { type: 'turn', actor: 'seat', payload: { i } }, fs3)),
  )
  const seqs = results.map((r) => r.seq).sort((a, b) => a - b)
  assert.deepEqual(seqs, Array.from({ length: M }, (_, i) => i + 1), 'every seq 1..M assigned exactly once')
  const cEntries = await readJournal(baseDir, 'concurrent', fs3)
  assert.equal(cEntries.length, M, 'no lost or duplicated lines under concurrency')
  assert.equal(new Set(cEntries.map((e) => e.seq)).size, M, 'all seqs unique on disk')

  // Settled tails are dropped, while an active journal keeps its cheap next-seq cache.
  await Promise.resolve()
  assert.deepEqual(
    _orchestrationJournalCacheSizes(),
    { tails: 0, nextSeq: 1 },
    'successful settled journal entries drop their promise tail but retain one bounded seq entry',
  )

  // Finalization evicts the remaining entry. A later append re-derives from disk and
  // remains monotonic without retaining completed missions/rooms forever.
  const finalized = await appendJournalEvent(baseDir, 'concurrent', { type: 'finalized', actor: 'seat' }, fs3)
  assert.equal(finalized.seq, M + 1)
  await Promise.resolve()
  assert.deepEqual(_orchestrationJournalCacheSizes(), { tails: 0, nextSeq: 0 })
  const afterEviction = await appendJournalEvent(baseDir, 'concurrent', { type: 'turn', actor: 'seat' }, fs3)
  assert.equal(afterEviction.seq, M + 2, 'same-file seq remains monotonic after terminal cache eviction')

  // distinct ids keep independent seq counters
  const otherFirst = await appendJournalEvent(baseDir, 'other', { type: 'turn', actor: 'seat' }, fs3)
  assert.equal(otherFirst.seq, 1, 'distinct ids have independent seq counters')

  _resetOrchestrationJournalState()
  const distinctResults = await Promise.all(
    Array.from({ length: MAX_ORCHESTRATION_SEQ_CACHE_ENTRIES + 32 }, (_, i) => appendJournalEvent(baseDir, `bounded-${i}`, { type: 'turn', actor: 'seat' }, fs3)),
  )
  assert.ok(distinctResults.every(({ seq }) => seq === 1), 'new journal files each start at seq 1')
  await Promise.resolve()
  assert.deepEqual(
    _orchestrationJournalCacheSizes(),
    { tails: 0, nextSeq: MAX_ORCHESTRATION_SEQ_CACHE_ENTRIES },
    'settled writes across many non-finalized journals retain only the bounded seq cache',
  )
  _resetOrchestrationJournalState()

  // --- corrupt + non-event lines are skipped on parse/read ---
  const goodA = JSON.stringify({ seq: 1, type: 'started', actor: 'room', ts: 10 })
  const goodB = JSON.stringify({ seq: 2, type: 'turn', actor: 'seat', ts: 11 })
  const corrupt = parseJournal(
    [
      goodA,
      '{ not json',
      JSON.stringify({ seq: 3, ts: 12 }), // missing type/actor -> not a valid event
      '',
      goodB,
    ].join('\n'),
  )
  assert.deepEqual(corrupt.map((e) => e.seq), [1, 2], 'corrupt + malformed-shape + blank lines skipped')

  // parse sorts by seq even when written out of order
  const unordered = parseJournal([goodB, goodA].join('\n'))
  assert.deepEqual(unordered.map((e) => e.seq), [1, 2], 'parse returns seq-ordered events')

  // absent journal reads empty, not an error
  assert.deepEqual(await readJournal(baseDir, 'never-existed', memFs()), [])

  // --- mission snapshot projection (read-model replay) ---
  const missionEvents: MissionJournalEvent[] = [
    { seq: 1, type: 'created', actor: 'system', ts: 100, payload: { status: 'queued' } },
    { seq: 2, type: 'started', actor: 'supervisor', ts: 200, payload: { runIndex: 1 } },
    { seq: 3, type: 'checkpoint', actor: 'supervisor', ts: 300, payload: { checkpointId: 'cp-1', label: 'pass 1', sequence: 1, runIndex: 1 } },
    { seq: 4, type: 'requeued', actor: 'supervisor', ts: 400, payload: { reason: 'continue' } },
    { seq: 5, type: 'checkpoint', actor: 'supervisor', ts: 500, payload: { checkpointId: 'cp-2', label: 'pass 2', sequence: 2, runIndex: 2 } },
    { seq: 6, type: 'finalized', actor: 'supervisor', ts: 600, payload: { status: 'completed', runIndex: 2 } },
  ]
  const ms = missionSnapshotFromJournal(missionEvents)
  assert.equal(ms.status, 'completed', 'finalized status wins as last state')
  assert.equal(ms.finalized, true)
  assert.equal(ms.runCount, 2, 'explicit runIndex payload drives runCount')
  assert.equal(ms.lastCheckpoint?.id, 'cp-2', 'latest checkpoint projected')
  assert.equal(ms.lastCheckpoint?.sequence, 2)
  assert.equal(ms.eventCount, 6)
  assert.equal(ms.updatedAt, 600)

  // requeued without an explicit run count increments runCount by one
  const reqOnly = missionSnapshotFromJournal([
    { seq: 1, type: 'started', actor: 'supervisor', ts: 1, payload: { runIndex: 1 } },
    { seq: 2, type: 'requeued', actor: 'supervisor', ts: 2 },
    { seq: 3, type: 'requeued', actor: 'supervisor', ts: 3 },
  ])
  assert.equal(reqOnly.runCount, 3, 'requeued bumps runCount when no explicit count given')
  assert.equal(reqOnly.status, 'queued', 'requeued falls back to queued status')

  // status_changed uses payload.to; replay is order-independent (events shuffled)
  const shuffled = missionSnapshotFromJournal([missionEvents[5], missionEvents[0], missionEvents[3], missionEvents[1], missionEvents[4], missionEvents[2]])
  assert.equal(shuffled.status, 'completed', 'snapshot reorders by seq before projecting')
  assert.equal(shuffled.lastCheckpoint?.id, 'cp-2')

  const statusChange = missionSnapshotFromJournal([
    { seq: 1, type: 'started', actor: 'supervisor', ts: 1 },
    { seq: 2, type: 'status_changed', actor: 'supervisor', ts: 2, payload: { from: 'running', to: 'paused' } },
  ])
  assert.equal(statusChange.status, 'paused', 'status_changed projects payload.to')

  // empty journal -> empty snapshot
  const empty = missionSnapshotFromJournal([])
  assert.equal(empty.status, null)
  assert.equal(empty.runCount, 0)
  assert.equal(empty.lastCheckpoint, null)
  assert.equal(empty.finalized, false)
  assert.equal(empty.updatedAt, null)

  // --- room snapshot projection ---
  const roomEvents: RoomJournalEvent[] = [
    { seq: 1, type: 'started', actor: 'room', ts: 10, payload: { state: 'running' } },
    { seq: 2, type: 'turn', actor: 'seat', ts: 20, payload: { seat: 'Ada' } },
    { seq: 3, type: 'turn', actor: 'seat', ts: 30, payload: { seat: 'Linus' } },
    { seq: 4, type: 'seat_stalled', actor: 'room', ts: 40, payload: { seat: 'Linus' } },
    { seq: 5, type: 'seat_stalled', actor: 'room', ts: 50, payload: { seat: 'Linus' } }, // dup name de-duped
    { seq: 6, type: 'turn', actor: 'seat', ts: 60, payload: { seat: 'Ada' } },
    { seq: 7, type: 'finalized', actor: 'seat', ts: 70, payload: { state: 'complete', seat: 'Ada' } },
  ]
  const rs = roomSnapshotFromJournal(roomEvents)
  assert.equal(rs.state, 'complete', 'finalized state projected')
  assert.equal(rs.finalized, true)
  assert.equal(rs.turnCount, 3, 'counts every turn event')
  assert.deepEqual(rs.stalledSeats, ['Linus'], 'stalled seat names de-duplicated')
  assert.equal(rs.eventCount, 7)
  assert.equal(rs.updatedAt, 70)

  // room snapshot: started without explicit state defaults to running; empty is empty
  const startedDefault = roomSnapshotFromJournal([{ seq: 1, type: 'started', actor: 'room', ts: 1 }])
  assert.equal(startedDefault.state, 'running')
  const emptyRoom = roomSnapshotFromJournal([])
  assert.equal(emptyRoom.state, null)
  assert.equal(emptyRoom.turnCount, 0)
  assert.deepEqual(emptyRoom.stalledSeats, [])
  assert.equal(emptyRoom.finalized, false)

  // --- full round-trip: append events through the fake fs, then snapshot from read ---
  _resetOrchestrationJournalState()
  const fsRoom = memFs()
  const roomBase = '/virtual/journals/rooms'
  await appendJournalEvent(roomBase, 'r1', { type: 'started', actor: 'room', payload: { state: 'running' } }, fsRoom)
  await appendJournalEvent(roomBase, 'r1', { type: 'turn', actor: 'seat', payload: { seat: 'Ada' } }, fsRoom)
  await appendJournalEvent(roomBase, 'r1', { type: 'seat_stalled', actor: 'room', payload: { seat: 'Ada' } }, fsRoom)
  await appendJournalEvent(roomBase, 'r1', { type: 'finalized', actor: 'room', payload: { state: 'waiting_user' } }, fsRoom)
  const replayed = await readJournal<RoomJournalEvent['type']>(roomBase, 'r1', fsRoom)
  const snap = roomSnapshotFromJournal(replayed as RoomJournalEvent[])
  assert.equal(snap.state, 'waiting_user', 'snapshot reconstructed from appended+read journal')
  assert.equal(snap.turnCount, 1)
  assert.deepEqual(snap.stalledSeats, ['Ada'])
  assert.equal(snap.finalized, true)

  // --- purge barrier: an admitted append must settle before writer state is cleared. ---
  _resetOrchestrationJournalState()
  let releaseAppend!: () => void
  const appendGate = new Promise<void>((resolve) => { releaseAppend = resolve })
  const delayedFs = memFs()
  const realAppend = delayedFs.appendFile.bind(delayedFs)
  delayedFs.appendFile = async (file, data, options) => {
    await appendGate
    await realAppend(file, data, options)
  }
  enableOrchestrationJournal(true)
  const admitted = appendJournalEvent('/virtual/journals/missions', 'queued-secret', { type: 'started', actor: 'system' }, delayedFs)
  let barrierDone = false
  const barrier = quiesceOrchestrationJournal().then(() => { barrierDone = true })
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(barrierDone, false, 'quiesce waits for an already-admitted journal append')
  assert.equal(isOrchestrationJournalEnabled(), false, 'quiesce immediately rejects new live journal writes')
  releaseAppend()
  await Promise.all([admitted, barrier])
  assert.equal(barrierDone, true, 'quiesce resolves only after the pending writer settles')

  // --- disk-recorder gate: off by default, so the live recorders are no-ops that never
  // touch the real fs (this is what keeps runRoom/decideNextMissionStep tests clean). ---
  _resetOrchestrationJournalState()
  assert.equal(isOrchestrationJournalEnabled(), false, 'disk journal defaults to disabled')
  // When disabled these resolve without throwing and without writing anything.
  await recordMissionEvent('gate-mission', { type: 'started', actor: 'supervisor' })
  await recordRoomEvent('gate-room', { type: 'started', actor: 'room' })
  // Empty ids are a no-op regardless of the flag.
  enableOrchestrationJournal(true)
  assert.equal(isOrchestrationJournalEnabled(), true, 'flag flips on')
  await recordMissionEvent('', { type: 'started', actor: 'supervisor' })
  await recordRoomEvent('', { type: 'started', actor: 'room' })
  enableOrchestrationJournal(false)
  assert.equal(isOrchestrationJournalEnabled(), false, 'flag flips off')
  _resetOrchestrationJournalState()

  console.log('orchestration-journal: all assertions passed')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
