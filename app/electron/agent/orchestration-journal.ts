// Durable, append-only NDJSON lifecycle journal for missions and rooms.
//
// This is the *on-disk* durability layer that sits underneath the in-memory mission
// journal (`missions.ts#appendMissionJournal`) and the live room event stream
// (`rooms.ts`). Where those hold the current run in process memory, this writes one
// JSON line per lifecycle transition to:
//   ~/.vontra/journals/missions/{id}.ndjson
//   ~/.vontra/journals/rooms/{id}.ndjson
// so that after a crash the current state of a mission/room is reconstructable by
// replaying its journal (see missionSnapshotFromJournal / roomSnapshotFromJournal).
//
// Design mirrors thread-journal.ts: pure + fs-injected (every op takes a baseDir, and
// the fs surface is overridable for no-disk unit tests), corrupt lines are skipped on
// read rather than fatal, and a per-file append chain serializes concurrent writers so
// the seq counter never forks. The disk side is fire-and-forget at the call sites
// (recordMissionEvent / recordRoomEvent swallow errors) so a journal failure can never
// break a live run.

import { promises as realFs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Who caused a lifecycle event. Mirrors MissionJournalActor + a 'room'/'seat' origin
// for room-side events, kept as a plain string union so callers stay decoupled from
// the in-memory journal's narrower actor type.
export type OrchestrationActor = 'system' | 'agent' | 'user' | 'connector' | 'supervisor' | 'seat' | 'room'

// Mission lifecycle event types written to the durable journal. These are the real
// transition points the supervisor/runtime drives a mission through.
export type MissionJournalEventType =
  | 'created'
  | 'started'
  | 'checkpoint'
  | 'finalized'
  | 'requeued'
  | 'status_changed'

// Room lifecycle event types.
export type RoomJournalEventType =
  | 'started'
  | 'turn'
  | 'seat_stalled'
  | 'finalized'

// One durable journal event, as written and read back. `seq` and `ts` are assigned at
// append time; `causationId` optionally links an event to the event/command that
// caused it (e.g. a checkpoint caused by a status_changed) for replay/debugging.
export interface OrchestrationJournalEvent<TType extends string = string> {
  seq: number
  type: TType
  actor: OrchestrationActor
  ts: number
  payload?: Record<string, unknown>
  causationId?: string
}

// What a caller supplies; seq/ts are filled in by the append.
export interface OrchestrationEventInput<TType extends string = string> {
  type: TType
  actor: OrchestrationActor
  ts?: number
  payload?: Record<string, unknown>
  causationId?: string
}

export type MissionJournalEvent = OrchestrationJournalEvent<MissionJournalEventType>
export type RoomJournalEvent = OrchestrationJournalEvent<RoomJournalEventType>

// Minimal fs surface the journal depends on. Defaults to node:fs/promises so callers
// can pass just (baseDir, id, event); tests inject a fake to avoid touching disk.
export interface OrchestrationJournalFsDeps {
  mkdir(dir: string, options: { recursive: boolean }): Promise<unknown>
  appendFile(file: string, data: string, options: { encoding: 'utf8' }): Promise<void>
  readFile(file: string, encoding: 'utf8'): Promise<string>
}

const defaultFs: OrchestrationJournalFsDeps = {
  mkdir: (dir, options) => realFs.mkdir(dir, options),
  appendFile: (file, data, options) => realFs.appendFile(file, data, options),
  readFile: (file, encoding) => realFs.readFile(file, encoding),
}

// The two journal kinds map to two subdirectories under the journals root.
export type OrchestrationJournalKind = 'missions' | 'rooms'

// Default journals root: ~/.vontra/journals, with the same VONTRA_HOME override the
// rest of persistence honours (electron/store.ts). Resolved with os.homedir() rather
// than Electron's `app` so this module stays Electron-free and unit-testable under
// node/jiti (importing `app` would break the agent test runner).
export function journalsRootDir(): string {
  const home = process.env.VONTRA_HOME?.trim() || path.join(os.homedir(), '.vontra')
  return path.join(home, 'journals')
}

// An id is part of a filename, so refuse separators / traversal that would escape baseDir.
function safeId(id: string): string {
  if (!id || /[\\/]/.test(id) || id === '.' || id === '..') {
    throw new Error(`orchestration-journal: unsafe id ${JSON.stringify(id)}`)
  }
  return id
}

/** Absolute path of a mission/room journal file under baseDir (the kind's subdir). */
export function journalPathFor(baseDir: string, id: string): string {
  return path.join(baseDir, `${safeId(id)}.ndjson`)
}

/** Directory holding a given journal kind under the journals root. */
export function journalDirFor(kind: OrchestrationJournalKind): string {
  return path.join(journalsRootDir(), kind)
}

// Per-file tail of the append chain. Concurrent appends to the same file queue behind
// one another so the read-last-seq -> write step never interleaves (which would fork
// the seq counter and corrupt the file). Different files have independent tails.
const tails = new Map<string, Promise<{ seq: number }>>()
// Cached next seq per file so the hot path does not re-scan the whole file each append.
const nextSeq = new Map<string, number>()
export const MAX_ORCHESTRATION_SEQ_CACHE_ENTRIES = 256

function cacheNextSeq(key: string, seq: number): void {
  // Map insertion order gives us a tiny LRU without another data structure. Refresh the
  // current journal, then evict least-recently-written entries at the process-wide bound.
  // An evicted journal safely re-derives its next sequence from disk on its next append.
  nextSeq.delete(key)
  nextSeq.set(key, seq)
  while (nextSeq.size > MAX_ORCHESTRATION_SEQ_CACHE_ENTRIES) {
    const oldest = nextSeq.keys().next().value as string | undefined
    if (oldest === undefined) break
    nextSeq.delete(oldest)
  }
}

/** Highest seq already on disk, or 0 if the journal is absent/empty. Skips corrupt lines. */
async function lastSeqOnDisk(fs: OrchestrationJournalFsDeps, file: string): Promise<number> {
  let raw: string
  try {
    raw = await fs.readFile(file, 'utf8')
  } catch {
    return 0 // no file yet
  }
  let max = 0
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue
    try {
      const parsed = JSON.parse(line)
      if (typeof parsed?.seq === 'number' && parsed.seq > max) max = parsed.seq
    } catch {
      // corrupt line, ignore for seq derivation
    }
  }
  return max
}

/**
 * Append one event to a journal file and return its assigned seq. seq is a monotonic
 * per-file counter; appends to the same file are serialized through a promise chain so
 * concurrent callers each get a distinct, strictly increasing seq.
 */
export function appendJournalEvent<TType extends string>(
  baseDir: string,
  id: string,
  input: OrchestrationEventInput<TType>,
  fs: OrchestrationJournalFsDeps = defaultFs,
): Promise<{ seq: number }> {
  const file = journalPathFor(baseDir, id)
  const key = file // path uniquely identifies the journal + baseDir
  const prior = tails.get(key) ?? Promise.resolve({ seq: 0 })

  const run = prior.then(async () => {
    let seq = nextSeq.get(key)
    if (seq === undefined) seq = (await lastSeqOnDisk(fs, file)) + 1
    const event: OrchestrationJournalEvent<TType> = {
      seq,
      type: input.type,
      actor: input.actor,
      ts: input.ts ?? Date.now(),
      ...(input.payload !== undefined ? { payload: input.payload } : {}),
      ...(input.causationId !== undefined ? { causationId: input.causationId } : {}),
    }
    const line = JSON.stringify(event) + '\n'
    await fs.mkdir(path.dirname(file), { recursive: true })
    await fs.appendFile(file, line, { encoding: 'utf8' })
    cacheNextSeq(key, seq + 1)
    return { seq }
  })

  // Keep the chain alive even if one append throws, but drop the cache so the next
  // append re-derives seq from disk rather than trusting an unwritten counter.
  const tail = run.catch(() => { nextSeq.delete(key); return { seq: 0 } })
  tails.set(key, tail)

  // Retain the promise chain only while this file still has queued work. A newer append
  // replaces `tail` before this callback runs, so the identity check preserves ordering.
  // Keep the cheap next-seq cache for active journals, but evict on terminal lifecycle;
  // cacheNextSeq also caps non-finalized/abandoned journals process-wide.
  void tail.then(() => {
    if (tails.get(key) !== tail) return
    tails.delete(key)
    if (input.type === 'finalized') nextSeq.delete(key)
  })
  return run
}

/** Pure: parse NDJSON journal text into events in seq order. Corrupt lines are skipped. */
export function parseJournal<TType extends string = string>(raw: string): OrchestrationJournalEvent<TType>[] {
  const out: OrchestrationJournalEvent<TType>[] = []
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue
    try {
      const parsed = JSON.parse(line)
      if (
        typeof parsed?.seq === 'number' &&
        typeof parsed?.type === 'string' &&
        typeof parsed?.actor === 'string' &&
        typeof parsed?.ts === 'number'
      ) {
        const event: OrchestrationJournalEvent<TType> = {
          seq: parsed.seq,
          type: parsed.type,
          actor: parsed.actor,
          ts: parsed.ts,
        }
        if (parsed.payload && typeof parsed.payload === 'object') event.payload = parsed.payload
        if (typeof parsed.causationId === 'string') event.causationId = parsed.causationId
        out.push(event)
      }
    } catch {
      // skip corrupt line
    }
  }
  out.sort((a, b) => a.seq - b.seq)
  return out
}

/** Read and parse every journal line for an id in seq order. Absent journal reads empty. */
export async function readJournal<TType extends string = string>(
  baseDir: string,
  id: string,
  fs: OrchestrationJournalFsDeps = defaultFs,
): Promise<OrchestrationJournalEvent<TType>[]> {
  const file = journalPathFor(baseDir, id)
  let raw: string
  try {
    raw = await fs.readFile(file, 'utf8')
  } catch {
    return [] // absent journal reads as empty
  }
  return parseJournal<TType>(raw)
}

// ---------------------------------------------------------------------------
// Read-model snapshots: pure projections that replay a journal's events into the
// current state, so a crashed mission/room is reconstructable from disk alone.
// ---------------------------------------------------------------------------

export interface MissionJournalSnapshot {
  // Last known status from a status_changed/started/created event, or null if unknown.
  status: string | null
  // How many runs the mission has been through (requeued bumps it; a payload.runCount
  // or payload.runIndex wins if present, mirroring missionRunCount's data sources).
  runCount: number
  // The most recent checkpoint's id + label + sequence, or null if none yet.
  lastCheckpoint: { id?: string; label?: string; sequence?: number; ts: number } | null
  // True once a 'finalized' event has been seen.
  finalized: boolean
  // Total lifecycle events replayed (useful as a cheap integrity/length check).
  eventCount: number
  // ts of the last replayed event, or null for an empty journal.
  updatedAt: number | null
}

function numFrom(payload: Record<string, unknown> | undefined, ...keys: string[]): number | null {
  if (!payload) return null
  for (const key of keys) {
    const value = payload[key]
    if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, Math.floor(value))
  }
  return null
}

function strFrom(payload: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = payload?.[key]
  return typeof value === 'string' ? value : undefined
}

/** Replay a mission journal into its current-state snapshot. Pure. */
export function missionSnapshotFromJournal(events: MissionJournalEvent[]): MissionJournalSnapshot {
  const ordered = [...events].sort((a, b) => a.seq - b.seq)
  let status: string | null = null
  let runCount = 0
  let lastCheckpoint: MissionJournalSnapshot['lastCheckpoint'] = null
  let finalized = false
  let updatedAt: number | null = null

  for (const event of ordered) {
    updatedAt = event.ts
    // An explicit run count in the payload always wins (matches missionRunCount).
    const explicitRun = numFrom(event.payload, 'runCount', 'runIndex')
    if (explicitRun != null) runCount = Math.max(runCount, explicitRun)

    switch (event.type) {
      case 'created':
        status = strFrom(event.payload, 'status') ?? status ?? 'queued'
        break
      case 'started':
        status = 'running'
        break
      case 'status_changed': {
        const next = strFrom(event.payload, 'to') ?? strFrom(event.payload, 'status')
        if (next) status = next
        break
      }
      case 'requeued':
        if (explicitRun == null) runCount += 1
        status = strFrom(event.payload, 'status') ?? 'queued'
        break
      case 'checkpoint':
        lastCheckpoint = {
          id: strFrom(event.payload, 'checkpointId') ?? strFrom(event.payload, 'id'),
          label: strFrom(event.payload, 'label'),
          sequence: numFrom(event.payload, 'sequence') ?? undefined,
          ts: event.ts,
        }
        break
      case 'finalized':
        finalized = true
        status = strFrom(event.payload, 'status') ?? strFrom(event.payload, 'to') ?? status
        break
    }
  }

  return { status, runCount, lastCheckpoint, finalized, eventCount: ordered.length, updatedAt }
}

export interface RoomJournalSnapshot {
  // Last known room state, or null if no started/finalized seen.
  state: string | null
  // Count of 'turn' events replayed (how far the deliberation got).
  turnCount: number
  // Names of seats that have been flagged stalled at least once.
  stalledSeats: string[]
  // True once a 'finalized' event has been seen.
  finalized: boolean
  eventCount: number
  updatedAt: number | null
}

/** Replay a room journal into its current-state snapshot. Pure. */
export function roomSnapshotFromJournal(events: RoomJournalEvent[]): RoomJournalSnapshot {
  const ordered = [...events].sort((a, b) => a.seq - b.seq)
  let state: string | null = null
  let turnCount = 0
  const stalled = new Set<string>()
  let finalized = false
  let updatedAt: number | null = null

  for (const event of ordered) {
    updatedAt = event.ts
    switch (event.type) {
      case 'started':
        state = strFrom(event.payload, 'state') ?? 'running'
        break
      case 'turn':
        turnCount += 1
        break
      case 'seat_stalled': {
        const seat = strFrom(event.payload, 'seat')
        if (seat) stalled.add(seat)
        break
      }
      case 'finalized':
        finalized = true
        state = strFrom(event.payload, 'state') ?? state ?? 'complete'
        break
    }
  }

  return { state, turnCount, stalledSeats: [...stalled], finalized, eventCount: ordered.length, updatedAt }
}

// ---------------------------------------------------------------------------
// Fire-and-forget disk recorders for the live call sites. These resolve the real
// journals root, swallow every error, and never throw — a journal failure must never
// break a mission/room run. They return the underlying promise so callers that want to
// await (e.g. tests) can, but the call sites use `void`.
//
// Disk writes are gated behind an explicit enable flag that the Electron host flips on
// at startup (enableOrchestrationJournal). It defaults to OFF so that pure unit tests
// which drive runRoom / decideNextMissionStep directly never touch the real ~/.vontra —
// the same reason real fs/shell only runs inside Electron (see src/lib/ipc.ts). The
// pure append/replay/snapshot functions above are unaffected and testable with an
// injected fs regardless of this flag.
let diskJournalEnabled = false

/** Enable durable on-disk lifecycle journalling. Called once by the Electron host. */
export function enableOrchestrationJournal(enabled = true): void {
  diskJournalEnabled = enabled
}

/** Whether durable on-disk journalling is currently enabled. */
export function isOrchestrationJournalEnabled(): boolean {
  return diskJournalEnabled
}

/**
 * Stop accepting live mission/room journal events, await every append already admitted, then
 * discard the writer tails and sequence cache. Security purge calls this before shredding the
 * journals directory; without the barrier, a fire-and-forget append queued before purge can
 * recreate a deleted journal after the UI reports success.
 */
export async function quiesceOrchestrationJournal(): Promise<void> {
  diskJournalEnabled = false
  const pending = [...tails.values()]
  await Promise.allSettled(pending)
  tails.clear()
  nextSeq.clear()
}

function recordSafely(kind: OrchestrationJournalKind, id: string, input: OrchestrationEventInput): Promise<void> {
  if (!diskJournalEnabled) return Promise.resolve()
  return appendJournalEvent(journalDirFor(kind), id, input)
    .then(() => undefined)
    .catch(() => undefined)
}

/** Append a mission lifecycle event to the durable journal (fire-and-forget, never throws). */
export function recordMissionEvent(missionId: string, input: OrchestrationEventInput<MissionJournalEventType>): Promise<void> {
  if (!missionId) return Promise.resolve()
  return recordSafely('missions', missionId, input)
}

/** Append a room lifecycle event to the durable journal (fire-and-forget, never throws). */
export function recordRoomEvent(roomId: string, input: OrchestrationEventInput<RoomJournalEventType>): Promise<void> {
  if (!roomId) return Promise.resolve()
  return recordSafely('rooms', roomId, input)
}

/** Test-only: clear the in-memory seq cache, append chains, and disk-enable flag. */
export function _resetOrchestrationJournalState(): void {
  tails.clear()
  nextSeq.clear()
  diskJournalEnabled = false
}

/** Test-only: expose writer-cache cardinality without leaking the mutable maps. */
export function _orchestrationJournalCacheSizes(): { tails: number; nextSeq: number } {
  return { tails: tails.size, nextSeq: nextSeq.size }
}
