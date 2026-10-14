// Pure conversation-state checkpoint / rewind / fork model.
//
// COMPLEMENT to workspace-checkpoints.ts, NOT a replacement:
//   - workspace-checkpoints.ts snapshots the FILESYSTEM (git tree of the worktree).
//   - THIS module snapshots the CONVERSATION (apiMessages + timeline + goal/plan).
// The two are deliberately separate so a user can rewind the dialogue without
// touching files on disk, rewind files without rewinding the dialogue, or pair
// them by capturing both at the same logical turn (the caller holds a fileSha to
// correlate a conversation snapshot with a workspace checkpoint commit).
//
// Fully PURE + dependency-injected: no fs, no git, no Date.now, no crypto RNG at
// module scope. The clock and id generator are injected (mirrors the
// clientFactory / fetchImpl / runner injection used elsewhere in this codebase),
// so every operation is deterministic and unit-testable with zero I/O.

import { createHash } from 'node:crypto'

// ---- Minimal structural shapes ----------------------------------------------
// Kept structural (not importing the app's ApiMessage/TimelineItem) so this
// module stays standalone and the integrator can pass the real objects without a
// runtime import across the renderer/main type boundary. Any object with these
// fields is accepted; extra fields are preserved verbatim through snapshot/restore.

export interface ConvMessage {
  role: string
  content: unknown
  [key: string]: unknown
}

export interface ConvTimelineItem {
  id: string
  kind: string
  [key: string]: unknown
}

/** The slice of a Thread that constitutes pure "conversation state". */
export interface ConversationState {
  apiMessages: ConvMessage[]
  timeline: ConvTimelineItem[]
  goal?: string
  goalStatus?: string
  plan?: unknown[]
  status?: string
}

export interface Clock {
  now(): number
}

/** Injected id generator. Kept explicit so snapshots are reproducible in tests. */
export type IdFactory = () => string

export interface CheckpointDeps {
  clock: Clock
  newId: IdFactory
}

// A monotonically increasing id source keeps snapshots totally ordered even when
// two are created in the same millisecond. Callers may inject their own (uuid).
export function sequentialIdFactory(prefix = 'cvk'): IdFactory {
  let n = 0
  return () => `${prefix}_${(++n).toString(36)}`
}

export const systemClock: Clock = { now: () => Date.now() }

// ---- Snapshot model ---------------------------------------------------------

export interface ConversationCheckpoint {
  id: string
  // The snapshot this one was captured from / branched off. null => a root.
  parentId: string | null
  // Fork lineage: which branch this snapshot lives on. Root + linear rewinds keep
  // 'main'; a fork opens a new named branch so alternates never collide.
  branch: string
  label?: string
  createdAt: number
  // Deep-frozen, structurally-cloned conversation state at capture time.
  state: ConversationState
  // Sizes are precomputed so a picker UI never has to walk the (possibly large)
  // arrays just to render "N messages · M events".
  messageCount: number
  timelineCount: number
  // sha256 over the canonical state — cheap identity / dedupe / change detection.
  fingerprint: string
  // Optional correlation to a workspace-checkpoints.ts commit captured at the same
  // turn, so a UI can offer "also restore files to this point".
  fileSha?: string | null
}

export const MAIN_BRANCH = 'main'

// ---- Deep clone + freeze (no structuredClone dependency) --------------------
// JSON round-trip is intentional: conversation state is plain data (strings,
// numbers, booleans, arrays, plain objects). It drops undefined-valued keys and
// functions, which is exactly what we want in a persisted snapshot. Dates/Maps
// are never present in this state.
function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value ?? null)) as T
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object') {
    for (const key of Object.keys(value as Record<string, unknown>)) {
      deepFreeze((value as Record<string, unknown>)[key])
    }
    Object.freeze(value)
  }
  return value
}

// Canonical, key-sorted JSON so the fingerprint is stable regardless of key order.
function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize)
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      out[key] = canonicalize((value as Record<string, unknown>)[key])
    }
    return out
  }
  return value
}

/** sha256 over the canonicalized conversation state. Order-stable, side-effect-free. */
export function fingerprintConversation(state: ConversationState): string {
  const canonical = canonicalize({
    apiMessages: state.apiMessages ?? [],
    timeline: state.timeline ?? [],
    goal: state.goal ?? '',
    goalStatus: state.goalStatus ?? '',
    plan: state.plan ?? [],
    status: state.status ?? '',
  })
  return createHash('sha256').update(JSON.stringify(canonical)).digest('hex')
}

function normalizeState(state: ConversationState): ConversationState {
  const cloned = deepClone({
    apiMessages: Array.isArray(state.apiMessages) ? state.apiMessages : [],
    timeline: Array.isArray(state.timeline) ? state.timeline : [],
    goal: state.goal,
    goalStatus: state.goalStatus,
    plan: Array.isArray(state.plan) ? state.plan : undefined,
    status: state.status,
  })
  return cloned
}

// ---- Capture ----------------------------------------------------------------

export interface CaptureOptions {
  parentId?: string | null
  branch?: string
  label?: string
  fileSha?: string | null
}

/**
 * Snapshot the CURRENT conversation state into an immutable checkpoint. The state
 * is deep-cloned then deep-frozen, so later mutation of the live thread can never
 * leak back into a captured snapshot (and vice-versa). Pure: reads clock + newId
 * only through the injected deps.
 */
export function captureConversationCheckpoint(
  deps: CheckpointDeps,
  state: ConversationState,
  options: CaptureOptions = {},
): ConversationCheckpoint {
  const normalized = normalizeState(state)
  const checkpoint: ConversationCheckpoint = {
    id: deps.newId(),
    parentId: options.parentId ?? null,
    branch: options.branch ?? MAIN_BRANCH,
    label: options.label,
    createdAt: deps.clock.now(),
    state: normalized,
    messageCount: normalized.apiMessages.length,
    timelineCount: normalized.timeline.length,
    fingerprint: fingerprintConversation(normalized),
    fileSha: options.fileSha ?? null,
  }
  return deepFreeze(checkpoint)
}

// ---- Restore (rewind) -------------------------------------------------------

/**
 * Rewind: produce a FRESH conversation state deep-cloned from a snapshot, ready to
 * assign back onto the live thread. Returns a new mutable object each call — the
 * frozen snapshot is never handed out, so the caller can keep editing after
 * restoring without corrupting the checkpoint.
 */
export function restoreConversationCheckpoint(checkpoint: ConversationCheckpoint): ConversationState {
  return normalizeState(checkpoint.state)
}

// ---- Fork -------------------------------------------------------------------

export interface ForkResult {
  branch: string
  // The new snapshot that roots the alternate branch (a copy of `from` on the new
  // branch, so the fork point itself is addressable on that branch).
  checkpoint: ConversationCheckpoint
  // The conversation state to load into the forked thread. Identical content to
  // `from`, freshly cloned + mutable.
  state: ConversationState
}

export interface ForkOptions {
  branch?: string
  label?: string
}

// Derive a fresh, collision-free branch name from existing branches.
export function nextForkBranch(existing: Iterable<string>, base = 'fork'): string {
  const taken = new Set(existing)
  let n = 1
  let name = `${base}-${n}`
  while (taken.has(name)) name = `${base}-${++n}`
  return name
}

/**
 * Fork an alternate history from a PAST snapshot. The returned checkpoint is a
 * copy of `from` re-homed on a new branch (parentId = from.id), and the returned
 * state is a mutable clone to seed the new thread. This is what powers "branch the
 * conversation here and try a different reply" without disturbing the original
 * branch — the original snapshot is untouched (it's frozen; we clone).
 */
export function forkConversationCheckpoint(
  deps: CheckpointDeps,
  from: ConversationCheckpoint,
  existingBranches: Iterable<string>,
  options: ForkOptions = {},
): ForkResult {
  const branch = options.branch ?? nextForkBranch(existingBranches)
  const checkpoint = captureConversationCheckpoint(deps, from.state, {
    parentId: from.id,
    branch,
    label: options.label ?? `fork of ${from.label ?? from.id}`,
    fileSha: from.fileSha ?? null,
  })
  return { branch, checkpoint, state: restoreConversationCheckpoint(checkpoint) }
}

// ---- Log / lineage helpers --------------------------------------------------

/**
 * A tiny append-only checkpoint log with rewind + fork built in. This is the
 * ergonomic surface the integrator wires to the thread; it holds no I/O and can be
 * serialized wholesale (it's just an array of frozen snapshots + a HEAD pointer).
 */
export class ConversationCheckpointLog {
  private readonly deps: CheckpointDeps
  private readonly items: ConversationCheckpoint[] = []
  private headId: string | null = null

  constructor(deps: CheckpointDeps = { clock: systemClock, newId: sequentialIdFactory() }) {
    this.deps = deps
  }

  /** All snapshots, capture order. Returns a shallow copy so callers can't splice internals. */
  list(): ConversationCheckpoint[] {
    return this.items.slice()
  }

  /** Distinct branch names present in the log. */
  branches(): string[] {
    return Array.from(new Set(this.items.map((c) => c.branch)))
  }

  get(id: string): ConversationCheckpoint | undefined {
    return this.items.find((c) => c.id === id)
  }

  head(): ConversationCheckpoint | undefined {
    return this.headId ? this.get(this.headId) : undefined
  }

  /**
   * Capture the live state as the next snapshot on the current HEAD's branch, with
   * HEAD as parent. Advances HEAD. Dedup: if the state is byte-identical to HEAD,
   * no new snapshot is created (returns the existing HEAD) — capturing an unchanged
   * turn just clutters the log.
   */
  capture(state: ConversationState, options: { label?: string; fileSha?: string | null } = {}): ConversationCheckpoint {
    const parent = this.head()
    if (parent && parent.fingerprint === fingerprintConversation(normalizeState(state))) {
      return parent
    }
    const checkpoint = captureConversationCheckpoint(this.deps, state, {
      parentId: parent?.id ?? null,
      branch: parent?.branch ?? MAIN_BRANCH,
      label: options.label,
      fileSha: options.fileSha ?? null,
    })
    this.items.push(checkpoint)
    this.headId = checkpoint.id
    return checkpoint
  }

  /**
   * Rewind HEAD to a past snapshot and return its state to load. Does not delete
   * later snapshots (they stay as an alternate future you can fork from) — it only
   * moves HEAD. Throws on an unknown id so a bad rewind can't silently no-op.
   */
  rewind(id: string): ConversationState {
    const target = this.get(id)
    if (!target) throw new Error(`conversation checkpoint not found: ${id}`)
    this.headId = target.id
    return restoreConversationCheckpoint(target)
  }

  /**
   * Fork a new branch off a past snapshot, append the fork-root snapshot, set HEAD
   * to it, and return the state to seed the alternate thread. Original branch is
   * untouched. Throws on an unknown id.
   */
  fork(id: string, options: ForkOptions = {}): ForkResult {
    const from = this.get(id)
    if (!from) throw new Error(`conversation checkpoint not found: ${id}`)
    const result = forkConversationCheckpoint(this.deps, from, this.branches(), options)
    this.items.push(result.checkpoint)
    this.headId = result.checkpoint.id
    return result
  }

  /** Walk parent links from a snapshot back to its root (inclusive), newest→oldest. */
  lineage(id: string): ConversationCheckpoint[] {
    const chain: ConversationCheckpoint[] = []
    let current = this.get(id)
    const seen = new Set<string>()
    while (current && !seen.has(current.id)) {
      seen.add(current.id)
      chain.push(current)
      current = current.parentId ? this.get(current.parentId) : undefined
    }
    return chain
  }
}
