// Turn checkpoints, plan lineage, and handoff (VST-036): the stitch layer that unifies the
// three existing pure checkpoint modules into ONE per-turn unit and adds lineage and
// handoff records (proposed plan → implementation thread; cross-engine thread handoff).
//
// - workspace-checkpoints.ts owns the FILE side (git refs + diffs).
// - conversation-checkpoints.ts owns the CONVERSATION side (message/timeline snapshots,
//   fingerprints, fork branches).
// - plan-gate.ts owns the FROZEN PLAN for a turn.
//
// This module is pure: it composes those (a git checkpoint is represented by its ref/sha
// string — the async git I/O stays in workspace-checkpoints), so revert/fork/lineage/handoff
// are all testable without a repo or a live engine.

import {
  restoreConversationCheckpoint,
  forkConversationCheckpoint,
  systemClock,
  sequentialIdFactory,
  MAIN_BRANCH,
  type ConversationCheckpoint,
  type ConversationState,
  type ConvMessage,
  type CheckpointDeps,
} from './conversation-checkpoints'

// A per-turn checkpoint ties the file side, the conversation side, and the plan together.
export interface TurnCheckpoint {
  threadId: string
  turn: number
  createdAt: number
  // File side (workspace-checkpoints): the git ref + short sha captured at this turn, or
  // null when the thread has no git workspace (chat, non-repo cwd).
  fileRef: string | null
  fileSha: string | null
  // Conversation side (conversation-checkpoints): the snapshot id + identity fields.
  conversationId: string
  fingerprint: string
  messageCount: number
  branch: string
  // The frozen plan id in effect at this turn, if any (plan-gate).
  planId: string | null
  label: string | null
}

export interface RecordTurnInput {
  threadId: string
  turn: number
  conversation: ConversationCheckpoint
  fileRef?: string | null
  fileSha?: string | null
  planId?: string | null
  label?: string | null
}

// The atomic revert result: restore the files to `fileRef` (caller runs the git restore) AND
// load `conversation` back into the thread — both from the same turn, so a revert is never
// half-applied (files-only or conversation-only).
export interface TurnRevert {
  turn: number
  fileRef: string | null
  conversation: ConversationState
}

export interface TurnFork {
  turn: number
  branch: string
  // The git ref to fork the files from (caller creates the fork ref); mirrors the
  // conversation branch so files and history diverge together.
  fileRef: string | null
  conversation: ConversationState
  checkpoint: ConversationCheckpoint
}

// A ledger of a thread's per-turn checkpoints. Pure + injected clock/id; holds the
// conversation checkpoints so revert/fork can compose the conversation module directly.
export class TurnCheckpointLedger {
  private entries = new Map<number, { turn: TurnCheckpoint; conversation: ConversationCheckpoint }>()
  private readonly deps: CheckpointDeps

  constructor(deps: Partial<CheckpointDeps> = {}) {
    this.deps = { clock: deps.clock ?? systemClock, newId: deps.newId ?? sequentialIdFactory('turn') }
  }

  record(input: RecordTurnInput): TurnCheckpoint {
    const turn: TurnCheckpoint = {
      threadId: input.threadId,
      turn: input.turn,
      createdAt: this.deps.clock.now(),
      fileRef: input.fileRef ?? null,
      fileSha: input.fileSha ?? null,
      conversationId: input.conversation.id,
      fingerprint: input.conversation.fingerprint,
      messageCount: input.conversation.messageCount,
      branch: input.conversation.branch,
      planId: input.planId ?? null,
      label: input.label ?? input.conversation.label ?? null,
    }
    this.entries.set(input.turn, { turn, conversation: input.conversation })
    return turn
  }

  list(): TurnCheckpoint[] {
    return [...this.entries.values()].map((entry) => entry.turn).sort((a, b) => a.turn - b.turn)
  }

  get(turn: number): TurnCheckpoint | null {
    return this.entries.get(turn)?.turn ?? null
  }

  // Revert files + conversation to `turn` as one unit (spec: "revert restores files +
  // conversation"). Throws on an unknown turn rather than silently no-op'ing.
  revertTo(turn: number): TurnRevert {
    const entry = this.entries.get(turn)
    if (!entry) throw new Error(`No turn checkpoint for turn ${turn}`)
    return {
      turn,
      fileRef: entry.turn.fileRef,
      conversation: restoreConversationCheckpoint(entry.conversation),
    }
  }

  // Fork an alternate branch from `turn`: a fresh conversation branch (isolated clone) + the
  // file ref to fork from. Mutating the returned state never touches the parent (spec: "fork
  // isolation").
  forkFrom(turn: number, options: { branch?: string; label?: string } = {}): TurnFork {
    const entry = this.entries.get(turn)
    if (!entry) throw new Error(`No turn checkpoint for turn ${turn}`)
    const existing = this.list().map((cp) => cp.branch)
    const fork = forkConversationCheckpoint(this.deps, entry.conversation, existing, { branch: options.branch, label: options.label })
    return {
      turn,
      branch: fork.branch,
      fileRef: entry.turn.fileRef,
      conversation: fork.state,
      checkpoint: fork.checkpoint,
    }
  }

  // ---- serialization (persist the ledger across restarts) ----

  /**
   * Plain-JSON snapshot of the ledger, oldest→newest. Each entry carries the turn summary AND
   * its full conversation checkpoint, so a restored ledger can revert/fork without the process
   * that recorded it. `maxTurns` keeps the newest N turns — per-turn conversation snapshots
   * multiply a thread's size on disk, so the persister bounds them.
   */
  serialize(options: { maxTurns?: number } = {}): SerializedTurnLedger {
    const ordered = this.list()
    const kept = options.maxTurns != null && options.maxTurns >= 0 && ordered.length > options.maxTurns
      ? ordered.slice(-options.maxTurns)
      : ordered
    return {
      version: 1,
      entries: kept.map((turn) => ({ turn: { ...turn }, conversation: this.entries.get(turn.turn)!.conversation })),
    }
  }

  /**
   * Fill this ledger from a serialized blob. Defensive: malformed entries are skipped, and a
   * turn that is ALREADY present wins over the persisted copy (an entry recorded live since
   * boot is newer than anything on disk). Returns how many entries were restored.
   */
  restoreEntries(blob: unknown): number {
    if (!blob || typeof blob !== 'object') return 0
    const entries = (blob as { entries?: unknown }).entries
    if (!Array.isArray(entries)) return 0
    let restored = 0
    for (const raw of entries) {
      const entry = raw as { turn?: Partial<TurnCheckpoint> | null; conversation?: (Partial<ConversationCheckpoint> & { id?: unknown }) | null }
      const turnNo = entry?.turn?.turn
      const conversation = entry?.conversation
      if (typeof turnNo !== 'number' || !Number.isFinite(turnNo)) continue
      if (!conversation || typeof conversation !== 'object' || typeof conversation.id !== 'string') continue
      if (this.entries.has(turnNo)) continue
      const turn: TurnCheckpoint = {
        threadId: typeof entry.turn?.threadId === 'string' ? entry.turn.threadId : '',
        turn: turnNo,
        createdAt: typeof entry.turn?.createdAt === 'number' ? entry.turn.createdAt : 0,
        fileRef: typeof entry.turn?.fileRef === 'string' ? entry.turn.fileRef : null,
        fileSha: typeof entry.turn?.fileSha === 'string' ? entry.turn.fileSha : null,
        conversationId: conversation.id,
        fingerprint: typeof conversation.fingerprint === 'string' ? conversation.fingerprint : '',
        messageCount: typeof conversation.messageCount === 'number' ? conversation.messageCount : 0,
        branch: typeof conversation.branch === 'string' ? conversation.branch : MAIN_BRANCH,
        planId: typeof entry.turn?.planId === 'string' ? entry.turn.planId : null,
        label: typeof entry.turn?.label === 'string' ? entry.turn.label : null,
      }
      // A restored conversation is mutable JSON (the frozen original didn't survive the disk
      // round-trip) — safe, because revertTo/forkFrom always hand out fresh clones, never the
      // stored object itself.
      this.entries.set(turnNo, { turn, conversation: conversation as ConversationCheckpoint })
      restored++
    }
    return restored
  }

  /**
   * Rebuild a ledger from a serialized blob. Callers that will FORK from a restored ledger
   * should inject a collision-free `newId` (e.g. uuid) — the default sequential factory
   * restarts at 1 and could re-mint ids the restored snapshots already use.
   */
  static restore(blob: unknown, deps: Partial<CheckpointDeps> = {}): TurnCheckpointLedger {
    const ledger = new TurnCheckpointLedger(deps)
    ledger.restoreEntries(blob)
    return ledger
  }
}

/** Plain-JSON persisted form of a ledger (see TurnCheckpointLedger.serialize). */
export interface SerializedTurnLedger {
  version: 1
  entries: Array<{ turn: TurnCheckpoint; conversation: ConversationCheckpoint }>
}

/** The on-disk shape for a host's turn-checkpoint state: per-thread ledgers + plan lineage. */
export interface PersistedTurnCheckpoints {
  version: 1
  ledgers: Record<string, SerializedTurnLedger>
  planLinks: PlanImplementationLink[]
}

// ---- plan → implementation lineage ------------------------------------------------------

// A proposed plan (drafted in a "plan" thread) that was implemented in a separate
// "implementation" thread, optionally spawned from a Kanban ticket.
export interface PlanImplementationLink {
  planId: string
  planThreadId: string
  implementationThreadId: string
  ticketId: string | null
  createdAt: number
}

export function linkPlanImplementation(
  input: { planId: string; planThreadId: string; implementationThreadId: string; ticketId?: string | null },
  clock = systemClock,
): PlanImplementationLink {
  if (!input.planId || !input.planThreadId || !input.implementationThreadId) {
    throw new Error('linkPlanImplementation requires planId, planThreadId, and implementationThreadId')
  }
  if (input.planThreadId === input.implementationThreadId) {
    throw new Error('A plan cannot be implemented in its own plan thread — implementation runs in a child thread')
  }
  return {
    planId: input.planId,
    planThreadId: input.planThreadId,
    implementationThreadId: input.implementationThreadId,
    ticketId: input.ticketId ?? null,
    createdAt: clock.now(),
  }
}

// ---- cross-engine handoff ---------------------------------------------------------------

export interface HandoffPayload {
  fromThreadId: string
  fromEngine: string
  toEngine: string
  atTurn: number
  // A single seed user-message flattening the carried transcript, so the receiving engine
  // (which has no shared session) starts with the full context as plain text — the same
  // trick the delegated CLIs use to resume a fresh session with prior history.
  seedMessage: ConvMessage
  // The raw messages carried across (for engines that can ingest structured history).
  transcript: ConvMessage[]
  createdAt: number
}

function flattenTranscript(messages: ConvMessage[]): string {
  const lines: string[] = []
  for (const message of messages) {
    const role = String(message.role || 'user')
    const content = typeof message.content === 'string'
      ? message.content
      : Array.isArray(message.content)
        ? message.content.map((part: any) => (typeof part === 'string' ? part : part?.text ?? '')).join(' ')
        : ''
    const trimmed = content.trim()
    if (!trimmed) continue
    lines.push(`[${role}] ${trimmed}`)
  }
  return lines.join('\n\n')
}

// Build the handoff payload from a thread's conversation at a turn. Continuity guarantee: the
// seed message contains every non-empty prior message in order, so the new engine can answer
// as if it had been in the conversation all along (spec: "handoff transcript continuity").
export function buildHandoff(input: {
  fromThreadId: string
  fromEngine: string
  toEngine: string
  atTurn: number
  conversation: ConversationState
}, clock = systemClock): HandoffPayload {
  if (input.fromEngine === input.toEngine) {
    throw new Error('Handoff requires a different target engine')
  }
  const transcript = (input.conversation.apiMessages ?? []).filter((message) => {
    const content = message.content
    if (typeof content === 'string') return content.trim().length > 0
    return Array.isArray(content) ? content.length > 0 : content != null
  })
  const flattened = flattenTranscript(transcript)
  const seedMessage: ConvMessage = {
    role: 'user',
    content: `You are continuing a conversation handed off from ${input.fromEngine}. Here is the full transcript so far — continue from where it leaves off:\n\n${flattened}`,
  }
  return {
    fromThreadId: input.fromThreadId,
    fromEngine: input.fromEngine,
    toEngine: input.toEngine,
    atTurn: input.atTurn,
    seedMessage,
    transcript,
    createdAt: clock.now(),
  }
}
