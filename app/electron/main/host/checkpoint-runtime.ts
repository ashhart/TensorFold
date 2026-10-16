// Host wiring for per-turn workspace checkpoints (undo-a-turn + turn diff).
//
// This module is intentionally electron-free so it stays unit-testable with a fake
// GitRunner: it only selects a runner (local host vs connected SSH backend),
// resolves the working directory, and delegates to the already-tested pure
// checkpoint primitives in agent/workspace-checkpoints.ts. The IPC surface lives in
// workspace-ipc.ts (which owns `handle`); thread-runtime calls `capture` per turn.

import { exec as cpExec } from 'node:child_process'
import {
  checkpointRef,
  createCheckpoint,
  createForkWorktree,
  deleteCheckpoints,
  diffCheckpointToWorktree,
  isGitRepository,
  listCheckpointTurns,
  restoreCheckpoint,
  type CheckpointResult,
  type GitRunner,
} from '../../agent/workspace-checkpoints'
import type { Thread, TimelineItem, WorkspaceLocation } from '../../agent/types'
import { captureConversationCheckpoint, systemClock, sequentialIdFactory, type ConversationState, type CheckpointDeps } from '../../agent/conversation-checkpoints'
import {
  TurnCheckpointLedger,
  linkPlanImplementation,
  type TurnCheckpoint,
  type TurnRevert,
  type TurnFork,
  type SerializedTurnLedger,
  type PersistedTurnCheckpoints,
  type PlanImplementationLink,
} from '../../agent/turn-checkpoints'

// Local host git runner — captures the exit code (mirrors workspace-ipc's
// localGitRunner; checkpoints need the code to gate each step).
export const localCheckpointRunner: GitRunner = {
  exec: (command, opts) => new Promise((resolve) => {
    cpExec(command, { cwd: opts?.cwd, timeout: 25_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
      const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
      resolve({ code, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
    })
  }),
}

// A connected SSH git backend: an exec runner plus the remote root cwd fallback.
export interface CheckpointBackend extends GitRunner {
  readonly rootCwd: string
}

// Where ledgers + plan links live between restarts (main-app: the protected
// turn-checkpoints.json). save() is fire-and-forget — the store's per-path write
// chain serializes + atomically renames, so a burst of turns can't tear the file.
export interface CheckpointPersistence {
  load(): Promise<PersistedTurnCheckpoints | null>
  save(state: PersistedTurnCheckpoints): Promise<void> | void
}

export interface CheckpointServiceDeps {
  // Resolve a connected SSH git backend for a workspace, or undefined for local.
  resolveBackend: (ws: WorkspaceLocation | null | undefined) => CheckpointBackend | undefined
  // Resolve + validate a local workspace cwd (mirrors resolveWorkspaceCwd).
  resolveCwd: (cwd: string | null | undefined) => Promise<string>
  // Injectable local runner (defaults to localCheckpointRunner; overridden in tests).
  localRunner?: GitRunner
  // Injectable clock + id source for the conversation snapshots recorded per turn
  // (VST-036). Defaults to systemClock + a sequential id factory. Hosts that persist
  // ledgers should inject a collision-free id (uuid) — the sequential default restarts
  // at 1 each boot and could re-mint ids that restored snapshots already carry.
  conversationDeps?: CheckpointDeps
  // Ledger persistence (VST-036). Absent → memory-only (the pre-persistence behavior).
  persistence?: CheckpointPersistence
  // Which threads may be written to disk. main-app: thread still exists AND is not
  // incognito (incognito conversations must never persist — same rule as threads.json).
  isPersistableThread?: (threadId: string) => boolean
  // Newest N turns kept per thread on disk (each turn embeds a full conversation
  // snapshot, so an unbounded ledger multiplies a long thread's footprint).
  maxPersistedTurnsPerThread?: number
  // Destination path factory for fork worktrees (VST-036). Absent → forks always share
  // the source cwd. main-app injects a folder under ~/.vontra/worktrees.
  forkWorktreePath?: (input: { threadId: string; turn: number; branch: string }) => string
}

// Where a thread's worktree lives, plus its kind (chat threads aren't checkpointed).
export interface CheckpointTarget {
  cwd?: string | null
  workspace?: WorkspaceLocation | null
  kind?: Thread['kind']
  /** Host-owned timeline used to scope restore to paths the agent actually changed. */
  timeline?: TimelineItem[]
}

export function agentOwnedPathsFromTurn(timeline: TimelineItem[] | undefined, turn: number): string[] {
  if (!timeline?.length || turn < 1) return []
  let userTurn = 0
  const paths = new Set<string>()
  for (const item of timeline) {
    if (item.kind === 'user') userTurn++
    if (userTurn < turn || item.kind !== 'tool' || !['ok', 'error'].includes(String(item.state ?? ''))) continue
    for (const candidate of item.changedPaths ?? []) {
      const value = String(candidate || '').trim()
      if (value) paths.add(value)
    }
  }
  return [...paths]
}

function preservationWarning(paths: string[] | undefined): string | undefined {
  if (!paths?.length) return undefined
  const shown = paths.slice(0, 5).join(', ')
  const more = paths.length > 5 ? ` and ${paths.length - 5} more` : ''
  return `Preserved ${paths.length} post-checkpoint path${paths.length === 1 ? '' : 's'} because the checkpoint has no prior file version: ${shown}${more}.`
}

// Turn number for the run about to start = count of user messages in the timeline.
// Monotonic and unique per turn. A checkpoint captured at a turn's start snapshots
// pre-turn state, so restoring turn N undoes turn N's file edits.
export function threadTurnNumber(timeline: TimelineItem[]): number {
  return timeline.reduce((n, item) => (item.kind === 'user' ? n + 1 : n), 0)
}

export function createCheckpointService(deps: CheckpointServiceDeps) {
  const local = deps.localRunner ?? localCheckpointRunner
  // Per-thread conversation ledger (VST-036): the file side lives in git refs; this holds
  // the conversation snapshot for each turn so a revert/fork restores BOTH together, closing
  // the old undoLast gap ("only files are reverted — the conversation is left intact").
  const conversationDeps: CheckpointDeps = deps.conversationDeps ?? { clock: systemClock, newId: sequentialIdFactory('turn') }
  const ledgers = new Map<string, TurnCheckpointLedger>()
  const ledgerFor = (threadId: string): TurnCheckpointLedger => {
    let ledger = ledgers.get(threadId)
    if (!ledger) { ledger = new TurnCheckpointLedger(conversationDeps); ledgers.set(threadId, ledger) }
    return ledger
  }

  // Plan → implementation-thread lineage (VST-036), persisted with the ledgers.
  const planLinks: PlanImplementationLink[] = []
  const maxTurns = deps.maxPersistedTurnsPerThread ?? 25

  function persistedState(): PersistedTurnCheckpoints {
    const out: Record<string, SerializedTurnLedger> = {}
    for (const [threadId, ledger] of ledgers) {
      if (deps.isPersistableThread && !deps.isPersistableThread(threadId)) continue
      const serialized = ledger.serialize({ maxTurns })
      if (serialized.entries.length > 0) out[threadId] = serialized
    }
    return { version: 1, ledgers: out, planLinks: [...planLinks] }
  }

  // Fire-and-forget: persistence failures must never block or fail a run (checkpoints
  // are best-effort by contract), and the store's write chain keeps writes ordered.
  function persist(): void {
    if (!deps.persistence) return
    try { void Promise.resolve(deps.persistence.save(persistedState())).catch(() => {}) } catch { /* ignore */ }
  }

  // Pick the runner + cwd: a connected SSH backend wins (remote path), else the
  // local host runner against the resolved local cwd.
  async function resolveRunner(target: CheckpointTarget): Promise<{ runner: GitRunner; cwd: string } | null> {
    const backend = deps.resolveBackend(target.workspace)
    if (backend) {
      const cwd = target.workspace?.path?.trim() || backend.rootCwd
      return cwd ? { runner: backend, cwd } : null
    }
    if (target.workspace?.kind === 'ssh') {
      throw new Error('The SSH workspace is disconnected. Reconnect it before using checkpoints.')
    }
    const cwd = await deps.resolveCwd(target.cwd)
    return cwd ? { runner: local, cwd } : null
  }

  return {
    /**
     * Forget every persisted conversation checkpoint and plan-link in one ordered write.
     * Unlike the normal best-effort `persist()` path, a security purge must be able to
     * await durability and surface a disk failure instead of claiming the snapshots were
     * erased while turn-checkpoints.json still contains them.
     */
    async clearAll(): Promise<void> {
      ledgers.clear()
      planLinks.splice(0, planLinks.length)
      if (deps.persistence) await Promise.resolve(deps.persistence.save(persistedState()))
    },

    /** Number of locally retained checkpoint/link records (used by truthful purge counts). */
    recordCount(): number {
      let count = planLinks.length
      for (const ledger of ledgers.values()) count += ledger.list().length
      return count
    },

    /**
     * Capture a checkpoint at the start of turn `turn`. Best-effort: it never
     * throws and never blocks a run — capture failures (no repo, no workspace,
     * detached transport) are reported as ok:false, not raised.
     */
    async capture(threadId: string, turn: number, target: CheckpointTarget): Promise<CheckpointResult> {
      try {
        if (target.kind === 'chat') return { ok: false, error: 'chat threads are not checkpointed' }
        const r = await resolveRunner(target)
        if (!r) return { ok: false, error: 'no workspace to checkpoint' }
        if (!(await isGitRepository(r.runner, r.cwd))) return { ok: false, error: 'not a git repository' }
        return await createCheckpoint(r.runner, r.cwd, threadId, turn)
      } catch (e) {
        return { ok: false, error: e instanceof Error ? e.message : String(e) }
      }
    },

    /** Enumerate the thread's captured turns (ascending). [] on any failure. */
    async listTurns(threadId: string, target: CheckpointTarget): Promise<number[]> {
      try {
        const r = await resolveRunner(target)
        if (!r) return []
        return await listCheckpointTurns(r.runner, r.cwd, threadId)
      } catch {
        return []
      }
    },

    /**
     * Revert the worktree to before the most recent turn's edits, then drop that
     * checkpoint so a subsequent undo walks back another turn (progressive undo).
     * Only files are reverted — the conversation timeline is left intact.
     */
    async undoLast(threadId: string, target: CheckpointTarget): Promise<{ ok: boolean; turn?: number; error?: string; warning?: string; preservedPaths?: string[] }> {
      const r = await resolveRunner(target)
      if (!r) return { ok: false, error: 'no workspace to restore' }
      const turns = await listCheckpointTurns(r.runner, r.cwd, threadId)
      const last = turns[turns.length - 1]
      if (last == null) return { ok: false, error: 'no checkpoints to undo' }
      const ref = checkpointRef(threadId, last)
      const res = await restoreCheckpoint(r.runner, r.cwd, ref, agentOwnedPathsFromTurn(target.timeline, last))
      if (!res.ok) return { ok: false, turn: last, error: res.error }
      await deleteCheckpoints(r.runner, r.cwd, [ref])
      return { ok: true, turn: last, warning: preservationWarning(res.preservedPaths), preservedPaths: res.preservedPaths }
    },

    /** Unified diff of the most recent turn's checkpoint against the live worktree. */
    async diffLast(threadId: string, target: CheckpointTarget): Promise<{ ok: boolean; turn?: number; diff?: string; error?: string }> {
      const r = await resolveRunner(target)
      if (!r) return { ok: false, error: 'no workspace to diff' }
      const turns = await listCheckpointTurns(r.runner, r.cwd, threadId)
      const last = turns[turns.length - 1]
      if (last == null) return { ok: false, error: 'no checkpoints to diff' }
      const res = await diffCheckpointToWorktree(r.runner, r.cwd, checkpointRef(threadId, last))
      return res.ok ? { ok: true, turn: last, diff: res.diff } : { ok: false, turn: last, error: res.error }
    },

    // ---- unified turn checkpoints (VST-036): files + conversation as one ----

    /**
     * Record the conversation snapshot for a turn alongside its git file checkpoint, so a
     * later revert/fork can restore BOTH. Called at the same per-turn seam as `capture`.
     * `fileSha` correlates to the git checkpoint captured for the same turn. Best-effort:
     * a chat thread or a capture with no conversation is simply not recorded.
     */
    recordTurn(threadId: string, turn: number, conversation: ConversationState, fileSha?: string | null): TurnCheckpoint {
      const snapshot = captureConversationCheckpoint(conversationDeps, conversation, { fileSha: fileSha ?? null, label: `turn ${turn}` })
      const recorded = ledgerFor(threadId).record({
        threadId,
        turn,
        conversation: snapshot,
        fileRef: fileSha ? checkpointRef(threadId, turn) : null,
        fileSha: fileSha ?? null,
      })
      persist()
      return recorded
    },

    /**
     * Revive persisted ledgers + plan links (call once at boot, AFTER threads are loaded so
     * the persistable-thread filter can prune ledgers of deleted threads). Merge-restores:
     * a turn already recorded live this boot wins over the disk copy. Best-effort — an
     * unreadable file starts empty rather than failing boot.
     */
    async restorePersisted(): Promise<{ threads: number; turns: number; links: number }> {
      if (!deps.persistence) return { threads: 0, turns: 0, links: 0 }
      let saved: PersistedTurnCheckpoints | null = null
      try { saved = await deps.persistence.load() } catch { saved = null }
      if (!saved || typeof saved !== 'object') return { threads: 0, turns: 0, links: 0 }
      let threads = 0
      let turns = 0
      const savedLedgers = saved.ledgers && typeof saved.ledgers === 'object' ? saved.ledgers : {}
      for (const [threadId, blob] of Object.entries(savedLedgers)) {
        if (!threadId) continue
        if (deps.isPersistableThread && !deps.isPersistableThread(threadId)) continue
        const restored = ledgerFor(threadId).restoreEntries(blob)
        if (restored > 0) { threads++; turns += restored }
      }
      let links = 0
      if (Array.isArray(saved.planLinks)) {
        for (const raw of saved.planLinks) {
          const link = raw as Partial<PlanImplementationLink> | null
          if (!link || typeof link.planId !== 'string' || typeof link.planThreadId !== 'string' || typeof link.implementationThreadId !== 'string') continue
          if (planLinks.some((existing) => existing.planId === link.planId && existing.implementationThreadId === link.implementationThreadId)) continue
          planLinks.push({
            planId: link.planId,
            planThreadId: link.planThreadId,
            implementationThreadId: link.implementationThreadId,
            ticketId: typeof link.ticketId === 'string' ? link.ticketId : null,
            createdAt: typeof link.createdAt === 'number' ? link.createdAt : 0,
          })
          links++
        }
      }
      return { threads, turns, links }
    },

    /**
     * Seed a freshly-forked thread's ledger with its fork-point checkpoint, so the new
     * thread can immediately revert back to where it diverged. Called by the fork-thread
     * creation path with the new thread's id and the TurnFork that seeded it.
     */
    adoptFork(newThreadId: string, fork: TurnFork): TurnCheckpoint {
      const adopted = ledgerFor(newThreadId).record({
        threadId: newThreadId,
        turn: fork.turn,
        conversation: fork.checkpoint,
        fileRef: fork.fileRef,
        fileSha: null,
      })
      persist()
      return adopted
    },

    /** Record proposed-plan → implementation-thread lineage (persisted with the ledgers). */
    recordPlanLink(input: { planId: string; planThreadId: string; implementationThreadId: string; ticketId?: string | null }): PlanImplementationLink {
      const link = linkPlanImplementation(input, conversationDeps.clock)
      planLinks.push(link)
      persist()
      return link
    },

    /** All recorded plan → implementation links (copy — callers can't splice internals). */
    planImplementationLinks(): PlanImplementationLink[] {
      return [...planLinks]
    },

    /** The thread's recorded turn checkpoints (turn-ordered), for a picker UI. */
    turnCheckpoints(threadId: string): TurnCheckpoint[] {
      return ledgers.get(threadId)?.list() ?? []
    },

    /**
     * Revert a thread to `turn` — restore the worktree files to that turn's git ref AND
     * return the conversation state to load back into the thread, atomically. The caller
     * (IPC) applies the conversation to the Thread + persists. Unlike `undoLast`, this
     * targets a specific turn and does not delete later checkpoints (they stay forkable).
     */
    async revertTurn(threadId: string, turn: number, target: CheckpointTarget): Promise<{ ok: boolean; turn: number; conversation?: ConversationState; error?: string; warning?: string; preservedPaths?: string[] }> {
      const ledger = ledgers.get(threadId)
      const recorded = ledger?.get(turn)
      if (!ledger || !recorded) return { ok: false, turn, error: `no turn checkpoint for turn ${turn}` }
      let revert: TurnRevert
      try { revert = ledger.revertTo(turn) } catch (e) { return { ok: false, turn, error: e instanceof Error ? e.message : String(e) } }
      // Restore files when the thread has a workspace; a chat/no-repo thread reverts
      // conversation-only (still a valid, useful revert).
      let preservedPaths: string[] | undefined
      let fileWarning: string | undefined
      if (target.kind !== 'chat') {
        const r = recorded.fileRef ? await resolveRunner(target) : null
        if (recorded.fileRef && r && await isGitRepository(r.runner, r.cwd)) {
          const res = await restoreCheckpoint(r.runner, r.cwd, recorded.fileRef, agentOwnedPathsFromTurn(target.timeline, turn))
          if (!res.ok) return { ok: false, turn, error: res.error }
          preservedPaths = res.preservedPaths
        } else {
          fileWarning = recorded.fileRef
            ? 'The file checkpoint is unavailable, so only the conversation was reverted.'
            : 'No file checkpoint was captured for this turn, so only the conversation was reverted.'
        }
      }
      const warning = [fileWarning, preservationWarning(preservedPaths)].filter(Boolean).join(' ') || undefined
      return { ok: true, turn, conversation: revert.conversation, warning, preservedPaths }
    },

    /**
     * Fork an alternate branch from `turn`: returns the isolated conversation state to seed
     * a new thread + the branch name + the file ref to diverge from. With
     * `options.worktree`, additionally materializes the fork point into its OWN detached
     * git worktree so the fork edits files without touching the source worktree. A
     * worktree failure never fails the fork itself — it surfaces as `worktreeError` so the
     * caller can show an explicit notice and fall back to the shared cwd.
     */
    async forkTurn(
      threadId: string,
      turn: number,
      target: CheckpointTarget,
      options: { branch?: string; label?: string; worktree?: boolean } = {},
    ): Promise<{ ok: boolean; error?: string; fork?: TurnFork; worktreePath?: string; worktreeError?: string }> {
      const ledger = ledgers.get(threadId)
      if (!ledger || !ledger.get(turn)) return { ok: false, error: `no turn checkpoint for turn ${turn}` }
      let fork: TurnFork
      try { fork = ledger.forkFrom(turn, { branch: options.branch, label: options.label }) } catch (e) {
        return { ok: false, error: e instanceof Error ? e.message : String(e) }
      }
      if (!options.worktree) return { ok: true, fork }
      // Fork worktrees are LOCAL-only for now: an SSH backend would need remote path
      // management. The fork itself still succeeds on the shared cwd.
      if (target.workspace?.kind === 'ssh') return { ok: true, fork, worktreeError: 'fork worktrees are local-only for now' }
      if (!deps.forkWorktreePath) return { ok: true, fork, worktreeError: 'no fork worktree directory is configured' }
      if (!fork.fileRef) return { ok: true, fork, worktreeError: 'this fork point has no file snapshot' }
      try {
        const r = await resolveRunner(target)
        if (!r || !(await isGitRepository(r.runner, r.cwd))) return { ok: true, fork, worktreeError: 'the source thread has no git repository' }
        const dest = deps.forkWorktreePath({ threadId, turn, branch: fork.branch })
        const made = await createForkWorktree(r.runner, r.cwd, fork.fileRef, dest)
        return made.ok ? { ok: true, fork, worktreePath: made.path } : { ok: true, fork, worktreeError: made.error }
      } catch (e) {
        return { ok: true, fork, worktreeError: e instanceof Error ? e.message : String(e) }
      }
    },
  }
}

export type CheckpointService = ReturnType<typeof createCheckpointService>
