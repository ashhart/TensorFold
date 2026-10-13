import { dialog } from 'electron'
import path from 'node:path'
import { promises as fs } from 'node:fs'
import { blockedCommandReason } from '../../agent/command-policy'
import { sanitizeCommitMessage, sanitizeGitHygieneFindings, scanGitHygiene } from '../../agent/git-hygiene'
import { buildInstructionGraph } from '../../agent/instruction-graph'
import { runSandboxedCommand } from '../../agent/sandbox'
import { hostGitEnvForWorkspace } from '../../agent/host-git-env'
import { auditPreview } from '../../agent/security'
import type { GitCommitResult, GitDiffFile, GitFileChange, GitHygieneFinding, GitStatus, ResolvedInstructionGraph, Room, Thread, WorkspaceLocation } from '../../agent/types'
import type { TurnFork } from '../../agent/turn-checkpoints'
import type { ConversationState } from '../../agent/conversation-checkpoints'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { handle } from './ipc'
import { initGitRepository, inspectGitRepository, qq, resolveWorkspaceCwd, sh, shStrict, validateRelativeGitPaths } from './workspace'
import { createLocalWorktree, resolveWorkspaceBranch, resolveWorkspaceGitDiff, resolveWorkspaceGitStatus, type WorkspaceGitDeps } from './workspace-git-runtime'
import { gitBranch, listBranches, listWorktrees, checkoutBranch, createBranchInPlace, commitWithoutHooks, type GitRunner } from '../../agent/workspace-git'
import { safeSegment, vontraPath } from '../../store'
import { exec as cpExec } from 'node:child_process'
import type { CheckpointService, CheckpointTarget } from './checkpoint-runtime'
import { requireNativeConfirmation, type ConfirmSensitiveAction } from './sensitive-confirmation'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, sshWorkspaceKey, type WorkspaceMutationGuard } from './workspace-mutation-guard'
import type { WorkspaceBackend } from '../../agent/workspace-backend'

// A local git runner that captures the exit code (the workspace `sh` helper drops it).
const localGitRunner: GitRunner = {
  exec: async (command, opts) => {
    const env = await hostGitEnvForWorkspace(opts?.cwd)
    return new Promise((resolve) => {
      cpExec(command, { cwd: opts?.cwd, timeout: 25_000, maxBuffer: 16 * 1024 * 1024, env }, (err, stdout, stderr) => {
        const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
        resolve({ code, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
      })
    })
  },
}

function sanitizeBranchName(raw: string): string {
  return (raw || '').trim().toLowerCase()
    .replace(/[^a-z0-9._/-]+/g, '-')
    .replace(/\/{2,}/g, '/')
    .replace(/^[-/.]+|[-/.]+$/g, '')
    .slice(0, 80) || 'feature'
}

// NUL-delimited (`-z`) parsers. Using `-z` avoids git's C-quoting of paths that
// contain non-ASCII or control characters (e.g. `"caf\303\251.txt"`), which the
// older newline parsers consumed verbatim — quotes, octal escapes and all — so the
// path never round-tripped back to a real file. Paths in `-z` output are raw bytes.

/** Parse `git status --porcelain=v1 -z`: NUL-separated entries; renames/copies put the
 *  NEW path in the entry record and the OLD path in the next NUL field (consumed & skipped). */
export function parsePorcelainZ(raw: string): Array<{ path: string; status: GitFileChange['status'] }> {
  const tokens = raw.split('\0')
  const out: Array<{ path: string; status: GitFileChange['status'] }> = []
  for (let i = 0; i < tokens.length; i++) {
    const entry = tokens[i]
    if (entry.length < 4) continue // empty trailing token / malformed
    const xy = entry.slice(0, 2)
    const p = entry.slice(3) // skip "XY " prefix; -z keeps the path raw (new path on renames)
    let status: GitFileChange['status'] = 'M'
    if (xy.includes('?')) status = '?'
    else if (xy.includes('A')) status = 'A'
    else if (xy.includes('D')) status = 'D'
    else if (xy.includes('R') || xy.includes('C')) { status = 'R'; i++ /* skip the trailing OLD path */ }
    out.push({ path: p, status })
  }
  return out
}

/** Parse `git diff --no-ext-diff --no-textconv --numstat -z`: NUL-separated; renames emit an empty path field
 *  followed by OLD\0NEW — keyed on the NEW path to match `parsePorcelainZ`. */
export function parseGitNumstat(raw: string): Map<string, { a: number; d: number }> {
  const nums = new Map<string, { a: number; d: number }>()
  const tokens = raw.split('\0')
  for (let i = 0; i < tokens.length; i++) {
    const m = tokens[i].match(/^(\d+|-)\t(\d+|-)\t(.*)$/)
    if (!m) continue
    const counts = { a: m[1] === '-' ? 0 : +m[1], d: m[2] === '-' ? 0 : +m[2] }
    let key = m[3]
    if (key === '') { i += 2; key = tokens[i] ?? '' } // rename: skip OLD, key on NEW
    if (key) nums.set(key, counts)
  }
  return nums
}

async function synthesizeUntrackedDiff(workspace: string, rel: string): Promise<string> {
  const full = path.join(workspace, rel)
  const stat = await fs.stat(full).catch(() => null)
  if (!stat?.isFile()) return `diff --git a/${rel} b/${rel}\nnew file mode 100644\n--- /dev/null\n+++ b/${rel}\n@@\n[untracked file is not readable as a regular file]\n`
  if (stat.size > 400_000) return `diff --git a/${rel} b/${rel}\nnew file mode 100644\n--- /dev/null\n+++ b/${rel}\n@@\n[untracked file is ${stat.size} bytes; preview skipped]\n`
  const content = await fs.readFile(full, 'utf8').catch(() => null)
  if (content == null) return `diff --git a/${rel} b/${rel}\nnew file mode 100644\n--- /dev/null\n+++ b/${rel}\n@@\n[binary or non-UTF-8 file]\n`
  const lines = content.split('\n')
  return [
    `diff --git a/${rel} b/${rel}`,
    'new file mode 100644',
    '--- /dev/null',
    `+++ b/${rel}`,
    `@@ -0,0 +1,${lines.length} @@`,
    ...lines.map((line) => `+${line}`),
  ].join('\n')
}

async function untrackedTextStats(workspace: string, rel: string): Promise<{ a: number; d: number }> {
  const full = path.join(workspace, rel)
  const stat = await fs.stat(full).catch(() => null)
  if (!stat?.isFile() || stat.size > 400_000) return { a: 0, d: 0 }
  const content = await fs.readFile(full, 'utf8').catch(() => null)
  if (content == null || content.length === 0) return { a: 0, d: 0 }
  return { a: content.replace(/\r?\n$/, '').split(/\r?\n/).length, d: 0 }
}

// --- local-path git (host machine). Routed through workspace-git-runtime, which
//     swaps these for the SSH backend when the workspace is a connected remote. ---

async function localGitBranch(cwd: string): Promise<string | null> {
  const workspace = await resolveWorkspaceCwd(cwd)
  const branch = (await sh('git rev-parse --abbrev-ref HEAD', workspace, 3000)).trim()
  return branch && branch !== 'HEAD' ? branch : null
}

async function localGitStatus(cwd: string): Promise<GitStatus> {
  const workspace = await resolveWorkspaceCwd(cwd)
  const branchRaw = (await sh('git rev-parse --abbrev-ref HEAD', workspace, 3000)).trim()
  const branch = branchRaw && branchRaw !== 'HEAD' ? branchRaw : null
  const numstat = await sh('git diff --no-ext-diff --no-textconv --numstat -z HEAD', workspace)
  const statusRaw = await sh('git status --porcelain=v1 -z', workspace)
  const nums = parseGitNumstat(numstat)
  const files: GitFileChange[] = []
  for (const { path: p, status } of parsePorcelainZ(statusRaw)) {
    const n = status === '?' ? await untrackedTextStats(workspace, p) : nums.get(p) || { a: 0, d: 0 }
    files.push({ path: p, status, additions: n.a, deletions: n.d })
  }
  return { branch, files, additions: files.reduce((s, f) => s + f.additions, 0), deletions: files.reduce((s, f) => s + f.deletions, 0) }
}

async function localGitDiff(cwd: string, paths?: string[]): Promise<GitDiffFile[]> {
  const workspace = await resolveWorkspaceCwd(cwd)
  const statusRaw = await sh('git status --porcelain=v1 -z', workspace)
  const statusByPath = new Map<string, GitDiffFile['status']>()
  for (const { path: p, status } of parsePorcelainZ(statusRaw)) {
    statusByPath.set(p, status)
  }
  const safePaths = validateRelativeGitPaths(paths?.length ? paths : [...statusByPath.keys()])
  if (!safePaths.length) return []
  const list = safePaths.map(qq).join(' ')
  const nums = parseGitNumstat(await sh(`git diff --no-ext-diff --no-textconv --numstat -z HEAD -- ${list}`, workspace))
  const out: GitDiffFile[] = []
  for (const rel of safePaths) {
    const status = statusByPath.get(rel) ?? 'M'
    const n = status === '?' ? await untrackedTextStats(workspace, rel) : nums.get(rel) || { a: 0, d: 0 }
    const diff = status === '?'
      ? await synthesizeUntrackedDiff(workspace, rel)
      : await sh(`git diff --no-color --find-renames --no-ext-diff --no-textconv HEAD -- ${qq(rel)}`, workspace, 8000)
    out.push({ path: rel, status, additions: n.a, deletions: n.d, diff: diff || `No textual diff for ${rel}.` })
  }
  return out
}

let resolveGitBackend: WorkspaceGitDeps['resolveBackend'] = () => undefined
const gitDeps = (): WorkspaceGitDeps => ({
  resolveBackend: resolveGitBackend,
  localBranch: localGitBranch,
  localStatus: localGitStatus,
  localDiff: localGitDiff,
})

/**
 * Resolve checkpoint mutation scope exclusively from host-owned thread state.
 * Renderer cwd/workspace/kind values are presentation hints and must never select
 * which filesystem an undo/revert/fork mutates. Snapshot the timeline so a run
 * finishing concurrently cannot widen this operation after authorization.
 */
export function hostCheckpointTarget(threads: readonly Thread[], threadId: string, projects = store.projects): CheckpointTarget | null {
  const thread = threads.find((candidate) => candidate.id === threadId)
  if (!thread || thread.archivedAt) return null
  const project = thread.projectId ? projects.find((candidate) => candidate.id === thread.projectId) ?? null : null
  if (thread.projectId && (!project || project.archivedAt)) return null
  const workspace = thread.workspace ?? project?.workspace ?? null
  return {
    cwd: thread.cwd || project?.cwd || workspace?.path || null,
    workspace: workspace ? { ...workspace } : null,
    kind: thread.kind,
    timeline: thread.timeline.map((item) => ({
      ...item,
      changedPaths: item.changedPaths ? [...item.changedPaths] : undefined,
    })),
  }
}

export function sameCheckpointTarget(a: CheckpointTarget, b: CheckpointTarget): boolean {
  return (a.cwd ?? null) === (b.cwd ?? null)
    && (a.kind ?? null) === (b.kind ?? null)
    && (a.workspace?.kind ?? null) === (b.workspace?.kind ?? null)
    && (a.workspace?.path ?? null) === (b.workspace?.path ?? null)
    && (a.workspace?.kind === 'ssh' ? a.workspace.hostId : null) === (b.workspace?.kind === 'ssh' ? b.workspace.hostId : null)
}

export function requireThreadStopped(isThreadRunning: ((threadId: string) => boolean) | undefined, threadId: string): void {
  if (!isThreadRunning) throw new Error('Cannot restore workspace state because host run-state checks are unavailable.')
  if (isThreadRunning(threadId)) throw new Error('Stop the running thread before restoring workspace state.')
}

/** Reject full-file restore while any running LOCAL thread owns the same canonical cwd. */
export async function requireNoRunningLocalWriterAtWorkspace(
  threads: readonly Thread[],
  rooms: readonly Room[],
  cwd: string,
  isThreadRunning: ((threadId: string) => boolean) | undefined,
  isRoomRunning: ((roomId: string) => boolean) | undefined,
): Promise<void> {
  if (!isThreadRunning || !isRoomRunning) throw new Error('Cannot restore workspace files because host run-state checks are unavailable.')
  const target = await canonicalLocalWorkspaceKey(cwd)
  if (!target) throw new Error('Cannot determine the workspace identity for file restoration.')
  await requireNoRunningWriterAtWorkspaceKey(threads, rooms, target, isThreadRunning, isRoomRunning)
}

async function threadWorkspaceKey(thread: Thread): Promise<string | null> {
  if (thread.kind === 'chat') return null
  const project = thread.projectId ? store.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null
  const workspace = thread.workspace ?? project?.workspace ?? null
  if (workspace?.kind === 'ssh') return sshWorkspaceKey(workspace.hostId, workspace.path)
  return canonicalLocalWorkspaceKey(workspace?.kind === 'local' ? workspace.path : (thread.cwd || project?.cwd))
}

async function roomWorkspaceKeys(room: Room): Promise<string[]> {
  const project = store.projects.find((candidate) => candidate.id === room.projectId) ?? null
  const workspace = room.workspace ?? project?.workspace ?? null
  const values = [workspace?.path, room.rootCwd, room.workspaceCwd, room.scratchpadCwd, project?.cwd]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
  const keys = workspace?.kind === 'ssh'
    ? values.map((value) => sshWorkspaceKey(workspace.hostId, value))
    : await Promise.all(values.map((value) => canonicalLocalWorkspaceKey(value)))
  return [...new Set(keys.filter((key): key is string => !!key))]
}

export async function checkpointWorkspaceKey(target: CheckpointTarget): Promise<string | null> {
  if (target.kind === 'chat') return null
  if (target.workspace?.kind === 'ssh') return sshWorkspaceKey(target.workspace.hostId, target.workspace.path)
  return canonicalLocalWorkspaceKey(target.workspace?.kind === 'local' ? target.workspace.path : target.cwd)
}

export function checkpointInitialWorkspaceKey(target: CheckpointTarget): string | null {
  if (target.kind === 'chat') return null
  if (target.workspace?.kind === 'ssh') return sshWorkspaceKey(target.workspace.hostId, target.workspace.path)
  return lexicalLocalWorkspaceKey(target.workspace?.kind === 'local' ? target.workspace.path : target.cwd)
}

export async function requireNoRunningWriterAtWorkspaceKey(
  threads: readonly Thread[],
  rooms: readonly Room[],
  target: string,
  isThreadRunning: ((threadId: string) => boolean) | undefined,
  isRoomRunning: ((roomId: string) => boolean) | undefined,
): Promise<void> {
  if (!isThreadRunning || !isRoomRunning) throw new Error('Cannot restore workspace state because host run-state checks are unavailable.')
  for (const thread of threads) {
    if (!isThreadRunning(thread.id)) continue
    if (await threadWorkspaceKey(thread) === target) throw new Error('Stop active threads in this workspace before restoring state.')
  }
  for (const room of rooms) {
    if (!isRoomRunning(room.id)) continue
    if ((await roomWorkspaceKeys(room)).includes(target)) throw new Error('Stop active rooms in this workspace before restoring state.')
  }
}

export function registerWorkspaceIpc(deps?: {
  resolveBackend: WorkspaceGitDeps['resolveBackend']
  checkpoints?: CheckpointService
  // Destructive file/conversation restoration must be approved by Electron's native
  // main-process dialog. Omission fails closed in requireNativeConfirmation.
  confirmSensitiveAction?: ConfirmSensitiveAction
  // Host-owned run state. Destructive restores check before AND after confirmation so a
  // renderer cannot race a new run through a dialog that was opened while idle.
  isThreadRunning?: (threadId: string) => boolean
  isRoomRunning?: (roomId: string) => boolean
  workspaceMutationGuard?: Pick<WorkspaceMutationGuard, 'acquire' | 'reserveStart'>
  // Applies a restored conversation to the live Thread (mutate store + persist + emit).
  // Owned by main-app (which has the store); absent → revert restores files only (VST-036).
  applyThreadRevert?: (threadId: string, conversation: ConversationState) => void
  // Creates the forked Thread from a TurnFork (VST-036). Owned by main-app: the fork's
  // conversation never round-trips through the renderer — the host seeds the new thread
  // itself (same rebuild-host-side rule as session import). Returns null when the source
  // thread no longer exists. `overrides.cwd` homes the fork in its own worktree;
  // `overrides.notice` appends an explanatory line to the fork notice.
  createForkThread?: (sourceThreadId: string, fork: TurnFork, overrides?: { cwd?: string; notice?: string }) => Thread | null
}): void {
  if (deps?.resolveBackend) resolveGitBackend = deps.resolveBackend

  const backendFor = (workspace?: WorkspaceLocation | null) => {
    const backend = resolveGitBackend(workspace)
    if (workspace?.kind === 'ssh' && !backend) throw new Error('Connect the SSH workspace before using this operation.')
    return backend
  }
  const localOnly = (workspace: WorkspaceLocation | null | undefined, operation: string) => {
    if (workspace?.kind === 'ssh') throw new Error(`${operation} is not supported for SSH workspaces yet.`)
  }
  const withWorkspaceStartReservation = async <T>(
    cwd: string,
    workspace: WorkspaceLocation | null | undefined,
    fn: (target: string, backend: ReturnType<typeof resolveGitBackend>) => Promise<T>,
  ): Promise<T> => {
    if (!deps?.workspaceMutationGuard) throw new Error('Cannot mutate workspace state because the mutation guard is unavailable.')
    const backend = backendFor(workspace)
    const initialKey = workspace?.kind === 'ssh'
      ? sshWorkspaceKey(workspace.hostId, workspace.path)
      : lexicalLocalWorkspaceKey(workspace?.kind === 'local' ? workspace.path : cwd)
    const lease = deps.workspaceMutationGuard.reserveStart({ workspaceKeys: [initialKey] })
    try {
      const target = workspace?.kind === 'ssh'
        ? String(workspace.path || backend?.rootCwd || '').trim()
        : await resolveWorkspaceCwd(workspace?.kind === 'local' ? workspace.path : cwd)
      if (!target) throw new Error('Choose a workspace folder first.')
      if (workspace?.kind !== 'ssh') lease.addWorkspaceKeys?.([await canonicalLocalWorkspaceKey(target)])
      return await fn(target, backend)
    } finally {
      lease.release()
    }
  }

  const checkpoints = deps?.checkpoints
  if (checkpoints) {
    const target = (threadId: unknown): CheckpointTarget | null =>
      typeof threadId === 'string' && threadId.length > 0 ? hostCheckpointTarget(store.threads, threadId) : null

    handle('checkpoint:list', async (_e, arg: { threadId: string }) => {
      const resolved = target(arg?.threadId)
      return resolved ? checkpoints.listTurns(arg.threadId, resolved) : []
    })

    handle('checkpoint:undoLast', async (_e, arg: { threadId: string }) => {
      // Reverting the worktree is an operator-grade mutation (gated like revert/checkout).
      requireSecurityRole('operator', 'undo the last agent turn')
      const resolved = target(arg?.threadId)
      if (!resolved) {
        await auditEvent('workspace.checkpoint_undo', { threadId: auditPreview(arg?.threadId ?? ''), turn: null, ok: false, reason: 'thread not found' })
        return { ok: false, error: 'thread not found' }
      }
      const workspaceKey = await checkpointWorkspaceKey(resolved)
      requireThreadStopped(deps?.isThreadRunning, arg.threadId)
      if (workspaceKey) await requireNoRunningWriterAtWorkspaceKey(store.threads, store.rooms, workspaceKey, deps?.isThreadRunning, deps?.isRoomRunning)
      await requireNativeConfirmation(deps?.confirmSensitiveAction, {
        title: 'Undo the last agent turn?',
        message: 'Restore the checkpointed workspace files from before the last agent turn?',
        detail: 'Only paths recorded as changed by that turn are restored. Newer unrelated files are preserved and the conversation is not changed.',
        confirmLabel: 'Restore files',
      })
      if (!deps?.workspaceMutationGuard) throw new Error('Cannot restore workspace state because the mutation guard is unavailable.')
      const lease = deps.workspaceMutationGuard.acquire({ threadId: arg.threadId, workspaceKey })
      let res: Awaited<ReturnType<CheckpointService['undoLast']>>
      try {
        const current = target(arg.threadId)
        if (!current || !sameCheckpointTarget(resolved, current)) throw new Error('The thread workspace changed while checkpoint restoration was awaiting confirmation.')
        const currentWorkspaceKey = await checkpointWorkspaceKey(current)
        if (currentWorkspaceKey !== workspaceKey) throw new Error('The thread workspace identity changed while checkpoint restoration was awaiting confirmation.')
        requireThreadStopped(deps?.isThreadRunning, arg.threadId)
        if (currentWorkspaceKey) await requireNoRunningWriterAtWorkspaceKey(store.threads, store.rooms, currentWorkspaceKey, deps?.isThreadRunning, deps?.isRoomRunning)
        res = await checkpoints.undoLast(arg.threadId, current)
      } finally {
        lease.release()
      }
      await auditEvent('workspace.checkpoint_undo', { threadId: auditPreview(arg?.threadId ?? ''), turn: res.turn ?? null, ok: res.ok, preservedPaths: res.preservedPaths?.length ?? 0 })
      return res
    })

    handle('checkpoint:diffLast', async (_e, arg: { threadId: string }) => {
      const resolved = target(arg?.threadId)
      return resolved ? checkpoints.diffLast(arg.threadId, resolved) : { ok: false, error: 'thread not found' }
    })

    // ---- unified turn checkpoints (VST-036): files + conversation ----
    handle('checkpoint:turns', async (_e, arg: { threadId: string }) => target(arg?.threadId) ? checkpoints.turnCheckpoints(arg.threadId) : [])

    // Revert to a specific turn: restore files AND conversation together (the gap undoLast
    // left open). Operator-gated + audited.
    handle('checkpoint:revertTurn', async (_e, arg: { threadId: string; turn: number }) => {
      requireSecurityRole('operator', 'revert a thread to a past turn')
      const resolved = target(arg?.threadId)
      if (!resolved) {
        await auditEvent('workspace.checkpoint_revert_turn', { threadId: auditPreview(arg?.threadId ?? ''), turn: arg?.turn, ok: false, reason: 'thread not found' })
        return { ok: false, turn: arg?.turn, error: 'thread not found', warning: null }
      }
      const workspaceKey = await checkpointWorkspaceKey(resolved)
      requireThreadStopped(deps?.isThreadRunning, arg.threadId)
      if (workspaceKey) await requireNoRunningWriterAtWorkspaceKey(store.threads, store.rooms, workspaceKey, deps?.isThreadRunning, deps?.isRoomRunning)
      await requireNativeConfirmation(deps?.confirmSensitiveAction, {
        title: 'Revert this thread?',
        message: `Restore files and conversation to before turn ${arg.turn}?`,
        detail: 'Later conversation is removed. Only checkpoint-owned workspace paths are restored; newer unrelated files are preserved.',
        confirmLabel: 'Revert turn',
      })
      if (!deps?.workspaceMutationGuard) throw new Error('Cannot restore workspace state because the mutation guard is unavailable.')
      const lease = deps.workspaceMutationGuard.acquire({ threadId: arg.threadId, workspaceKey })
      let res: Awaited<ReturnType<CheckpointService['revertTurn']>>
      try {
        const current = target(arg.threadId)
        if (!current || !sameCheckpointTarget(resolved, current)) throw new Error('The thread workspace changed while checkpoint restoration was awaiting confirmation.')
        const currentWorkspaceKey = await checkpointWorkspaceKey(current)
        if (currentWorkspaceKey !== workspaceKey) throw new Error('The thread workspace identity changed while checkpoint restoration was awaiting confirmation.')
        requireThreadStopped(deps?.isThreadRunning, arg.threadId)
        if (currentWorkspaceKey) await requireNoRunningWriterAtWorkspaceKey(store.threads, store.rooms, currentWorkspaceKey, deps?.isThreadRunning, deps?.isRoomRunning)
        res = await checkpoints.revertTurn(arg.threadId, arg.turn, current)
        if (res.ok && res.conversation && deps?.applyThreadRevert) {
          deps.applyThreadRevert(arg.threadId, res.conversation)
        }
      } finally {
        lease.release()
      }
      await auditEvent('workspace.checkpoint_revert_turn', { threadId: auditPreview(arg?.threadId ?? ''), turn: arg?.turn, ok: res.ok, appliedConversation: !!res.conversation, preservedPaths: res.preservedPaths?.length ?? 0 })
      return { ok: res.ok, turn: res.turn, error: res.error ?? null, warning: res.warning ?? null }
    })

    // Fork an alternate branch from a turn. The HOST creates the forked thread (injected
    // from main-app) and seeds its ledger with the fork point — the raw conversation never
    // crosses to the renderer, which only receives the finished Thread. `worktree: true`
    // additionally materializes the fork point into its OWN detached git worktree and homes
    // the fork there; a worktree failure degrades to the shared cwd with the reason spelled
    // out in the fork notice, never silently.
    handle('checkpoint:forkTurn', async (_e, arg: { threadId: string; turn: number; branch?: string; label?: string; worktree?: boolean }) => {
      requireSecurityRole('operator', 'fork a thread from a past turn')
      const resolved = target(arg?.threadId)
      if (!resolved) {
        await auditEvent('workspace.checkpoint_fork_turn', { threadId: auditPreview(arg?.threadId ?? ''), turn: arg?.turn, ok: false, reason: 'thread not found' })
        return { ok: false, error: 'thread not found' }
      }
      if (!deps?.workspaceMutationGuard) throw new Error('Cannot fork checkpoint state because the mutation guard is unavailable.')
      const forkLease = deps.workspaceMutationGuard.reserveStart({ threadId: arg.threadId, workspaceKeys: [checkpointInitialWorkspaceKey(resolved)] })
      try {
      forkLease.addWorkspaceKeys?.([await checkpointWorkspaceKey(resolved)])
      const current = target(arg.threadId)
      if (!current || !sameCheckpointTarget(resolved, current)) throw new Error('The thread workspace changed while the checkpoint fork was starting.')
      const res = await checkpoints.forkTurn(arg.threadId, arg.turn, current, { branch: arg.branch, label: arg.label, worktree: arg.worktree === true })
      let thread: Thread | null = null
      if (res.ok && res.fork && deps?.createForkThread) {
        const overrides = res.worktreePath
          ? { cwd: res.worktreePath, notice: `This fork lives in its own worktree: ${res.worktreePath}` }
          : arg.worktree && res.worktreeError
            ? { notice: `Fork worktree unavailable (${res.worktreeError}) — this fork shares the source folder.` }
            : undefined
        thread = deps.createForkThread(arg.threadId, res.fork, overrides)
        if (thread) checkpoints.adoptFork(thread.id, res.fork)
      }
      await auditEvent('workspace.checkpoint_fork_turn', { threadId: auditPreview(arg.threadId), turn: arg.turn, ok: res.ok, branch: res.fork?.branch ?? null, forkThreadId: thread ? auditPreview(thread.id) : null, worktree: res.worktreePath ? auditPreview(res.worktreePath) : null })
      if (!res.ok || !res.fork) return { ok: false, error: res.error ?? 'fork failed' }
      if (deps?.createForkThread && !thread) return { ok: false, error: 'the source thread no longer exists' }
      return { ok: true, branch: res.fork.branch, fileRef: res.fork.fileRef, thread, worktreePath: res.worktreePath ?? null, worktreeError: res.worktreeError ?? null }
      } finally {
        forkLease.release()
      }
    })
  }

  handle('workspace:branch', async (_e, arg: string | { cwd: string; workspace?: WorkspaceLocation | null }) => {
    const { cwd, workspace } = typeof arg === 'string' ? { cwd: arg, workspace: null } : arg
    return resolveWorkspaceBranch({ cwd, workspace }, gitDeps())
  })

  handle('workspace:gitRepository', async (_e, arg: string | { cwd: string; workspace?: WorkspaceLocation | null }) => {
    const { cwd, workspace } = typeof arg === 'string' ? { cwd: arg, workspace: null } : arg
    localOnly(workspace, 'Git repository inspection')
    return inspectGitRepository(cwd)
  })
  handle('workspace:initGitRepository', async (_e, arg: string | { cwd: string; workspace?: WorkspaceLocation | null }) => {
    const { cwd, workspace } = typeof arg === 'string' ? { cwd: arg, workspace: null } : arg
    localOnly(workspace, 'Git initialization')
    return withWorkspaceStartReservation(cwd, workspace, (target) => initGitRepository(target))
  })
  handle('workspace:gitHygiene', async (_e, { cwd, message = '', workspace }: { cwd: string; message?: string; workspace?: WorkspaceLocation | null }) => {
    localOnly(workspace, 'Git hygiene scanning')
    const target = await resolveWorkspaceCwd(cwd)
    await auditEvent('workspace.git_hygiene_scan', { cwd: auditPreview(target) })
    return scanGitHygiene(target, message)
  })
  handle('workspace:sanitizeGitHygiene', async (_e, { cwd, findings, workspace }: { cwd: string; findings: GitHygieneFinding[]; workspace?: WorkspaceLocation | null }) => {
    requireSecurityRole('operator', 'sanitize git hygiene findings')
    localOnly(workspace, 'Git hygiene sanitization')
    return withWorkspaceStartReservation(cwd, workspace, async (target) => {
      await auditEvent('workspace.git_hygiene_sanitize', { cwd: auditPreview(target), findings: findings.length })
      return sanitizeGitHygieneFindings(target, findings)
    })
  })

  handle('instructions:resolve', async (_e, cwd: string, providerId?: string | null, target?: WorkspaceLocation | null): Promise<ResolvedInstructionGraph> => {
    if (target?.kind === 'ssh') {
      return { rootPath: target.path || cwd, providerId: providerId ?? null, nodes: [], cycles: [], resolvedContent: '', tokenEstimate: 0 }
    }
    const workspace = await resolveWorkspaceCwd(cwd)
    return buildInstructionGraph(workspace, providerId ?? null)
  })

  handle('workspace:files', async (_e, arg: string | { cwd: string; workspace?: WorkspaceLocation | null }) => {
    requireSecurityRole('operator', 'browse workspace files')
    const { cwd, workspace: location } = typeof arg === 'string' ? { cwd: arg, workspace: null } : arg
    const backend = backendFor(location)
    const fileBackend = backend as WorkspaceBackend | undefined
    const workspace = location?.kind === 'ssh' ? String(location.path || backend?.rootCwd || '').trim() : await resolveWorkspaceCwd(cwd)
    if (!workspace) throw new Error('Choose a workspace folder first.')
    const SKIP = new Set(['node_modules', '.git', 'dist', 'dist-electron', '.next', 'build', '.cache', 'vendor', '.venv', '.turbo'])
    const out: string[] = []
    const walk = async (dir: string, depth: number) => {
      if (depth > 8 || out.length >= 3000) return
      let entries: any[] = []
      try { entries = fileBackend ? await fileBackend.readdir(dir) : await fs.readdir(dir, { withFileTypes: true }) } catch { return }
      for (const e of entries) {
        if (out.length >= 3000) break
        if (SKIP.has(e.name)) continue
        if (e.name.startsWith('.')) continue
        const full = fileBackend ? path.posix.join(dir, e.name) : path.join(dir, e.name)
        const isDirectory = typeof e.isDirectory === 'function' ? e.isDirectory() : e.isDirectory
        if (isDirectory) await walk(full, depth + 1)
        else out.push(fileBackend ? path.posix.relative(workspace, full) : path.relative(workspace, full))
      }
    }
    try { await walk(workspace, 0) } catch { /* ignore */ }
    return out.sort()
  })

  handle('workspace:gitStatus', async (_e, arg: string | { cwd: string; workspace?: WorkspaceLocation | null }) => {
    const { cwd, workspace } = typeof arg === 'string' ? { cwd: arg, workspace: null } : arg
    return resolveWorkspaceGitStatus({ cwd, workspace }, gitDeps())
  })

  handle('workspace:gitDiff', async (_e, { cwd, paths, workspace }: { cwd: string; paths?: string[]; workspace?: WorkspaceLocation | null }): Promise<GitDiffFile[]> => {
    return resolveWorkspaceGitDiff({ cwd, paths, workspace }, gitDeps())
  })

  handle('workspace:worktrees', async (_e, arg: string | { cwd: string; workspace?: WorkspaceLocation | null }) => {
    const { cwd, workspace } = typeof arg === 'string' ? { cwd: arg, workspace: null } : arg
    const backend = backendFor(workspace)
    const runner: GitRunner = backend ?? localGitRunner
    const target = workspace?.path?.trim() || (backend ? backend.rootCwd : await resolveWorkspaceCwd(cwd))
    const [branches, worktrees, current] = await Promise.all([
      listBranches(runner, target),
      listWorktrees(runner, target),
      gitBranch(runner, target),
    ])
    return { branches, worktrees, current }
  })

  handle('workspace:checkoutBranch', async (_e, { cwd, workspace, branch }: { cwd: string; workspace?: WorkspaceLocation | null; branch: string }) => {
    requireSecurityRole('operator', 'switch git branch')
    // checkoutBranch shell-quotes the ref but adds no `--` guard, so a ref that begins
    // with `-` is parsed by git as a flag (e.g. `--force`, `--orphan`, `-b<x>`). A forced
    // checkout can discard the working tree, so reject option-shaped refs outright.
    const ref = String(branch ?? '').trim()
    if (!ref || ref.startsWith('-')) throw new Error(`Invalid branch name: ${branch}`)
    return withWorkspaceStartReservation(cwd, workspace, async (target, backend) => {
      const runner: GitRunner = backend ?? localGitRunner
      await auditEvent('workspace.git_checkout', { cwd: auditPreview(target), branch: auditPreview(ref) })
      return checkoutBranch(runner, target, ref)
    })
  })

  handle('workspace:createWorktree', async (_e, { cwd, workspace, name, mode }: { cwd: string; workspace?: WorkspaceLocation | null; name: string; mode: 'worktree' | 'branch' }) => {
    requireSecurityRole('operator', 'create git worktree or branch')
    const branch = sanitizeBranchName(name)
    const backend = backendFor(workspace)
    if (mode === 'branch') {
      return withWorkspaceStartReservation(cwd, workspace, async (target, liveBackend) => {
        const runner: GitRunner = liveBackend ?? localGitRunner
        await auditEvent('workspace.git_branch_create', { cwd: auditPreview(target), branch: auditPreview(branch) })
        return { ...(await createBranchInPlace(runner, target, branch)), branch, mode: 'branch', path: target }
      })
    }
    if (backend) return { ok: false, branch, mode: 'worktree', path: '', error: 'Worktrees on remote SSH hosts are not supported yet — use "Branch here" instead.' }
    return withWorkspaceStartReservation(cwd, workspace, async (local) => {
      const repo = await inspectGitRepository(local)
      if (!repo.exists || !repo.topLevel) return { ok: false, branch, mode: 'worktree' as const, path: '', error: 'This folder is not a Git repository.' }
      if (!repo.hasCommits) return { ok: false, branch, mode: 'worktree' as const, path: '', error: 'Make an initial commit before creating a worktree.' }
      const wtPath = vontraPath('worktrees', safeSegment(path.basename(repo.topLevel)), safeSegment(branch.replace(/\//g, '-')))
      await auditEvent('workspace.git_worktree_create', { repo: auditPreview(repo.topLevel), branch: auditPreview(branch), path: auditPreview(wtPath) })
      return { ...(await createLocalWorktree(localGitRunner, repo.topLevel, branch, wtPath)), branch, mode: 'worktree' as const }
    })
  })

  handle('workspace:revertFiles', async (_e, { cwd, paths, workspace: location }: { cwd: string; paths: string[]; workspace?: WorkspaceLocation | null }) => {
    requireSecurityRole('operator', 'revert workspace files')
    localOnly(location, 'Selected-file restoration')
    const safePaths = validateRelativeGitPaths(paths)
    if (!safePaths.length) return
    const workspace = await resolveWorkspaceCwd(cwd)
    const workspaceKey = await canonicalLocalWorkspaceKey(workspace)
    if (!workspaceKey) throw new Error('Cannot determine the workspace identity for file restoration.')
    await requireNoRunningLocalWriterAtWorkspace(store.threads, store.rooms, workspace, deps?.isThreadRunning, deps?.isRoomRunning)
    await requireNativeConfirmation(deps?.confirmSensitiveAction, {
      title: 'Discard selected workspace changes?',
      message: `Restore ${safePaths.length} selected workspace ${safePaths.length === 1 ? 'path' : 'paths'} to Git state?`,
      detail: 'Tracked changes are discarded and selected untracked files or directories are removed. Path names are omitted from this dialog; review the selected diff before confirming.',
      confirmLabel: 'Discard changes',
    })
    await requireNoRunningLocalWriterAtWorkspace(store.threads, store.rooms, workspace, deps?.isThreadRunning, deps?.isRoomRunning)
    const list = safePaths.map(qq).join(' ')
    await auditEvent('workspace.revert_files', { cwd: auditPreview(workspace), paths: safePaths.slice(0, 50) })
    await requireNoRunningLocalWriterAtWorkspace(store.threads, store.rooms, workspace, deps?.isThreadRunning, deps?.isRoomRunning)
    if (!deps?.workspaceMutationGuard) throw new Error('Cannot restore workspace files because the mutation guard is unavailable.')
    const lease = deps.workspaceMutationGuard.acquire({ workspaceKey })
    try {
      await sh(`git checkout -- ${list}`, workspace, 8000)
      await sh(`git clean -fd -- ${list}`, workspace, 8000)
    } finally {
      lease.release()
    }
  })

  handle('workspace:commit', async (_e, { cwd, message, workspace }: { cwd: string; message: string; workspace?: WorkspaceLocation | null }): Promise<GitCommitResult> => {
    requireSecurityRole('operator', 'commit workspace changes')
    localOnly(workspace, 'Workspace commit')
    return withWorkspaceStartReservation(cwd, workspace, async (target) => {
      const cleanMessage = sanitizeCommitMessage(message)
      const hygiene = await scanGitHygiene(target, cleanMessage)
      await auditEvent('workspace.commit_preflight', { cwd: auditPreview(target), blocked: hygiene.blocked, critical: hygiene.summary.critical, high: hygiene.summary.high, medium: hygiene.summary.medium, low: hygiene.summary.low })
      if (hygiene.blocked) {
        const total = hygiene.summary.critical + hygiene.summary.high + hygiene.summary.medium
        throw new Error(`Git hygiene blocked commit: ${total} blocking finding${total === 1 ? '' : 's'} found. Review and sanitize before committing.`)
      }
      await auditEvent('workspace.commit', { cwd: auditPreview(target), message: auditPreview(cleanMessage) })
      await shStrict('git add -A', target)
      const committed = await commitWithoutHooks(localGitRunner, target, cleanMessage)
      if (!committed.ok) throw new Error(committed.error || 'Git commit failed.')
      return { ok: true, hygiene, message: cleanMessage }
    })
  })

  handle('workspace:exec', async (_e, { cwd, command, workspace }: { cwd: string; command: string; workspace?: WorkspaceLocation | null }) => {
    requireSecurityRole('operator', 'execute workspace commands')
    return withWorkspaceStartReservation(cwd, workspace, async (target, backend) => {
      const blocked = blockedCommandReason(command)
      if (blocked) {
        await auditEvent('workspace.exec_blocked', { cwd: auditPreview(target), command: auditPreview(command), reason: blocked })
        return { output: `Blocked by TensorFold safety policy: ${blocked}.`, code: 126 }
      }
      await auditEvent('workspace.exec', { cwd: auditPreview(target), command: auditPreview(command) })
      const result = backend
        ? await (backend as WorkspaceBackend).exec(command, { cwd: target, timeoutMs: 120_000 })
        : await runSandboxedCommand(command, { cwd: target, timeoutMs: 120_000, maxBuffer: 8 * 1024 * 1024 })
      const output = [result.stdout, result.stderr].filter(Boolean).join('\n').replace(/\n$/, '') || (result.code === 0 ? '' : `Exited with code ${result.code}`)
      return { output, code: result.code ?? 1 }
    })
  })

  handle('dialog:pickDir', async () => {
    if (!store.win) return null
    const r = await dialog.showOpenDialog(store.win, { properties: ['openDirectory', 'createDirectory'] })
    return r.canceled || !r.filePaths[0] ? null : r.filePaths[0]
  })
}
