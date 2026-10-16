// Per-turn git checkpoints living on hidden refs under refs/vontra/checkpoints/*.
// Each turn's full worktree is captured into a tree+commit through an ISOLATED
// GIT_INDEX_FILE so the user's real index, HEAD and branch are never touched.
// This powers undo-a-turn (restore) and a turn N -> M diff in the files-changed UI.
//
// Backend-agnostic: it drives git through the same GitRunner injection used by
// workspace-git.ts, so the same logic works for local and SSH workspaces and is
// fully testable with a fake runner (no real subprocess / git / network).

import { randomUUID } from 'node:crypto'
import path from 'node:path'

// Reuse the exec-capable runner shape from workspace-git.ts (do not import it to
// keep this module standalone; the integrator passes the same object).
export interface GitRunner {
  exec(command: string, options?: { cwd?: string }): Promise<{ code: number | null; stdout: string; stderr: string }>
}

export interface CheckpointResult {
  ok: boolean
  sha?: string
  error?: string
}

export interface RestoreResult {
  ok: boolean
  error?: string
  restoredPaths?: string[]
  preservedPaths?: string[]
}

export interface DiffResult {
  ok: boolean
  diff?: string
  error?: string
}

// Hidden namespace. Never refs/heads/* so the user's branches are untouched.
export const CHECKPOINT_REFS_PREFIX = 'refs/vontra/checkpoints'

// Identity for checkpoint commits. Author/committer are pinned so checkpoints
// never inherit (and never mutate) the user's git config or commit identity.
const CHECKPOINT_AUTHOR_NAME = 'TensorFold Checkpoints'
const CHECKPOINT_AUTHOR_EMAIL = 'checkpoints@example.com'

function shQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

// Pinned author+committer identity, as a shell env prefix. Reused by every
// index/commit op so checkpoints never read or mutate the user's git config.
function identityEnv(): string {
  return (
    `GIT_AUTHOR_NAME=${shQuote(CHECKPOINT_AUTHOR_NAME)} ` +
    `GIT_AUTHOR_EMAIL=${shQuote(CHECKPOINT_AUTHOR_EMAIL)} ` +
    `GIT_COMMITTER_NAME=${shQuote(CHECKPOINT_AUTHOR_NAME)} ` +
    `GIT_COMMITTER_EMAIL=${shQuote(CHECKPOINT_AUTHOR_EMAIL)}`
  )
}

// A turn key must be ref-safe: encode anything that git refs disallow. Slashes in
// threadId would create nested ref dirs, so they are encoded too.
function encodeSegment(value: string): string {
  return encodeURIComponent(value)
}

/** Hidden ref path for a thread's turn checkpoint. Never refs/heads/* (user branches). */
export function checkpointRef(threadId: string, turn: number): string {
  return `${CHECKPOINT_REFS_PREFIX}/${encodeSegment(threadId)}/turn/${turn}`
}

function gitError(r: { stdout: string; stderr: string }, fallback: string): string {
  return (r.stderr || r.stdout || fallback).trim().slice(0, 300)
}

/** True only when cwd is inside a git work tree. Gates every checkpoint op. */
export async function isGitRepository(runner: GitRunner, cwd: string): Promise<boolean> {
  const r = await runner.exec('git rev-parse --is-inside-work-tree', { cwd })
  return r.code === 0 && r.stdout.trim() === 'true'
}

// Does HEAD point at a real commit? A fresh repo with no commits has no HEAD tree
// to read, so we skip read-tree in that case.
async function hasHeadCommit(runner: GitRunner, cwd: string): Promise<boolean> {
  const r = await runner.exec('git rev-parse --verify --quiet HEAD', { cwd })
  return r.code === 0 && r.stdout.trim().length > 0
}

// Locate the shared git dir so the temp index sits next to the repo's real index
// (same filesystem) rather than in the worktree, where it could be staged/cleaned.
async function resolveGitCommonDir(runner: GitRunner, cwd: string): Promise<string> {
  const r = await runner.exec('git rev-parse --git-common-dir', { cwd })
  const dir = r.stdout.trim()
  return dir.length > 0 ? dir : '.git'
}

// The workspace cwd may be an arbitrary SUBDIRECTORY of the repo (thread.cwd / SSH
// lastRemoteDir need not be the repo root). All capture/restore/diff ops use the
// cwd-relative `-- .` pathspec and a cwd-relative index, so running them from a
// subdir would snapshot/restore only that subtree (re-rooted), corrupting undo/diff.
// Anchor every op at the worktree top-level instead. Falls back to the given cwd
// when toplevel can't be resolved (e.g. a fake runner / detached state).
async function resolveWorktreeRoot(runner: GitRunner, cwd: string): Promise<string> {
  const r = await runner.exec('git rev-parse --show-toplevel', { cwd })
  const top = r.stdout.trim()
  return r.code === 0 && top.length > 0 ? top : cwd
}

async function resolveWorktreePrefix(runner: GitRunner, cwd: string): Promise<string | null> {
  const r = await runner.exec('git rev-parse --show-prefix', { cwd })
  if (r.code !== 0) return null
  const prefix = r.stdout.trim().replace(/\\/g, '/').replace(/^\/+|\/+$/g, '')
  if (!prefix) return ''
  return prefix.startsWith('../') || prefix.split('/').some((segment) => !segment || segment === '.' || segment === '..') ? null : prefix
}

/**
 * Capture the FULL worktree (tracked + untracked) into a tree+commit on a hidden
 * ref, WITHOUT staging the user's index or moving HEAD. Mechanics:
 *   GIT_INDEX_FILE=<temp>  (isolated index; the real index is never touched)
 *   git read-tree HEAD     (seed temp index from HEAD, when HEAD exists)
 *   git add -A -- .        (stage everything into the TEMP index only)
 *   git write-tree         (-> tree oid)
 *   git commit-tree <tree> (-> commit oid; NOT 'git commit', so HEAD never moves)
 *   git update-ref <ref>   (point only the hidden checkpoint ref at the commit)
 * Returns the commit sha.
 */
export async function createCheckpoint(
  runner: GitRunner,
  cwd: string,
  threadId: string,
  turn: number,
): Promise<CheckpointResult> {
  const ref = checkpointRef(threadId, turn)
  // Anchor capture at the worktree top-level so a subdir cwd still snapshots the whole tree.
  cwd = await resolveWorktreeRoot(runner, cwd)
  const commonDir = await resolveGitCommonDir(runner, cwd)
  // Temp index lives in the git dir, never in the worktree, and is unique per call.
  const tempIndex = `${commonDir}/vontra-checkpoint-index-${randomUUID()}`

  // Every index-touching command carries the isolated GIT_INDEX_FILE plus a pinned
  // identity. The env prefix on the command string is the only way to pass env
  // through a GitRunner whose exec takes (command, { cwd }) and no env option.
  const idEnv = identityEnv()
  const indexEnv = `GIT_INDEX_FILE=${shQuote(tempIndex)}`
  const env = `${indexEnv} ${idEnv}`

  try {
    if (await hasHeadCommit(runner, cwd)) {
      const rt = await runner.exec(`${env} git read-tree HEAD`, { cwd })
      if (rt.code !== 0) return { ok: false, error: gitError(rt, 'read-tree failed') }
    }

    const add = await runner.exec(`${env} git add -A -- .`, { cwd })
    if (add.code !== 0) return { ok: false, error: gitError(add, 'add failed') }

    const wt = await runner.exec(`${env} git write-tree`, { cwd })
    const tree = wt.stdout.trim()
    if (wt.code !== 0 || tree.length === 0) return { ok: false, error: gitError(wt, 'write-tree failed') }

    const message = `vontra checkpoint ref=${ref}`
    const ct = await runner.exec(`${env} git commit-tree ${tree} -m ${shQuote(message)}`, { cwd })
    const commit = ct.stdout.trim()
    if (ct.code !== 0 || commit.length === 0) return { ok: false, error: gitError(ct, 'commit-tree failed') }

    // update-ref needs no temp index, but keeping the env is harmless and uniform.
    const ur = await runner.exec(`git update-ref ${shQuote(ref)} ${commit}`, { cwd })
    if (ur.code !== 0) return { ok: false, error: gitError(ur, 'update-ref failed') }

    return { ok: true, sha: commit }
  } finally {
    // Best-effort: drop the throwaway index. Failure here must not mask success.
    await runner.exec(`rm -f ${shQuote(tempIndex)}`, { cwd }).catch(() => {})
  }
}

/**
 * Snapshot the CURRENT worktree (tracked + untracked) into a tree object WITHOUT
 * staging the user's index or moving HEAD — same isolated-GIT_INDEX_FILE mechanics
 * as createCheckpoint, but it stops at write-tree and never commits or moves a ref.
 * Used to diff a checkpoint against live working-tree state. Returns the tree oid.
 */
async function snapshotWorktreeTree(
  runner: GitRunner,
  cwd: string,
): Promise<{ ok: boolean; tree?: string; error?: string }> {
  // Anchor at the worktree top-level so a subdir cwd snapshots the whole tree (matches createCheckpoint).
  cwd = await resolveWorktreeRoot(runner, cwd)
  const commonDir = await resolveGitCommonDir(runner, cwd)
  const tempIndex = `${commonDir}/vontra-checkpoint-index-${randomUUID()}`
  const env = `GIT_INDEX_FILE=${shQuote(tempIndex)} ${identityEnv()}`
  try {
    if (await hasHeadCommit(runner, cwd)) {
      const rt = await runner.exec(`${env} git read-tree HEAD`, { cwd })
      if (rt.code !== 0) return { ok: false, error: gitError(rt, 'read-tree failed') }
    }
    const add = await runner.exec(`${env} git add -A -- .`, { cwd })
    if (add.code !== 0) return { ok: false, error: gitError(add, 'add failed') }
    const wt = await runner.exec(`${env} git write-tree`, { cwd })
    const tree = wt.stdout.trim()
    if (wt.code !== 0 || tree.length === 0) return { ok: false, error: gitError(wt, 'write-tree failed') }
    return { ok: true, tree }
  } finally {
    await runner.exec(`rm -f ${shQuote(tempIndex)}`, { cwd }).catch(() => {})
  }
}

/**
 * Enumerate a thread's checkpoint turns from its hidden refs, ascending. git
 * for-each-ref lists refs lexically ("1","10","2"), so the turn numbers are
 * re-sorted numerically. Read-only; a thread with no checkpoints yields [].
 */
export async function listCheckpointTurns(runner: GitRunner, cwd: string, threadId: string): Promise<number[]> {
  const prefix = `${CHECKPOINT_REFS_PREFIX}/${encodeSegment(threadId)}/turn`
  const r = await runner.exec(`git for-each-ref --format=${shQuote('%(refname)')} ${shQuote(prefix)}`, { cwd })
  if (r.code !== 0) return []
  const turns: number[] = []
  for (const line of r.stdout.split('\n')) {
    const m = line.trim().match(/\/turn\/(\d+)$/)
    if (m) turns.push(Number(m[1]))
  }
  return turns.sort((a, b) => a - b)
}

/**
 * Diff a checkpoint (turn snapshot) against the CURRENT worktree as a unified
 * patch — "what changed since this turn started". The live side is captured into
 * an ephemeral tree (snapshotWorktreeTree) so the diff includes untracked files
 * yet never touches the real index/HEAD/refs. Read-only.
 */
export async function diffCheckpointToWorktree(runner: GitRunner, cwd: string, ref: string): Promise<DiffResult> {
  const commit = await resolveRefCommit(runner, cwd, ref)
  if (!commit) return { ok: false, error: `checkpoint ref not found: ${ref}` }
  const snap = await snapshotWorktreeTree(runner, cwd)
  if (!snap.ok || !snap.tree) return { ok: false, error: snap.error ?? 'worktree snapshot failed' }
  const r = await runner.exec(
    `git diff --patch --no-color --no-ext-diff --no-textconv ${commit} ${snap.tree}`,
    { cwd },
  )
  if (r.code !== 0) return { ok: false, error: gitError(r, 'diff failed') }
  return { ok: true, diff: r.stdout }
}

// Resolve a checkpoint ref to its commit sha, or null when the ref is missing.
async function resolveRefCommit(runner: GitRunner, cwd: string, ref: string): Promise<string | null> {
  const r = await runner.exec(`git rev-parse --verify --quiet ${shQuote(ref)}^{commit}`, { cwd })
  const sha = r.stdout.trim()
  return r.code === 0 && sha.length > 0 ? sha : null
}

/**
 * Restore the worktree + index to a checkpoint (undo-a-turn). It only rewrites the
 * working tree and the index back to the checkpoint snapshot:
 *   git restore --source <commit> --worktree --staged -- <agent-owned paths>
 *   git reset --quiet -- <agent-owned paths>                  (re-sync index; never moves HEAD)
 * It NEVER runs 'git checkout <branch>' and NEVER moves HEAD, so the user's branch
 * stays put. It intentionally NEVER runs git clean: an untracked file created by the
 * user or another process after the checkpoint is not provably agent-owned.
 */
export async function restoreCheckpoint(runner: GitRunner, cwd: string, ref: string, ownedPaths: string[]): Promise<RestoreResult> {
  const commit = await resolveRefCommit(runner, cwd, ref)
  if (!commit) return { ok: false, error: `checkpoint ref not found: ${ref}` }

  const cwdPrefix = await resolveWorktreePrefix(runner, cwd)
  // Checkpoint trees are rooted at the git top-level, while tool changedPaths are
  // rooted at the thread's cwd (which may be a nested package). Translate through
  // the real repo root before validating pathspecs.
  cwd = await resolveWorktreeRoot(runner, cwd)

  const paths = cwdPrefix == null ? [] : [...new Set(ownedPaths.map((value) => {
    const raw = value.trim().replace(/\\/g, '/')
    if (!raw || raw.includes('\0') || raw.startsWith('/') || /^[A-Za-z]:\//.test(raw)) return ''
    return path.posix.normalize(path.posix.join(cwdPrefix, raw))
  }).filter((value) => {
    // A path must identify one concrete repo-relative leaf. In particular, "."
    // must never become a supposedly-scoped restore of the entire worktree, and
    // git's own metadata is never a checkpoint-owned workspace file.
    if (!value || value === '.' || value.startsWith('/') || value.startsWith('../') || value.includes('\0')) return false
    const segments = value.split('/')
    return segments[0] !== '.git' && !segments.some((segment) => segment === '.' || segment === '..' || segment === '')
  }))]
  if (!paths.length) {
    return { ok: false, error: 'No agent-owned file paths were recorded for this turn, so TensorFold refused a broad destructive restore.' }
  }

  // A path absent from the checkpoint was created after it. Even when the agent
  // recorded that path, another process may now own/replace it; without a content
  // ownership proof, preserve it rather than recreating `git clean` under a new name.
  const restorable: string[] = []
  const preserved: string[] = []
  for (const candidate of paths) {
    // `cat-file -e` also accepts tree objects, which would let a recorded directory
    // path recursively widen the restore. Only a blob (regular file or symlink in a
    // git tree) is a concrete leaf path.
    const probe = await runner.exec(`git cat-file -t ${shQuote(`${commit}:${candidate}`)}`, { cwd })
    if (probe.code === 0 && probe.stdout.trim() === 'blob') restorable.push(candidate)
    else preserved.push(candidate)
  }
  if (!restorable.length) return { ok: true, restoredPaths: [], preservedPaths: preserved }

  const pathspec = restorable.map(shQuote).join(' ')
  const restore = await runner.exec(
    `git --literal-pathspecs restore --source ${commit} --worktree --staged -- ${pathspec}`,
    { cwd },
  )
  if (restore.code !== 0) return { ok: false, error: gitError(restore, 'restore failed') }

  // Re-sync the index to HEAD so the user's staged view reflects reality. Pathspec
  // form ('-- .') resets index entries only and never moves HEAD. Run this
  // unconditionally: `git reset -- .` un-stages cleanly even with zero commits
  // (no HEAD), so gating it on hasHeadCommit() would skip the re-sync in exactly
  // the no-commit-repo case where it is needed, leaving restored files staged.
  const reset = await runner.exec(`git --literal-pathspecs reset --quiet -- ${pathspec}`, { cwd })
  if (reset.code !== 0) return { ok: false, error: gitError(reset, 'reset failed') }

  return { ok: true, restoredPaths: restorable, preservedPaths: preserved }
}

/**
 * Materialize a checkpoint into a NEW detached git worktree at destPath (VST-036), so a
 * forked thread edits its own copy of the fork-point snapshot instead of sharing the
 * source worktree. `git worktree add --detach <dest> <commit>` never moves the source
 * worktree's HEAD or branches, and a hidden checkpoint commit is a legal detach target.
 * The snapshot commit's tree carries the turn's untracked files too, so the new worktree
 * starts clean at exactly the captured state.
 */
export async function createForkWorktree(runner: GitRunner, cwd: string, ref: string, destPath: string): Promise<{ ok: boolean; path?: string; error?: string }> {
  const root = await resolveWorktreeRoot(runner, cwd)
  const rp = await runner.exec(`git rev-parse --verify --quiet ${shQuote(`${ref}^{commit}`)}`, { cwd: root })
  const commit = rp.stdout.trim()
  if (rp.code !== 0 || !commit) return { ok: false, error: `checkpoint ref not found: ${ref}` }
  const add = await runner.exec(`git worktree add --detach ${shQuote(destPath)} ${commit}`, { cwd: root })
  if (add.code !== 0) return { ok: false, error: gitError(add, 'worktree add failed') }
  return { ok: true, path: destPath }
}

/**
 * Diff two checkpoint refs (turn N -> turn M) as a unified patch. Read-only: it
 * compares two hidden commits and touches nothing in the worktree, index or HEAD.
 */
export async function diffCheckpoints(
  runner: GitRunner,
  cwd: string,
  fromRef: string,
  toRef: string,
): Promise<DiffResult> {
  const from = await resolveRefCommit(runner, cwd, fromRef)
  if (!from) return { ok: false, error: `checkpoint ref not found: ${fromRef}` }
  const to = await resolveRefCommit(runner, cwd, toRef)
  if (!to) return { ok: false, error: `checkpoint ref not found: ${toRef}` }

  const r = await runner.exec(
    `git diff --patch --no-color --no-ext-diff --no-textconv ${from} ${to}`,
    { cwd },
  )
  if (r.code !== 0) return { ok: false, error: gitError(r, 'diff failed') }
  return { ok: true, diff: r.stdout }
}

/** Best-effort delete of hidden checkpoint refs (e.g. when a thread is pruned). */
export async function deleteCheckpoints(runner: GitRunner, cwd: string, refs: string[]): Promise<void> {
  for (const ref of refs) {
    await runner.exec(`git update-ref -d ${shQuote(ref)}`, { cwd }).catch(() => {})
  }
}
