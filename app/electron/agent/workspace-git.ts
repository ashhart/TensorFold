// Backend-agnostic git surface. Runs git through any exec-capable runner — a
// LocalBackend or an SshBackend — so the same parsing powers local and remote
// workspaces. Pure parsing helpers are exported for unit testing without a repo.

import type { GitStatus, GitDiffFile, GitFileChange } from './types'
import os from 'node:os'

export interface GitRunner {
  exec(command: string, options?: { cwd?: string }): Promise<{ code: number | null; stdout: string; stderr: string }>
}

function shQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

export async function gitBranch(runner: GitRunner, cwd: string): Promise<string | null> {
  const r = await runner.exec('git rev-parse --abbrev-ref HEAD', { cwd })
  const branch = r.stdout.trim()
  return r.code === 0 && branch && branch !== 'HEAD' ? branch : null
}

function mapStatusCode(xy: string): GitFileChange['status'] {
  if (xy === '??') return '?'
  const codes = xy.replace(/\s/g, '')
  if (codes.includes('R')) return 'R'
  if (codes.includes('A')) return 'A'
  if (codes.includes('D')) return 'D'
  return 'M'
}

/** Parse `git status --porcelain=v1` into path + change-kind, resolving renames. */
export function parsePorcelain(stdout: string): Array<{ path: string; status: GitFileChange['status'] }> {
  const out: Array<{ path: string; status: GitFileChange['status'] }> = []
  for (const line of stdout.split('\n')) {
    if (line.length < 4) continue
    const xy = line.slice(0, 2)
    let path = line.slice(3)
    const arrow = path.indexOf(' -> ')
    if (arrow >= 0) path = path.slice(arrow + 4) // renames: keep the new path
    out.push({ path: path.replace(/^"|"$/g, ''), status: mapStatusCode(xy) })
  }
  return out
}

/**
 * Resolve a `git diff --numstat` path field to the new path. Renames are emitted either as
 * `old.txt => new.txt` or with a brace segment `src/{old => new}/f.ts`; `parsePorcelain` keeps the
 * new path, so numstat must too or rename counts are dropped (lookup miss → 0/0).
 */
function numstatNewPath(field: string): string {
  if (field.includes('{') && field.includes('=>')) {
    return field.replace(/\{[^{}]*=>\s*([^{}]*)\}/g, '$1').replace(/\/\//g, '/').replace(/\/$/, '')
  }
  const arrow = field.indexOf(' => ')
  if (arrow >= 0) return field.slice(arrow + 4).trim()
  return field
}

/** Parse `git diff --numstat` into per-path add/delete counts. */
export function parseNumstat(stdout: string): Map<string, { additions: number; deletions: number }> {
  const map = new Map<string, { additions: number; deletions: number }>()
  for (const line of stdout.split('\n')) {
    const parts = line.split('\t')
    if (parts.length < 3) continue
    const [a, d, ...rest] = parts
    map.set(numstatNewPath(rest.join('\t')), {
      additions: a === '-' ? 0 : Number(a) || 0,
      deletions: d === '-' ? 0 : Number(d) || 0,
    })
  }
  return map
}

export async function gitStatus(runner: GitRunner, cwd: string): Promise<GitStatus> {
  const branch = await gitBranch(runner, cwd)
  const [porc, nums] = await Promise.all([
    runner.exec('git status --porcelain=v1', { cwd }),
    runner.exec('git diff HEAD --numstat', { cwd }),
  ])
  const numMap = parseNumstat(nums.stdout)
  const files: GitFileChange[] = parsePorcelain(porc.stdout).map((f) => {
    const n = numMap.get(f.path) ?? { additions: 0, deletions: 0 }
    return { path: f.path, status: f.status, additions: n.additions, deletions: n.deletions }
  })
  return {
    branch,
    files,
    additions: files.reduce((s, f) => s + f.additions, 0),
    deletions: files.reduce((s, f) => s + f.deletions, 0),
  }
}

export interface WorktreeEntry {
  path: string
  head: string | null
  branch: string | null
  detached: boolean
  bare: boolean
}

/** Parse `git worktree list --porcelain` (blank-line-separated blocks). */
export function parseWorktreeList(stdout: string): WorktreeEntry[] {
  const out: WorktreeEntry[] = []
  let cur: WorktreeEntry | null = null
  for (const raw of stdout.split('\n')) {
    const line = raw.trimEnd()
    if (line.startsWith('worktree ')) {
      if (cur) out.push(cur)
      cur = { path: line.slice(9).trim(), head: null, branch: null, detached: false, bare: false }
    } else if (!cur) {
      continue
    } else if (line.startsWith('HEAD ')) {
      cur.head = line.slice(5).trim() || null
    } else if (line.startsWith('branch ')) {
      cur.branch = line.slice(7).trim().replace(/^refs\/heads\//, '') || null
    } else if (line === 'detached') {
      cur.detached = true
    } else if (line === 'bare') {
      cur.bare = true
    } else if (line === '') {
      out.push(cur)
      cur = null
    }
  }
  if (cur) out.push(cur)
  return out
}

/** Local branches, newest-activity first via for-each-ref's default ordering. */
export async function listBranches(runner: GitRunner, cwd: string): Promise<string[]> {
  const r = await runner.exec("git for-each-ref --format='%(refname:short)' refs/heads", { cwd })
  if (r.code !== 0) return []
  return r.stdout.split('\n').map((s) => s.trim().replace(/^'|'$/g, '')).filter(Boolean)
}

export async function listWorktrees(runner: GitRunner, cwd: string): Promise<WorktreeEntry[]> {
  const r = await runner.exec('git worktree list --porcelain', { cwd })
  if (r.code !== 0) return []
  return parseWorktreeList(r.stdout)
}

function gitError(r: { stdout: string; stderr: string }, fallback: string): string {
  return (r.stderr || r.stdout || fallback).trim().slice(0, 300)
}

export async function checkoutBranch(runner: GitRunner, cwd: string, branch: string): Promise<{ ok: boolean; error?: string }> {
  const r = await runner.exec(`git checkout ${shQuote(branch)}`, { cwd })
  return r.code === 0 ? { ok: true } : { ok: false, error: gitError(r, 'checkout failed') }
}

export async function createBranchInPlace(runner: GitRunner, cwd: string, branch: string): Promise<{ ok: boolean; error?: string }> {
  const r = await runner.exec(`git checkout -b ${shQuote(branch)}`, { cwd })
  return r.code === 0 ? { ok: true } : { ok: false, error: gitError(r, 'branch create failed') }
}

/**
 * Commit staged changes without executing repository/user Git hooks. `--no-verify` covers the
 * pre-commit and commit-msg gates; the command-scoped core.hooksPath override also suppresses
 * post-commit and any globally configured hook directory. The override is per invocation and
 * never mutates repository config.
 */
export async function commitWithoutHooks(
  runner: GitRunner,
  cwd: string,
  message: string,
  options: { allowEmpty?: boolean; userName?: string; userEmail?: string } = {},
): Promise<{ ok: boolean; error?: string }> {
  const config = [
    `-c core.hooksPath=${shQuote(os.devNull)}`,
    ...(options.userName ? [`-c user.name=${shQuote(options.userName)}`] : []),
    ...(options.userEmail ? [`-c user.email=${shQuote(options.userEmail)}`] : []),
  ].join(' ')
  const command = `git ${config} commit --no-verify${options.allowEmpty ? ' --allow-empty' : ''} -m ${shQuote(message)}`
  const r = await runner.exec(command, { cwd })
  return r.code === 0 ? { ok: true } : { ok: false, error: gitError(r, 'commit failed') }
}

/** Create a worktree at `worktreePath` for `branch` (reusing the room-worktree command shape). */
export async function addWorktree(runner: GitRunner, repoTop: string, branch: string, worktreePath: string): Promise<{ ok: boolean; path: string; error?: string }> {
  const exists = (await runner.exec(`git branch --list ${shQuote(branch)}`, { cwd: repoTop })).stdout.trim().length > 0
  const cmd = exists
    ? `git -C ${shQuote(repoTop)} worktree add ${shQuote(worktreePath)} ${shQuote(branch)}`
    : `git -C ${shQuote(repoTop)} worktree add -b ${shQuote(branch)} ${shQuote(worktreePath)} HEAD`
  const r = await runner.exec(cmd, { cwd: repoTop })
  return r.code === 0 ? { ok: true, path: worktreePath } : { ok: false, path: worktreePath, error: gitError(r, 'worktree add failed') }
}

export async function gitDiff(runner: GitRunner, cwd: string, paths?: string[]): Promise<GitDiffFile[]> {
  const status = await gitStatus(runner, cwd)
  const targets = paths && paths.length ? status.files.filter((f) => paths.includes(f.path)) : status.files
  const out: GitDiffFile[] = []
  for (const f of targets) {
    const q = shQuote(f.path)
    const cmd = f.status === '?'
      ? `git diff --no-index --no-color -- /dev/null ${q}`
      : `git diff HEAD --no-color -- ${q}`
    const r = await runner.exec(cmd, { cwd })
    out.push({ ...f, diff: r.stdout })
  }
  return out
}
