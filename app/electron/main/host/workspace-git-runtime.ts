// Backend-aware git routing for the workspace panel IPC. When a thread/project's
// workspace is a connected SSH host, git status/diff/branch are computed by running
// git on that backend (so the panel reflects the REMOTE repo); otherwise they fall
// back to the existing local-path implementations. The command execution and
// porcelain/numstat parsing live in agent/workspace-git (shared, backend-agnostic).

import { addWorktree, gitBranch, gitStatus as engineGitStatus, gitDiff as engineGitDiff, type GitRunner } from '../../agent/workspace-git'
import type { GitDiffFile, GitStatus, WorkspaceLocation } from '../../agent/types'
import { realpath } from 'node:fs/promises'

/** Local worktree creation at the host boundary, separate from SSH Git runners. */
export async function createLocalWorktree(runner: GitRunner, repoTop: string, branch: string, target: string): Promise<{ ok: boolean; path: string; error?: string }> {
  const result = await addWorktree(runner, repoTop, branch, target)
  // Git's worktree list resolves directory aliases, including /var on macOS.
  // Return the same identity to clients that verify path and branch together.
  return result.ok ? { ...result, path: await realpath(result.path) } : result
}

/** A git-capable backend: an exec runner plus the remote root cwd to fall back to. */
export interface GitBackend extends GitRunner {
  readonly rootCwd: string
}

export interface WorkspaceGitDeps {
  /** Resolve a connected git backend for a workspace, or undefined for local. */
  resolveBackend: (ws: WorkspaceLocation | null | undefined) => GitBackend | undefined
  /** Existing local-path implementations (preserve current behavior + security checks). */
  localBranch: (cwd: string) => Promise<string | null>
  localStatus: (cwd: string) => Promise<GitStatus>
  localDiff: (cwd: string, paths?: string[]) => Promise<GitDiffFile[]>
}

interface GitArgs {
  cwd: string
  workspace?: WorkspaceLocation | null
}

function requireExplicitSshBackend(args: GitArgs, backend: GitBackend | undefined): GitBackend | undefined {
  if (args.workspace?.kind === 'ssh' && !backend) {
    throw new Error('The SSH workspace is disconnected. Reconnect it before using Git operations.')
  }
  return backend
}

// The remote working directory: the workspace's absolute remote path is canonical,
// then the caller's cwd, then the backend's connected root as a last resort.
function remoteCwd(backend: GitBackend, args: GitArgs): string {
  return args.workspace?.path?.trim() || args.cwd?.trim() || backend.rootCwd
}

export async function resolveWorkspaceBranch(args: GitArgs, deps: WorkspaceGitDeps): Promise<string | null> {
  const backend = requireExplicitSshBackend(args, deps.resolveBackend(args.workspace))
  if (backend) return gitBranch(backend, remoteCwd(backend, args))
  return deps.localBranch(args.cwd)
}

export async function resolveWorkspaceGitStatus(args: GitArgs, deps: WorkspaceGitDeps): Promise<GitStatus> {
  const backend = requireExplicitSshBackend(args, deps.resolveBackend(args.workspace))
  if (backend) return engineGitStatus(backend, remoteCwd(backend, args))
  return deps.localStatus(args.cwd)
}

export async function resolveWorkspaceGitDiff(
  args: GitArgs & { paths?: string[] },
  deps: WorkspaceGitDeps,
): Promise<GitDiffFile[]> {
  const backend = requireExplicitSshBackend(args, deps.resolveBackend(args.workspace))
  if (backend) return engineGitDiff(backend, remoteCwd(backend, args), args.paths)
  return deps.localDiff(args.cwd, args.paths)
}
