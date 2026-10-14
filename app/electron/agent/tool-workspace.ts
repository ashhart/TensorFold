import path from 'node:path'
import { runSandboxedCommand, sandboxProfileForCommand } from './sandbox'
import { LocalBackend, type LocalExecImpl, type WorkspaceBackend } from './workspace-backend'
import { rel, safeResolve } from './tool-paths'

interface WorkspaceToolContext {
  cwd: string
  backend?: WorkspaceBackend
  workspace?: { kind: 'local' | 'ssh'; path: string; hostId?: string | null } | null
}

// --- workspace backend helpers (local default; SSH when ctx.backend is remote) ---

const localExecImpl: LocalExecImpl = async (command, options) => {
  const profile = sandboxProfileForCommand(command)
  const result = await runSandboxedCommand(command, {
    cwd: options.cwd ?? process.cwd(),
    profile,
    allowNetwork: profile === 'package-install' ? true : undefined,
    timeoutMs: options.timeoutMs ?? 120_000,
    maxBuffer: 8 * 1024 * 1024,
    signal: options.signal,
  })
  return { code: result.code, stdout: result.stdout, stderr: result.stderr }
}

export function workspaceBackend(ctx: WorkspaceToolContext): WorkspaceBackend {
  if (ctx.workspace?.kind === 'ssh' && ctx.backend?.kind !== 'ssh') {
    throw new Error('The SSH workspace is disconnected; refusing to fall back to the local filesystem.')
  }
  return ctx.backend ?? new LocalBackend(ctx.cwd, localExecImpl)
}

export function isRemote(ctx: WorkspaceToolContext): boolean {
  return ctx.workspace?.kind === 'ssh' || ctx.backend?.kind === 'ssh'
}

// Remote path containment: pure POSIX resolution + prefix check (no local realpath,
// which is meaningless for a remote filesystem).
export function safeResolveRemote(cwd: string, relPath: string | undefined): string {
  const base = cwd.replace(/\/+$/, '') || '/'
  const target = path.posix.resolve(base, (relPath || '.').replace(/\\/g, '/'))
  if (target !== base && !target.startsWith(base + '/')) {
    throw new Error(`Path "${relPath}" escapes the workspace.`)
  }
  return target
}

export function resolveWorkspacePath(ctx: WorkspaceToolContext, relPath: string | undefined): string {
  return isRemote(ctx) ? safeResolveRemote(ctx.cwd, relPath) : safeResolve(ctx.cwd, relPath)
}

export function relWorkspace(ctx: WorkspaceToolContext, abs: string): string {
  if (isRemote(ctx)) {
    const base = ctx.cwd.replace(/\/+$/, '') || '/'
    return path.posix.relative(base, abs) || '.'
  }
  return rel(ctx.cwd, abs)
}

// Tools that still only operate on the local filesystem. On a remote SSH
// workspace they are refused with a clear message rather than silently reading
// the wrong (local) filesystem. Staged for a follow-up.
export const REMOTE_UNSUPPORTED_TOOLS = new Set([
  'apply_patch', 'repo_wiki', 'codebase_map', 'symbol_index', 'retrieval_search',
  'ast_grep_search', 'ast_grep_rewrite_preview', 'semgrep_scan', 'safe_refactor_preview',
])

export function shArg(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}
