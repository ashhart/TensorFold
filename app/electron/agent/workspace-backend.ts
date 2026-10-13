// WorkspaceBackend: the abstraction the tool layer executes against. A LocalBackend
// wraps node fs + the existing sandboxed shell; an SshBackend runs the same
// operations on a remote host over SFTP/exec. tools.ts routes file and shell
// operations through this interface so the agent loop is unchanged whether the
// workspace is local or remote.

import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import type { SshSftpLike } from './ssh-connection'
import { atomicWriteWorkspaceFile, workspaceContentSha256 } from './atomic-file-write'
import { assertWorkspacePathPolicy, canonicalWorkspacePath, type WorkspacePathAccess } from './workspace-path-policy'
import { readOpenedTextFile, withFileReadBudget, type WorkspaceReadOptions, type WorkspaceRangeOptions, type WorkspaceTextRange } from './bounded-file-read'

export interface WorkspaceFileEntry {
  name: string
  isDirectory: boolean
  isFile: boolean
  size: number
}

export interface WorkspaceStat {
  isDirectory: boolean
  isFile: boolean
  size: number
  mtimeMs: number
}

export interface WorkspaceExecResult {
  code: number | null
  stdout: string
  stderr: string
  meta?: string
}

export interface WorkspaceExecOptions {
  cwd?: string
  env?: Record<string, string>
  timeoutMs?: number
  signal?: AbortSignal
}

export interface WorkspaceWriteOptions {
  ifExists?: 'replace' | 'error'
  expectedSha256?: string
}

export interface WorkspaceBackend {
  readonly kind: 'local' | 'ssh'
  readonly rootCwd: string
  readFile(absPath: string, options?: WorkspaceReadOptions): Promise<string>
  readFileRange?(absPath: string, options: WorkspaceRangeOptions): Promise<WorkspaceTextRange>
  writeFile(absPath: string, content: string, options?: WorkspaceWriteOptions): Promise<void>
  readdir(absPath: string): Promise<WorkspaceFileEntry[]>
  stat(absPath: string): Promise<WorkspaceStat>
  mkdirp(absPath: string): Promise<void>
  rename(from: string, to: string): Promise<void>
  remove(absPath: string): Promise<void>
  exists(absPath: string): Promise<boolean>
  /** Optional host-side canonical containment check (required by the SSH backend). */
  assertContained?(absPath: string): Promise<void>
  exec(command: string, options?: WorkspaceExecOptions): Promise<WorkspaceExecResult>
}

const S_IFMT = 0o170000
const S_IFDIR = 0o040000
const S_IFREG = 0o100000

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

// --- local ---

export type LocalExecImpl = (command: string, options: WorkspaceExecOptions) => Promise<WorkspaceExecResult>

export class LocalBackend implements WorkspaceBackend {
  readonly kind = 'local' as const
  constructor(readonly rootCwd: string, private readonly execImpl: LocalExecImpl) {}

  private async readLocal(absPath: string, options: WorkspaceReadOptions, range?: WorkspaceRangeOptions): Promise<Buffer | WorkspaceTextRange> {
    // Re-validate at read time: a leaf symlink could be repointed outside the
    // workspace between safeResolve (tool-paths.ts) and this read. realpath + check.
    // Canonicalize BOTH the target and the root (rootCwd may itself be a symlinked
    // prefix, e.g. macOS /var -> /private/var), so the containment compare is apples
    // to apples and a valid in-workspace read is never rejected.
    return withFileReadBudget(options, async (budget) => {
      const resolved = await budget.wait(canonicalWorkspacePath(this.rootCwd, absPath, 'read'))
      const opening = fs.open(resolved, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
      opening.then((handle) => { if (budget.signal.aborted) void handle.close().catch(() => {}) }, () => {})
      const handle = await budget.wait(opening)
      return readOpenedTextFile({
        stat: async () => {
          const stat = await handle.stat()
          const current = await canonicalWorkspacePath(this.rootCwd, absPath, 'read')
          const currentStat = await fs.stat(current)
          if (current !== resolved || currentStat.dev !== stat.dev || currentStat.ino !== stat.ino) throw new Error('Workspace file changed during open.')
          return { isFile: stat.isFile(), size: stat.size }
        },
        read: (buffer, length, position) => handle.read(buffer, 0, length, position).then((read) => read.bytesRead),
        close: () => handle.close(),
      }, budget, range)
    })
  }

  async readFile(absPath: string, options: WorkspaceReadOptions = {}): Promise<string> {
    return (await this.readLocal(absPath, options) as Buffer).toString('utf8')
  }

  async readFileRange(absPath: string, options: WorkspaceRangeOptions): Promise<WorkspaceTextRange> {
    return await this.readLocal(absPath, options, options) as WorkspaceTextRange
  }

  async writeFile(absPath: string, content: string, options: WorkspaceWriteOptions = {}) {
    await atomicWriteWorkspaceFile(this.rootCwd, absPath, content, options)
  }

  async readdir(absPath: string): Promise<WorkspaceFileEntry[]> {
    const entries = await fs.readdir(absPath, { withFileTypes: true })
    return entries.map((e) => ({ name: e.name, isDirectory: e.isDirectory(), isFile: e.isFile(), size: 0 }))
  }

  async stat(absPath: string): Promise<WorkspaceStat> {
    const s = await fs.stat(absPath)
    return { isDirectory: s.isDirectory(), isFile: s.isFile(), size: s.size, mtimeMs: s.mtimeMs }
  }

  async mkdirp(absPath: string) {
    const target = await canonicalWorkspacePath(this.rootCwd, absPath, 'write')
    await fs.mkdir(target, { recursive: true })
  }
  async rename(from: string, to: string) {
    await canonicalWorkspacePath(this.rootCwd, from, 'write')
    await canonicalWorkspacePath(this.rootCwd, to, 'write')
    await fs.rename(from, to)
  }
  async remove(absPath: string) {
    await canonicalWorkspacePath(this.rootCwd, absPath, 'write')
    await fs.rm(absPath, { recursive: true, force: true })
  }

  async exists(absPath: string) {
    try { await fs.access(absPath); return true } catch { return false }
  }

  exec(command: string, options: WorkspaceExecOptions = {}) {
    return this.execImpl(command, { cwd: options.cwd ?? this.rootCwd, ...options })
  }
}

// --- ssh ---

export interface SshBackendSession {
  exec(command: string, options?: { cwd?: string; env?: Record<string, string>; timeoutMs?: number; signal?: AbortSignal }): Promise<{ code: number | null; stdout: string; stderr: string }>
  sftp(): Promise<SshSftpLike>
}

// SECURITY MODEL — read before relying on the path checks below.
// On an SSH remote workspace, file-tool path containment (safeResolveRemote + assertContained)
// is best-effort DEFENCE-IN-DEPTH, not a hard sandbox. Two structural reasons:
//   1. run_command on this backend is an UNJAILED remote shell (tools.ts: session.exec of an
//      arbitrary command). An adversarial/injected agent can already `cat /etc/passwd` or
//      `echo k >> ~/.ssh/authorized_keys` directly — a far wider door than any file-tool path.
//   2. The containment check (canonicalize over the exec channel) and the actual op (SFTP, raw
//      path) run on SEPARATE round-trips, so they are inherently TOCTOU-racy: a backgrounded
//      `run_command` symlink-flip can swap an in-workspace component between check and op. SFTP
//      v3 offers no atomic open-no-follow, so this cannot be fully closed client-side.
// The checks here DO close accidental/lexical/static-symlink escapes (the common, non-adversarial
// cases) and are worth keeping. But the REAL boundary for an untrusted SSH workspace is the SSH
// account's own OS permissions and/or a server-side chroot/sftp-only jail — configure that if the
// remote agent is not trusted. See docs and the bug-audit memory note.
export class SshBackend implements WorkspaceBackend {
  readonly kind = 'ssh' as const
  private sftpPromise: Promise<SshSftpLike> | null = null
  constructor(readonly rootCwd: string, private readonly session: SshBackendSession) {}

  private getSftp(): Promise<SshSftpLike> {
    if (!this.sftpPromise) {
      // Don't cache a rejected promise: a single SFTP channel failure on a
      // still-live SSH transport must not permanently break file ops. Clear the
      // cache on rejection so the next access retries opening a fresh channel.
      const p: Promise<SshSftpLike> = this.session.sftp().catch((e) => {
        if (this.sftpPromise === p) this.sftpPromise = null
        throw e
      })
      this.sftpPromise = p
    }
    return this.sftpPromise
  }

  // safeResolveRemote only does lexical containment; it can't see remote symlinks. Before any
  // file op, canonicalize the path's parent on the remote (pwd -P resolves symlinks portably)
  // and verify it stays under the canonical workspace root, so a symlinked directory on the
  // host can't be used to read/write outside the workspace.
  private canonRootPromise: Promise<string> | null = null
  // Canonicalize the FULL path on the remote. `realpath -m` resolves symlinks in EVERY
  // component INCLUDING the leaf (so a symlinked leaf like `ln -s /etc/passwd evil` is caught,
  // not just a symlinked parent) and tolerates a not-yet-created trailing component. Fall back
  // to pwd -P on the parent (portable) only if realpath is unavailable.
  private resolveRemotePath(p: string, options: WorkspaceReadOptions = {}): Promise<string> {
    const full = shellQuote(p)
    const base = path.posix.basename(p).replace(/'/g, `'\\''`)
    // Prefer GNU `realpath -m` (resolves every component incl. the leaf, tolerates missing
    // tail). Portable fallback for BSD/busybox/Alpine hosts that lack it: reject a symlinked
    // LEAF outright (test -L is POSIX), and canonicalize the DEEPEST EXISTING ANCESTOR via
    // pwd -P (so a symlinked parent dir — even under a not-yet-created path — is resolved and
    // caught, not just the immediate parent).
    const fallback =
      `if [ -L ${full} ]; then printf '/__vontra_reject__'; else ` +
      `d=$(dirname -- ${full}); r=; ` +
      `while [ ! -d "$d" ] && [ "$d" != / ]; do r=/$(basename -- "$d")$r; d=$(dirname -- "$d"); done; ` +
      // cd into the deepest existing ancestor. If that FAILS (e.g. the ancestor is a symlink
      // to a dir this user can't enter — /root, another user's home), emit the reject sentinel
      // and FAIL CLOSED rather than empty output (which assertContained would otherwise allow).
      `b=$(cd -- "$d" 2>/dev/null && pwd -P) || { printf '/__vontra_reject__'; exit 0; }; ` +
      `printf '%s%s/%s' "$b" "$r" '${base}'; fi`
    return this.exec(`realpath -m -- ${full} 2>/dev/null || { ${fallback}; }`, { signal: options.signal, timeoutMs: options.timeoutMs })
      .then((r) => r.stdout.trim().replace(/\/+$/, ''))
  }
  async assertContained(absPath: string, access?: WorkspacePathAccess, options: WorkspaceReadOptions = {}): Promise<void> {
    if (access) assertWorkspacePathPolicy(this.rootCwd, absPath, access, path.posix)
    // Resolve the canonical root, but NEVER cache an empty/failed result: a single transient empty
    // stdout (exec timeout, premature channel close, a flaky realpath emitting nothing) would
    // otherwise permanently downgrade symlink containment to lexical-only for the backend's lifetime.
    // Cache only a real (non-empty) root; on empty, clear the cache so the next op retries (mirrors
    // getSftp's don't-cache-failure rule).
    if (!this.canonRootPromise) {
      const p = this.resolveRemotePath(this.rootCwd, options).then((root) => {
        if (!root && this.canonRootPromise === p) this.canonRootPromise = null
        return root
      }).catch((error) => { if (this.canonRootPromise === p) this.canonRootPromise = null; throw error })
      this.canonRootPromise = p
    }
    const root = await this.canonRootPromise
    // Can't canonicalize the root on THIS attempt => the canonicalizer is unavailable/broken right
    // now. Fail CLOSED rather than silently allowing: an unresolvable path (empty / reject sentinel)
    // must be rejected, since an attacker can force an empty result (symlink to a dir the SSH user
    // can't cd into) and the whole point of this check is to catch exactly that.
    if (!root) {
      throw new Error(`Cannot verify workspace containment for "${absPath}" (workspace root could not be canonicalized).`)
    }
    const canon = await this.resolveRemotePath(absPath, options)
    if (!canon || (canon !== root && !canon.startsWith(root + '/'))) {
      throw new Error(`Path "${absPath}" escapes the workspace through a symlink.`)
    }
    if (access) {
      assertWorkspacePathPolicy(root, canon, access, path.posix)
      // A .git directory alias or worktree pointer can have a canonical name
      // without a .git component. Inspect only these small Git control pointers.
      const dotGit = path.posix.join(this.rootCwd, '.git')
      let control = await this.resolveRemotePath(dotGit, options)
      if (!control || control === '/__vontra_reject__') throw new Error('Cannot verify Git control path safely.')
      const sftp = await this.getSftp()
      const readPointer = async (file: string, prefix = ''): Promise<string | null> => {
        const stat = await sftp.stat(file).catch((error: any) => {
          if (error?.code === 2 || error?.code === 'ENOENT' || error?.message === 'ENOENT') return null
          throw error
        })
        if (!stat?.isFile) return null
        if (stat.size > 4096) throw new Error('Git control pointer exceeds the size limit.')
        const bytes = await sftp.readFile(file, { ...options, maxBytes: 4096 })
        if (bytes.length > 4096) throw new Error('Git control pointer exceeds the size limit.')
        const value = bytes.toString('utf8').trim()
        if (!value.startsWith(prefix)) return null
        return this.resolveRemotePath(path.posix.resolve(path.posix.dirname(file), value.slice(prefix.length).trim()), options)
      }
      control = await readPointer(dotGit, 'gitdir:') ?? control
      const common = await readPointer(path.posix.join(control, 'commondir'))
      for (const directory of [control, common]) {
        if (directory && (canon === directory || canon.startsWith(directory + '/'))) {
          throw new Error('Blocked by TensorFold safety policy: refuses accessing Git private metadata.')
        }
      }
    }
  }

  private async readRemote(absPath: string, options: WorkspaceReadOptions, range?: WorkspaceRangeOptions): Promise<Buffer | WorkspaceTextRange> {
    return withFileReadBudget(options, async (budget) => {
      const bounded = { ...options, signal: budget.signal, maxBytes: budget.maxBytes }
      budget.signal.addEventListener('abort', () => { this.sftpPromise = null }, { once: true })
      await budget.wait(this.assertContained(absPath, 'read', bounded))
      const sftp = await budget.wait(this.getSftp())
      const stat = await budget.wait(sftp.stat(absPath))
      if (!stat.isFile) throw new Error('Refusing to read a non-regular workspace file.')
      if (range && sftp.readFileRange) return budget.wait(sftp.readFileRange(absPath, { ...range, ...bounded }))
      if (stat.size > budget.maxBytes) throw new Error(`Workspace file exceeds the ${budget.maxBytes}-byte read limit.`)
      const bytes = await budget.wait(sftp.readFile(absPath, bounded))
      if (bytes.length > budget.maxBytes) throw new Error(`Workspace file exceeds the ${budget.maxBytes}-byte read limit.`)
      if (!range) return bytes
      return readOpenedTextFile({
        stat: async () => ({ isFile: true, size: bytes.length }),
        read: async (buffer, length, position) => bytes.copy(buffer, 0, position, position + length),
        close: async () => {},
      }, budget, range)
    })
  }

  async readFile(absPath: string, options: WorkspaceReadOptions = {}): Promise<string> {
    return (await this.readRemote(absPath, options) as Buffer).toString('utf8')
  }

  async readFileRange(absPath: string, options: WorkspaceRangeOptions): Promise<WorkspaceTextRange> {
    return await this.readRemote(absPath, options, options) as WorkspaceTextRange
  }

  async writeFile(absPath: string, content: string, options: WorkspaceWriteOptions = {}) {
    // Containment BEFORE mkdirp: `mkdir -p` follows a symlinked parent and would create the
    // directory OUTSIDE the workspace (e.g. /etc/newdir via a symlinked /srv/app/evil) before
    // a post-write check could reject it. Reject first, then create.
    await this.assertContained(absPath, 'write')
    if (options.ifExists === 'error' && await this.exists(absPath)) {
      throw new Error(`Target "${absPath}" already exists; refusing to overwrite concurrent work.`)
    }
    if (options.expectedSha256) {
      const current = await this.readFile(absPath)
      if (workspaceContentSha256(current) !== options.expectedSha256) {
        throw new Error(`Target "${absPath}" changed after it was read; refusing to overwrite concurrent work.`)
      }
    }
    await this.mkdirp(path.posix.dirname(absPath))
    const sftp = await this.getSftp()
    await sftp.writeFile(absPath, content)
  }

  async readdir(absPath: string): Promise<WorkspaceFileEntry[]> {
    await this.assertContained(absPath)
    const sftp = await this.getSftp()
    const entries = await sftp.readdir(absPath)
    return entries.map((e) => ({
      name: e.filename,
      isDirectory: (e.attrs.mode & S_IFMT) === S_IFDIR,
      isFile: (e.attrs.mode & S_IFMT) === S_IFREG,
      size: e.attrs.size,
    }))
  }

  async stat(absPath: string): Promise<WorkspaceStat> {
    await this.assertContained(absPath)
    const sftp = await this.getSftp()
    const s = await sftp.stat(absPath)
    return { isDirectory: s.isDirectory, isFile: s.isFile, size: s.size, mtimeMs: s.mtime * 1000 }
  }

  async mkdirp(absPath: string) {
    await this.assertContained(absPath, 'write')
    const res = await this.session.exec(`mkdir -p ${shellQuote(absPath)}`)
    if (res.code !== 0) throw new Error(res.stderr || `mkdir -p failed for ${absPath}`)
  }

  async rename(from: string, to: string) {
    await this.assertContained(from, 'write')
    await this.assertContained(to, 'write')
    const sftp = await this.getSftp()
    await sftp.rename(from, to)
  }

  async remove(absPath: string) {
    await this.assertContained(absPath, 'write')
    const res = await this.session.exec(`rm -rf ${shellQuote(absPath)}`)
    if (res.code !== 0) throw new Error(res.stderr || `rm -rf failed for ${absPath}`)
  }

  async exists(absPath: string) {
    try { await this.stat(absPath); return true } catch { return false }
  }

  async exec(command: string, options: WorkspaceExecOptions = {}): Promise<WorkspaceExecResult> {
    const res = await this.session.exec(command, { cwd: options.cwd ?? this.rootCwd, env: options.env, timeoutMs: options.timeoutMs, signal: options.signal })
    return { code: res.code, stdout: res.stdout, stderr: res.stderr, meta: 'ssh' }
  }
}
