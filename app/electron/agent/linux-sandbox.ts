import { spawnSync } from 'node:child_process'
import { existsSync, lstatSync, readFileSync, readdirSync, realpathSync } from 'node:fs'
import type { Dirent } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { commandShell } from './cli-path'

// Linux command sandbox on bubblewrap: macOS sandbox-exec profiles, mirrored as bwrap argv only.

export const LINUX_SANDBOX_IMPLEMENTATION = 'linux-bwrap'

export function linuxSandboxBinary(): string | null {
  for (const candidate of ['/usr/bin/bwrap', '/bin/bwrap']) if (existsSync(candidate)) return candidate
  return null
}

let linuxSandboxAvailable: boolean | undefined

/** Probe the real production argv with a read-only fixture, so loader and mount rules are exercised. */
export function probeLinuxSandbox(cwd: string, run = spawnSync): boolean {
  const binary = linuxSandboxBinary()
  if (!binary) return false
  const probe = run(binary, [...linuxSandboxArgv(cwd, { allowCwdWrite: false, allowNetwork: false, production: false }), commandShell(), '-c', 'exit 0'], {
    stdio: 'ignore',
    timeout: 4_000,
  })
  return probe.status === 0
}

export function canUseLinuxSandbox(): boolean {
  // Opt-in only until conformance is qualified; never enabled by default on an unverified host.
  const qualified = process.env.TENSORFOLD_LINUX_SANDBOX === '1'
  if (!qualified || process.platform !== 'linux') return false
  const binary = linuxSandboxBinary()
  if (!binary) return false
  if (linuxSandboxAvailable !== undefined) return linuxSandboxAvailable
  linuxSandboxAvailable = probeLinuxSandbox(process.cwd())
  return linuxSandboxAvailable
}

interface ResolvedRoot {
  requested: string
  resolved: string
}

/** Existing roots, realpath'd (usrmerge collapses /bin into /usr/bin), parents before children. */
function resolveRoots(requested: Array<string | null | undefined>): ResolvedRoot[] {
  const seen = new Set<string>()
  const out: ResolvedRoot[] = []
  for (const item of requested) {
    if (!item) continue
    let resolved: string
    try {
      resolved = realpathSync(path.resolve(item))
    } catch { continue }
    if (seen.has(resolved)) continue
    seen.add(resolved)
    out.push({ requested: path.resolve(item), resolved })
  }
  return out.sort((a, b) => a.resolved.length - b.resolved.length || a.resolved.localeCompare(b.resolved))
}

/** usrmerge compatibility links: /bin -> usr/bin etc. so absolute interpreter paths still resolve. */
function compatSymlinks(roots: ResolvedRoot[]): string[] {
  const args: string[] = []
  for (const root of roots) {
    if (root.requested === root.resolved) continue
    try {
      if (!lstatSync(root.requested).isSymbolicLink()) continue
    } catch { continue }
    args.push('--symlink', root.resolved, root.requested)
  }
  return args
}

/** A workspace the sandbox cannot protect: git metadata the profile would not mount. */
export class LinuxSandboxError extends Error {}

// Git control files stay read-only under a writable workspace; refs and objects stay writable.
const GIT_CONTROL_ENTRIES = ['config', 'config.worktree', 'hooks', 'info']
const NESTED_ENTRY_LIMIT = 2_048
const NESTED_DEPTH_LIMIT = 8

const insideWorkspace = (cwd: string, target: string) => target === cwd || target.startsWith(`${cwd}${path.sep}`)

/** Real path under the workspace, or null when the entry does not exist. */
function workspacePath(cwd: string, target: string): string | null {
  let resolved: string
  try { resolved = realpathSync(target) } catch { return null }
  if (!insideWorkspace(cwd, resolved)) throw new LinuxSandboxError(`The sandbox will not mount git metadata outside the workspace: ${target}`)
  return resolved
}

function gitdirPointer(pointer: string): string | null {
  try {
    const match = /^gitdir:[ \t]*(.+)$/m.exec(readFileSync(pointer, 'utf8'))
    return match ? match[1].trim() : null
  } catch { return null }
}

/** A git directory the profile pins: writable itself, its control entries read-only. */
function pinGitDirectory(cwd: string, gitDir: string, args: string[]): void {
  const resolved = workspacePath(cwd, gitDir)
  if (!resolved) return
  args.push('--bind', resolved, gitDir)
  for (const entry of GIT_CONTROL_ENTRIES) {
    const resolvedEntry = workspacePath(cwd, path.join(gitDir, entry))
    if (resolvedEntry) args.push('--ro-bind', resolvedEntry, path.join(gitDir, entry))
  }
}

/** A worktree's real gitdir, refusing one that lives outside the workspace. */
function pinWorktree(cwd: string, pointer: string, args: string[]): void {
  const target = gitdirPointer(pointer)
  if (!target) throw new LinuxSandboxError(`The sandbox cannot read the worktree pointer at ${pointer}.`)
  let gitdir: string
  try { gitdir = realpathSync(path.resolve(path.dirname(pointer), target)) } catch { throw new LinuxSandboxError(`The sandbox cannot resolve the worktree gitdir in ${pointer}.`) }
  if (!insideWorkspace(cwd, gitdir)) throw new LinuxSandboxError(`The sandbox refuses a worktree whose gitdir is outside the workspace: ${gitdir}`)
  pinGitDirectory(cwd, gitdir, args)
  const common = workspacePath(cwd, path.join(gitdir, 'commondir'))
  if (!common) return
  let commonDir: string
  try { commonDir = realpathSync(path.resolve(path.dirname(common), readFileSync(common, 'utf8').trim())) } catch { throw new LinuxSandboxError(`The sandbox cannot resolve the shared git directory in ${common}.`) }
  if (!insideWorkspace(cwd, commonDir)) throw new LinuxSandboxError(`The sandbox refuses a worktree whose shared git directory is outside the workspace: ${commonDir}`)
  for (const entry of GIT_CONTROL_ENTRIES) {
    const resolvedEntry = workspacePath(cwd, path.join(commonDir, entry))
    if (resolvedEntry) args.push('--ro-bind', resolvedEntry, path.join(commonDir, entry))
  }
}

/** Every .git under the workspace, bounded: an unbounded walk would silently pin a subset. */
function nestedGitPaths(cwd: string): string[] {
  const found: string[] = []
  let visited = 0
  const walk = (dir: string, depth: number): void => {
    if (depth > NESTED_DEPTH_LIMIT) return
    let entries: Dirent[]
    try { entries = readdirSync(dir, { withFileTypes: true }) } catch { return }
    for (const entry of entries) {
      if (++visited > NESTED_ENTRY_LIMIT) throw new LinuxSandboxError(`The sandbox will not walk more than ${NESTED_ENTRY_LIMIT} workspace entries to find nested repositories.`)
      if (entry.name === '.git' || entry.name === 'node_modules' || !entry.isDirectory()) continue
      const child = path.join(dir, entry.name)
      if (existsSync(path.join(child, '.git'))) found.push(path.join(child, '.git'))
      walk(child, depth + 1)
    }
  }
  walk(cwd, 0)
  return found
}

/** Root git metadata plus every nested repository, all pinned the same way. */
export function gitMetadataArgs(cwd: string): string[] {
  const args: string[] = []
  const rootGit = path.join(cwd, '.git')
  let rootKind: 'dir' | 'file' | null = null
  try { rootKind = lstatSync(rootGit).isDirectory() ? 'dir' : 'file' } catch { rootKind = null }
  // A workspace without a repository keeps commands refused: bwrap cannot deny creating .git by name.
  if (!rootKind) throw new LinuxSandboxError('The sandbox needs a git workspace; commands stay refused here.')
  if (rootKind === 'file') pinWorktree(cwd, rootGit, args)
  else pinGitDirectory(cwd, rootGit, args)
  for (const nested of nestedGitPaths(cwd)) {
    let kind: 'dir' | 'file'
    try { kind = lstatSync(nested).isDirectory() ? 'dir' : 'file' } catch { continue }
    if (kind === 'file') pinWorktree(cwd, nested, args)
    else pinGitDirectory(cwd, nested, args)
  }
  return args
}

export interface LinuxSandboxProfileOptions {
  allowCwdWrite: boolean
  allowNetwork: boolean
  extraReadRoots?: Array<string | null | undefined>
  /** False for the capability probe, which must not inspect or depend on the caller's workspace. */
  production?: boolean
}

/** Every requested read root, kept with its unresolved name so usrmerge links stay visible. */
export function linuxReadRoots(cwd: string, options: LinuxSandboxProfileOptions): ResolvedRoot[] {
  return resolveRoots([
    cwd,
    '/bin',
    '/sbin',
    '/lib',
    '/lib64',
    '/usr/bin',
    '/usr/sbin',
    '/usr/lib',
    '/usr/libexec',
    '/usr/share',
    '/usr/local',
    '/etc',
    '/opt',
    os.tmpdir(),
    process.env.TMPDIR,
    ...(options.extraReadRoots ?? []),
  ])
}

/** One final decision per resolved path; later mounts win, so children must follow their parents. */
export function planLinuxMounts(cwd: string, options: LinuxSandboxProfileOptions): Array<{ mode: 'ro' | 'rw'; path: string }> {
  const readRoots = linuxReadRoots(cwd, options)
  const writeRoots = resolveRoots([
    ...(options.allowCwdWrite ? [cwd] : []),
    os.tmpdir(),
    process.env.TMPDIR,
  ])
  const writable = new Set(writeRoots.map((root) => root.resolved))
  const byPath = new Map<string, 'ro' | 'rw'>()
  for (const root of readRoots) byPath.set(root.resolved, writable.has(root.resolved) ? 'rw' : 'ro')
  for (const root of writeRoots) byPath.set(root.resolved, 'rw')
  // A read-only workspace always wins, even when it sits inside a writable temporary parent.
  const canonicalCwd = (() => { try { return realpathSync(cwd) } catch { return path.resolve(cwd) } })()
  if (!options.allowCwdWrite) byPath.set(canonicalCwd, 'ro')
  return [...byPath.entries()]
    .map(([mountPath, mode]) => ({ mode, path: mountPath }))
    .sort((a, b) => a.path.length - b.path.length || a.path.localeCompare(b.path))
}

/** bwrap argv (without the binary) implementing the same contract as macSandboxProfile. */
export function linuxSandboxArgv(cwd: string, options: LinuxSandboxProfileOptions): string[] {
  const mounts = planLinuxMounts(cwd, options)
  const args: string[] = ['--unshare-all', '--die-with-parent']
  if (options.allowNetwork) args.push('--share-net')
  args.push('--proc', '/proc', '--dev', '/dev')
  // Parent-first order is what makes a read-only child decisive.
  for (const mount of mounts) args.push(mount.mode === 'rw' ? '--bind' : '--ro-bind', mount.path, mount.path)
  if (options.allowCwdWrite) args.push(...gitMetadataArgs(realpathSync(cwd)))
  args.push(...compatSymlinks(linuxReadRoots(cwd, options)))
  args.push('--chdir', cwd, '--')
  return args
}
