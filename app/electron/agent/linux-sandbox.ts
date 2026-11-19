import { spawnSync } from 'node:child_process'
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, statSync, writeFileSync } from 'node:fs'
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
const GIT_CONTROL_FILES = ['config', 'config.worktree']
const GIT_CONTROL_DIRS = ['hooks', 'info']
const NESTED_DEPTH_LIMIT = 32
const NESTED_DIRECTORY_LIMIT = 50_000

const insideWorkspace = (cwd: string, target: string) => target === cwd || target.startsWith(`${cwd}${path.sep}`)

/** Missing is safe to skip; any other inspection failure refuses, because it could hide a repository. */
const isMissing = (error: unknown) => {
  const code = (error as NodeJS.ErrnoException)?.code
  return code === 'ENOENT' || code === 'ENOTDIR'
}

function refuseInspection(target: string, error: unknown): never {
  throw new LinuxSandboxError(`The sandbox cannot inspect ${target}: ${(error as Error)?.message ?? String(error)}`)
}

function workspacePath(cwd: string, target: string): string | null {
  let resolved: string
  try { resolved = realpathSync(target) } catch (error) { if (isMissing(error)) return null; refuseInspection(target, error) }
  if (!insideWorkspace(cwd, resolved)) throw new LinuxSandboxError(`The sandbox will not mount git metadata outside the workspace: ${target}`)
  return resolved
}

// Read-only placeholders prevent writable runs from creating absent Git control entries.
let fixtures: { file: string; dir: string } | null = null
function placeholderSources(cwd: string): { file: string; dir: string } {
  if (fixtures) return fixtures
  const root = path.join(os.homedir(), '.cache', 'tensorfold-linux-sandbox')
  if (insideWorkspace(cwd, root)) throw new LinuxSandboxError(`The sandbox cannot keep placeholders outside a workspace rooted at ${cwd}.`)
  mkdirSync(root, { recursive: true, mode: 0o700 })
  mkdirSync(path.join(root, 'empty-dir'), { recursive: true, mode: 0o700 })
  const emptyFile = path.join(root, 'empty-file')
  if (!existsSync(emptyFile)) writeFileSync(emptyFile, '', { mode: 0o600 })
  fixtures = { file: emptyFile, dir: path.join(root, 'empty-dir') }
  return fixtures
}

function gitdirPointer(pointer: string): string | null {
  try {
    const match = /^gitdir:[ \t]*(.+)$/m.exec(readFileSync(pointer, 'utf8'))
    return match ? match[1].trim() : null
  } catch { return null }
}

function pinControlEntries(cwd: string, gitDir: string, args: string[]): void {
  const sources = placeholderSources(cwd)
  for (const [entry, isDir] of [...GIT_CONTROL_FILES.map((name) => [name, false] as const), ...GIT_CONTROL_DIRS.map((name) => [name, true] as const)]) {
    const target = path.join(gitDir, entry)
    const resolved = workspacePath(cwd, target)
    args.push('--ro-bind', resolved ?? (isDir ? sources.dir : sources.file), target)
  }
}

function pinGitDirectory(cwd: string, gitDir: string, args: string[]): void {
  const resolved = workspacePath(cwd, gitDir)
  if (!resolved) throw new LinuxSandboxError(`The sandbox cannot resolve the git directory ${gitDir}.`)
  args.push('--bind', resolved, gitDir)
  pinControlEntries(cwd, gitDir, args)
}

function pinWorktree(cwd: string, pointer: string, args: string[]): void {
  const pointerPath = workspacePath(cwd, pointer)
  if (!pointerPath) throw new LinuxSandboxError(`The sandbox cannot resolve the worktree pointer at ${pointer}.`)
  const target = gitdirPointer(pointer)
  if (!target) throw new LinuxSandboxError(`The sandbox cannot read the worktree pointer at ${pointer}.`)
  let gitdir: string
  try { gitdir = realpathSync(path.resolve(path.dirname(pointer), target)) } catch { throw new LinuxSandboxError(`The sandbox cannot resolve the worktree gitdir in ${pointer}.`) }
  if (!insideWorkspace(cwd, gitdir)) throw new LinuxSandboxError(`The sandbox refuses a worktree whose gitdir is outside the workspace: ${gitdir}`)
  pinGitDirectory(cwd, gitdir, args)
  // Freezing the pointer stops a run from repointing the workspace at metadata it built itself.
  args.push('--ro-bind', pointerPath, pointer)
  const common = workspacePath(cwd, path.join(gitdir, 'commondir'))
  if (!common) return
  let commonDir: string
  try { commonDir = realpathSync(path.resolve(path.dirname(common), readFileSync(common, 'utf8').trim())) } catch { throw new LinuxSandboxError(`The sandbox cannot resolve the shared git directory in ${common}.`) }
  if (!insideWorkspace(cwd, commonDir)) throw new LinuxSandboxError(`The sandbox refuses a worktree whose shared git directory is outside the workspace: ${commonDir}`)
  pinControlEntries(cwd, commonDir, args)
}

/** Every .git under the workspace. Bounds refuse: silently pinning a subset would be a false promise. */
function nestedGitPaths(cwd: string): string[] {
  const found: string[] = []
  const seen = new Set([realpathSync(cwd)])
  let directories = 0
  const walk = (dir: string, depth: number): void => {
    if (depth > NESTED_DEPTH_LIMIT) throw new LinuxSandboxError(`The sandbox will not walk below ${NESTED_DEPTH_LIMIT} directory levels to find nested repositories.`)
    let entries: Dirent[]
    try { entries = readdirSync(dir, { withFileTypes: true }) } catch (error) {
      if (isMissing(error)) return
      refuseInspection(dir, error)
    }
    for (const entry of entries) {
      if (entry.name === '.git') continue
      const candidate = path.join(dir, entry.name)
      let child: string | null = null
      if (entry.isDirectory()) child = candidate
      else if (entry.isSymbolicLink()) {
        let resolved: string
        let kind: 'file' | 'dir'
        try {
          resolved = realpathSync(candidate)
          kind = statSync(resolved).isDirectory() ? 'dir' : 'file'
        } catch (error) {
          if (isMissing(error)) continue
          refuseInspection(candidate, error)
        }
        if (kind === 'dir' && insideWorkspace(cwd, resolved)) child = resolved
      }
      if (!child) continue
      if (++directories > NESTED_DIRECTORY_LIMIT) throw new LinuxSandboxError(`The sandbox will not walk more than ${NESTED_DIRECTORY_LIMIT} workspace directories to find nested repositories.`)
      const real = realpathSync(child)
      if (seen.has(real)) continue
      seen.add(real)
      const nested = path.join(real, '.git')
      try { lstatSync(nested); found.push(nested) } catch (error) { if (!isMissing(error)) refuseInspection(nested, error) }
      walk(real, depth + 1)
    }
  }
  walk(cwd, 0)
  return found
}

export function gitMetadataArgs(cwd: string): string[] {
  const args: string[] = []
  const rootGit = path.join(cwd, '.git')
  let rootKind: 'dir' | 'file' | null = null
  try { rootKind = lstatSync(rootGit).isDirectory() ? 'dir' : 'file' } catch (error) {
    if (!isMissing(error)) refuseInspection(rootGit, error)
    rootKind = null
  }
  // A workspace without a repository keeps commands refused: bwrap cannot deny creating .git by name.
  if (!rootKind) throw new LinuxSandboxError('The sandbox needs a git workspace; commands stay refused here.')
  if (rootKind === 'file') pinWorktree(cwd, rootGit, args)
  else pinGitDirectory(cwd, rootGit, args)
  for (const nested of nestedGitPaths(cwd)) {
    let kind: 'dir' | 'file'
    try { kind = lstatSync(nested).isDirectory() ? 'dir' : 'file' } catch (error) {
      if (isMissing(error)) continue
      refuseInspection(nested, error)
    }
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
  for (const mount of mounts) args.push(mount.mode === 'rw' ? '--bind' : '--ro-bind', mount.path, mount.path)
  if (options.allowCwdWrite) args.push(...gitMetadataArgs(realpathSync(cwd)))
  args.push(...compatSymlinks(linuxReadRoots(cwd, options)))
  args.push('--chdir', cwd, '--')
  return args
}
