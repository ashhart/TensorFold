import { spawnSync } from 'node:child_process'
import { existsSync, lstatSync, realpathSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { commandShell } from './cli-path'

// Linux command sandbox on bubblewrap, mirroring the macOS sandbox-exec profiles in
// sandbox.ts: deny-by-default filesystem with per-profile read/write roots and an
// optional network namespace. Args only; the caller spawns and owns timeouts/kill.

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

// A writable workspace still may not touch git's control files (see gitControlWriteDeny in
// sandbox.ts). A writable bind of .git itself pins the entry against rename/replace (mountpoints
// reject it with EBUSY); the control entries are ro-bound over themselves, which also blocks
// unlink. Refs and objects stay writable so git-mutate commands work. Two gaps versus SBPL:
// a worktree .git pointer file's external gitdir is not covered, and nested repos' .git entries
// below the top level are not pinned.
function gitControlReadOnlyBinds(cwd: string): string[] {
  const args: string[] = []
  const git = path.join(cwd, '.git')
  try {
    const resolved = realpathSync(git)
    if (resolved === git || resolved.startsWith(`${cwd}/`)) args.push('--bind', resolved, git)
  } catch { return args }
  for (const rel of ['.git/config', '.git/config.worktree', '.git/hooks', '.git/info']) {
    const full = path.join(cwd, rel)
    try {
      const resolved = realpathSync(full)
      if (resolved.startsWith(`${cwd}/`)) args.push('--ro-bind', resolved, full)
    } catch { /* absent control entry: nothing to pin */ }
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
  if (options.allowCwdWrite) args.push(...gitControlReadOnlyBinds(realpathSync(cwd)))
  args.push(...compatSymlinks(linuxReadRoots(cwd, options)))
  args.push('--chdir', cwd, '--')
  return args
}
