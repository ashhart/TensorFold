import { spawnSync } from 'node:child_process'
import { existsSync, lstatSync, realpathSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Linux command sandbox on bubblewrap, mirroring the macOS sandbox-exec profiles in
// sandbox.ts: deny-by-default filesystem with per-profile read/write roots and an
// optional network namespace. Args only; the caller spawns and owns timeouts/kill.

export const LINUX_SANDBOX_IMPLEMENTATION = 'linux-bwrap'

export function linuxSandboxBinary(): string | null {
  for (const candidate of ['/usr/bin/bwrap', '/bin/bwrap']) if (existsSync(candidate)) return candidate
  return null
}

let linuxSandboxAvailable: boolean | undefined

/** Probe bwrap with a representative minimal namespace; cached like the macOS probe. */
export function canUseLinuxSandbox(): boolean {
  if (process.platform !== 'linux') return false
  const binary = linuxSandboxBinary()
  if (!binary) return false
  if (linuxSandboxAvailable !== undefined) return linuxSandboxAvailable
  const probe = spawnSync(binary, ['--ro-bind', '/usr', '/usr', '--dev', '/dev', '--unshare-net', '--die-with-parent', '--', '/usr/bin/true'], {
    stdio: 'ignore',
    timeout: 2_000,
  })
  linuxSandboxAvailable = probe.status === 0
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
}

/** bwrap argv (without the binary) implementing the same contract as macSandboxProfile. */
export function linuxSandboxArgv(cwd: string, options: LinuxSandboxProfileOptions): string[] {
  const readRoots = resolveRoots([
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
  const writeRoots = resolveRoots([
    ...(options.allowCwdWrite ? [cwd] : []),
    os.tmpdir(),
    process.env.TMPDIR,
  ])
  const writable = new Set(writeRoots.map((root) => root.resolved))
  const args: string[] = ['--unshare-all', '--die-with-parent']
  if (options.allowNetwork) args.push('--share-net')
  args.push('--proc', '/proc', '--dev', '/dev')
  for (const root of readRoots) {
    if (writable.has(root.resolved)) continue
    args.push('--ro-bind', root.resolved, root.resolved)
  }
  for (const root of writeRoots) args.push('--bind', root.resolved, root.resolved)
  if (options.allowCwdWrite) args.push(...gitControlReadOnlyBinds(realpathSync(cwd)))
  args.push(...compatSymlinks(readRoots))
  args.push('--chdir', cwd, '--')
  return args
}
