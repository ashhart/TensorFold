import { existsSync, realpathSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import os from 'node:os'
import { resolveCliBinary } from './cli-path'
import { linuxGitReadRoots } from './sandbox-git-roots'

const BWRAP = '/usr/bin/bwrap'
export function linuxSandboxReadRoots(): string[] {
  return [...new Set(['/usr', '/bin', '/sbin', '/lib', '/lib64', '/nix/store', ...linuxNodeRuntimeRoots()].filter(existsSync).map(root => realpathSync(root)))]
}
export function linuxNodeRuntimeRoots(): string[] {
  const binary = resolveCliBinary(null, ['node'])
  if (!path.isAbsolute(binary) || !existsSync(binary)) return []
  const bin = path.dirname(realpathSync(binary))
  const prefix = path.dirname(bin)
  if (path.basename(bin) !== 'bin') return [bin]
  return ['bin', 'lib', 'lib64', 'include'].map(name => path.join(prefix, name)).filter(existsSync)
}
export interface LinuxSandboxLaunch {
  command: string
  args: string[]
  cwd: string
  allowNetwork: boolean
  writableStage?: boolean
  workspaceSource?: string
  readRoots?: string[]
  protectedPaths?: string[]
  readViews?: Array<{ source: string; target: string }>
  environment?: Record<string, string>
}

export function linuxSandboxArgs(input: LinuxSandboxLaunch): string[] {
  const cwd = realpathSync(input.cwd)
  const home = path.relative(cwd, os.homedir())
  if (cwd === path.parse(cwd).root || home === '' || (!home.startsWith('..') && !path.isAbsolute(home))) {
    throw new Error('Choose a project directory instead of a filesystem or home root.')
  }
  const args = ['--die-with-parent', '--new-session', '--unshare-all', '--cap-drop', 'ALL']
  if (input.allowNetwork) args.push('--share-net')
  args.push('--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp')
  // Scratch mounts come first; a workspace under /tmp remains read-only after its bind mount.
  const roots = new Set<string>()
  for (const root of ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/nix/store']) {
    if (!existsSync(root)) continue
    const canonical = realpathSync(root)
    if (!roots.has(canonical)) { args.push('--ro-bind', canonical, canonical); roots.add(canonical) }
    if (canonical !== root) args.push('--symlink', path.relative('/', canonical), root)
  }
  for (const root of linuxNodeRuntimeRoots()) {
    const canonical = realpathSync(root)
    if (![...roots].some(parent => canonical === parent || canonical.startsWith(parent + path.sep))) {
      args.push('--ro-bind', canonical, canonical); roots.add(canonical)
    }
  }
  for (const inputRoot of input.readRoots ?? []) {
    const canonical = realpathSync(inputRoot)
    const relative = path.relative(canonical, cwd)
    const home = path.relative(canonical, os.homedir())
    if (canonical === path.parse(canonical).root || canonical === realpathSync(os.tmpdir())
      || home === '' || (!home.startsWith('..') && !path.isAbsolute(home))
      || relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))) {
      throw new Error('A sandbox runtime root cannot expose the workspace or its ancestors.')
    }
    if (![...roots].some(parent => canonical === parent || canonical.startsWith(parent + path.sep))) {
      args.push('--ro-bind', canonical, canonical); roots.add(canonical)
    }
  }
  for (const view of input.readViews ?? []) args.push('--ro-bind', view.target, view.source)
  if (input.allowNetwork) for (const file of ['/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/ssl/certs']) {
    if (existsSync(file)) args.push('--ro-bind', file, file)
  }
  args.push(input.writableStage ? '--bind' : '--ro-bind', input.workspaceSource ?? cwd, cwd)
  for (const metadata of linuxGitReadRoots(cwd)) args.push('--ro-bind', metadata, metadata)
  for (const relative of input.protectedPaths ?? []) {
    if (!relative || path.isAbsolute(relative) || relative.split(/[\\/]/).includes('..')) throw new Error('Invalid protected sandbox path.')
    const original = path.join(cwd, relative)
    args.push('--ro-bind', original, original)
  }
  for (const [key, value] of Object.entries(input.environment ?? {})) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || value.includes('\0')) throw new Error('Invalid isolated environment entry.')
    args.push('--setenv', key, value)
  }
  args.push('--chdir', cwd, '--', input.command, ...input.args)
  return args
}

let available: boolean | undefined
export function canUseLinuxSandbox(): boolean {
  if (process.platform !== 'linux' || !existsSync(BWRAP)) return false
  if (available !== undefined) return available
  const probe = spawnSync(BWRAP, linuxSandboxArgs({ command: '/usr/bin/true', args: [], cwd: '/tmp', allowNetwork: false }),
    { timeout: 5000, stdio: 'ignore' })
  available = probe.status === 0
  return available
}
export function linuxSandboxCommand(input: LinuxSandboxLaunch): string[] { return [BWRAP, ...linuxSandboxArgs(input)] }
