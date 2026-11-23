import { realpathSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { resolveDeveloperReadRoot } from './mac-developer-directory'

export type SandboxProfileId =
  | 'read-only'
  | 'build-test'
  | 'package-install'
  | 'browser'
  | 'network-fetch'
  | 'connector-send'
  | 'system-control'
  | 'untrusted-code'

export interface SandboxProfileDefinition {
  id: SandboxProfileId
  label: string
  allowCwdWrite: boolean
  allowNetwork: boolean
  envSecretsAllowed: boolean
  description: string
}

export const SANDBOX_PROFILES: Record<SandboxProfileId, SandboxProfileDefinition> = {
  'read-only': {
    id: 'read-only',
    label: 'Read-only',
    allowCwdWrite: false,
    allowNetwork: false,
    envSecretsAllowed: false,
    description: 'Read workspace files and system tools without workspace writes or network.',
  },
  'build-test': {
    id: 'build-test',
    label: 'Build/test',
    allowCwdWrite: true,
    allowNetwork: false,
    envSecretsAllowed: false,
    description: 'Run local builds, typechecks, tests, and git-local operations in the workspace with network denied.',
  },
  'package-install': {
    id: 'package-install',
    label: 'Package install',
    allowCwdWrite: false,
    allowNetwork: false,
    envSecretsAllowed: false,
    description: 'Evaluate package-install commands away from host project writes; network still requires separate policy.',
  },
  browser: {
    id: 'browser',
    label: 'Browser',
    allowCwdWrite: false,
    allowNetwork: true,
    envSecretsAllowed: false,
    description: 'Browser/preview automation with network allowed and no workspace writes.',
  },
  'network-fetch': {
    id: 'network-fetch',
    label: 'Network fetch',
    allowCwdWrite: false,
    allowNetwork: true,
    envSecretsAllowed: false,
    description: 'Explicit public network fetches without workspace writes.',
  },
  'connector-send': {
    id: 'connector-send',
    label: 'Connector send',
    allowCwdWrite: false,
    allowNetwork: true,
    envSecretsAllowed: false,
    description: 'Approval-gated outbound connector delivery without workspace writes.',
  },
  'system-control': {
    id: 'system-control',
    label: 'System control',
    allowCwdWrite: false,
    allowNetwork: false,
    envSecretsAllowed: false,
    description: 'System-control commands stay isolated and approval-gated; no network or workspace writes.',
  },
  'untrusted-code': {
    id: 'untrusted-code',
    label: 'Untrusted code',
    allowCwdWrite: false,
    allowNetwork: false,
    envSecretsAllowed: false,
    description: 'Untrusted code interpreter profile with no network, no workspace writes, and no inherited secrets.',
  },
}

export function sandboxProfileForCommandClass(classification: string | null | undefined): SandboxProfileId {
  switch (classification) {
    case 'git-read':
    case 'read-only':
    case 'secret-read':
      return 'read-only'
    case 'test':
    case 'build':
    case 'git-mutate':
    case 'destructive-filesystem':
    case 'shell':
    case 'compound-shell':
    case 'empty-command':
      return 'build-test'
    case 'package-install':
      return 'package-install'
    case 'network-fetch':
    case 'network-shell':
      return 'network-fetch'
    case 'system-control':
      return 'system-control'
    default:
      return 'untrusted-code'
  }
}

export function sandboxProfileAllowsNetwork(profile: SandboxProfileId, override?: boolean): boolean {
  return override ?? SANDBOX_PROFILES[profile].allowNetwork
}

// The global network toggle only applies to commands already classified as network-bearing.
export function taskSandboxNetworkOverride(profile: SandboxProfileId, enabled: boolean | undefined): boolean | undefined {
  if (profile === 'package-install') return true
  if (profile === 'network-fetch') return enabled === true
  return undefined
}

export function macSandboxProfile(cwd: string, profileOrNetwork: SandboxProfileId | boolean = 'build-test', extraReadRoots: Array<string | null | undefined> = [], allowNetworkOverride?: boolean): string {
  const profile = typeof profileOrNetwork === 'boolean' ? (profileOrNetwork ? 'network-fetch' : 'build-test') : profileOrNetwork
  const definition = SANDBOX_PROFILES[profile]
  const allowNetwork = allowNetworkOverride ?? definition.allowNetwork
  const readRoots = uniqueExistingPaths([
    cwd,
    '/bin',
    '/sbin',
    '/usr/bin',
    '/usr/sbin',
    '/usr/lib',
    '/usr/libexec',
    '/usr/share',
    '/System',
    '/Library',
    '/opt/homebrew',
    '/usr/local',
    // /usr/bin/git, make, clang… are xcrun shims that load from the selected developer dir.
    developerDir(),
    os.tmpdir(),
    process.env.TMPDIR,
    darwinUserTempDir(),
    ...extraReadRoots,
  ])
  const writeRoots = uniqueExistingPaths([
    ...(definition.allowCwdWrite ? [cwd] : []),
    os.tmpdir(),
    process.env.TMPDIR,
    darwinUserTempDir(),
  ])
  const readRules = readRoots.map((root) => `(subpath ${sbplString(root)})`).join('\n      ')
  const writeRules = writeRoots.map((root) => `(subpath ${sbplString(root)})`).join('\n      ')
  // Deny network-adjacent mach services after the broad lookup allowance, preserving runtime and compiler access.
  const machNetworkDeny = allowNetwork
    ? ''
    : `
(deny mach-lookup
      (global-name "com.apple.mDNSResponder")
      (global-name "com.apple.mDNSResponder.dnsproxy")
      (global-name "com.apple.dnssd.service")
      (global-name "com.apple.SystemConfiguration.DNSConfiguration")
      (global-name "com.apple.SystemConfiguration.PPPController")
      (global-name "com.apple.SystemConfiguration.SCNetworkReachability")
      (global-name "com.apple.SystemConfiguration.configd")
      (global-name "com.apple.network.statistics")
      (global-name "com.apple.networkd")
      (global-name "com.apple.nehelper")
      (global-name "com.apple.nesessionmanager"))`
  return `(version 1)
(deny default)
(allow process*)
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow file-map-executable)
(allow file-read-metadata)
(allow file-read-data (literal "/"))
(allow file-read*
      ${readRules}
      (literal "/dev/null")
      (literal "/dev/random")
      (literal "/dev/urandom"))
(allow file-write*
      ${writeRules}
      (literal "/dev/null"))${definition.allowCwdWrite ? gitControlWriteDeny(cwd) : ''}${machNetworkDeny}
${allowNetwork ? '(allow network*)' : ''}`
}

// Deny git control-file writes at any depth because host-side git runs outside the sandbox.
export function gitControlWriteDeny(cwd: string): string {
  let root: string
  try { root = realpathSync(path.resolve(cwd)) } catch { return '' }
  const literal = `
      (literal ${sbplString(path.join(root, '.git'))})
      (subpath ${sbplString(path.join(root, '.git', 'config'))})
      (subpath ${sbplString(path.join(root, '.git', 'config.worktree'))})
      (subpath ${sbplString(path.join(root, '.git', 'hooks'))})
      (subpath ${sbplString(path.join(root, '.git', 'info'))})`
  // A quote would end the #"..." literal; such a root keeps the top-level rules only.
  const nested = root.includes('"') ? '' : `
      (regex #"^${sbplRegexEscape(root)}/.+/\\.git$")
      (regex #"^${sbplRegexEscape(root)}/(.+/)?\\.git/(config|config\\.worktree|hooks|info|modules/.+/(config|hooks|info))(/.*)?$")`
  return `
(deny file-write*${literal}${nested})`
}

let cachedDeveloperDir: string | null | undefined
function developerDir(): string | null {
  if (cachedDeveloperDir === undefined) cachedDeveloperDir = resolveDeveloperReadRoot()
  return cachedDeveloperDir
}

// xcrun caches lookups in DARWIN_USER_TEMP_DIR even when a scrubbed launch environment omits TMPDIR.
let cachedUserTempDir: string | null | undefined
function darwinUserTempDir(): string | null {
  if (cachedUserTempDir === undefined) {
    const result = process.platform === 'darwin' ? spawnSync('/usr/bin/getconf', ['DARWIN_USER_TEMP_DIR'], { encoding: 'utf8', timeout: 2000 }) : null
    cachedUserTempDir = result?.status === 0 && result.stdout.trim() ? result.stdout.trim() : null
  }
  return cachedUserTempDir
}

function sbplRegexEscape(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function uniqueExistingPaths(paths: Array<string | null | undefined>): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const item of paths) {
    if (!item) continue
    try {
      const resolved = realpathSync(path.resolve(item))
      if (!seen.has(resolved)) {
        seen.add(resolved)
        out.push(resolved)
      }
    } catch { /* ignore missing roots */ }
  }
  return out
}

export function sbplString(value: string): string {
  return JSON.stringify(value)
}
