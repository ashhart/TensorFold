import { canUseLinuxSandbox } from './linux-sandbox'
import { canUseWindowsSandbox } from './windows-sandbox'
import { runPortableSandboxCommand } from './portable-sandbox-command'
import { spawnSync } from 'node:child_process'
import { existsSync, realpathSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { approvalPolicyScope, isReadOnlyShellCommand } from './approval-policy'
import { commandShell, commandShellArgs } from './cli-path'
import { commandApprovalReason } from './command-policy'
import { attachSandboxArtifacts, captureSandboxArtifactBaseline } from './sandbox-artifacts'
import { runSandboxProcess } from './sandbox-process'
import { installWorkspacePackages } from './package-install'
import type { SandboxProfileDefinition, SandboxProfileId, SandboxedCommandOptions, SandboxedCommandResult } from './sandbox-types'

export * from './sandbox-types'
export { buildSandboxArtifactBundle, captureSandboxArtifactBaseline, summarizeSandboxArtifactBundle } from './sandbox-artifacts'
export { sandboxCommandEnv } from './sandbox-process'

const UNSANDBOXED_ENV = 'VONTRA_ALLOW_UNSANDBOXED_COMMANDS'
let macSandboxAvailable: boolean | undefined

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
    description: 'Prepare npm dependencies without lifecycle scripts, then publish checked changes into the workspace.',
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

export function commandNeedsNetwork(command: string): boolean {
  return !!commandApprovalReason(command)
}

export function sandboxProfileForCommand(command: string): SandboxProfileId {
  if (isReadOnlyShellCommand(command)) return 'read-only'
  const classification = approvalPolicyScope('tool', 'run_command', { command }).classification
  return sandboxProfileForCommandClass(classification)
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

export function taskSandboxNetworkOverride(profile: SandboxProfileId, enabled: boolean | undefined): boolean | undefined {
  if (profile === 'package-install') return true
  if (profile === 'network-fetch') return enabled === true
  return undefined
}

export function canUseMacSandbox(): boolean {
  if (process.platform !== 'darwin' || !existsSync('/usr/bin/sandbox-exec')) return false
  if (macSandboxAvailable !== undefined) return macSandboxAvailable
  const probe = spawnSync('/usr/bin/sandbox-exec', ['-p', '(version 1)\n(allow default)', '/usr/bin/true'], {
    stdio: 'ignore',
    timeout: 2_000,
  })
  macSandboxAvailable = probe.status === 0
  return macSandboxAvailable
}

export async function runSandboxedCommand(command: string, options: SandboxedCommandOptions): Promise<SandboxedCommandResult> {
  const cwd = realpathSync(options.cwd)
  const profile = options.profile ?? sandboxProfileForCommand(command)
  const allowNetwork = sandboxProfileAllowsNetwork(profile, options.allowNetwork)
  const captureWrites = profile === 'package-install' || SANDBOX_PROFILES[profile].allowCwdWrite
  const baseline = captureSandboxArtifactBaseline(cwd, Date.now(), captureWrites)
  const macSandboxed = canUseMacSandbox()
  const linuxSandboxed = !macSandboxed && canUseLinuxSandbox()
  const windowsSandboxed = !macSandboxed && !linuxSandboxed && canUseWindowsSandbox()
  const sandboxed = macSandboxed || linuxSandboxed || windowsSandboxed
  if (!sandboxed && !allowUnsandboxedCommands()) {
    return attachSandboxArtifacts(command, cwd, {
      code: 126,
      stdout: '',
      stderr: `No supported command sandbox is available on this platform. Set ${UNSANDBOXED_ENV}=1 only for deliberate unsandboxed local development.`,
      sandboxed: false,
      profile,
      implementation: 'unsandboxed',
    }, baseline)
  }
  const run = (text: string, directory: string, isolatedWorkspace?: string) => {
    const argv = [commandShell(), ...commandShellArgs(), text]
    if (linuxSandboxed || windowsSandboxed) return runPortableSandboxCommand(argv, {
      ...options, cwd: directory, profile, allowNetwork,
      platform: linuxSandboxed ? 'linux' : 'win32',
      allowWrite: SANDBOX_PROFILES[profile].allowCwdWrite || !!isolatedWorkspace, privateInstall: !!isolatedWorkspace,
    })
    if (macSandboxed) argv.unshift(
      '/usr/bin/sandbox-exec', '-p', macSandboxProfile(directory, profile, [], allowNetwork, isolatedWorkspace),
    )
    return runSandboxProcess(argv, {
      cwd: directory,
      timeoutMs: options.timeoutMs,
      maxBuffer: options.maxBuffer,
      sandboxed,
      profile: sandboxed ? profile : 'unsandboxed',
      implementation: macSandboxed ? 'macos-sandbox-exec' : 'unsandboxed',
      signal: options.signal,
      privateTemp: isolatedWorkspace ? path.join(directory, '.npm-tmp') : undefined,
    })
  }
  const result = profile === 'package-install'
    ? await installWorkspacePackages(command, cwd, (input) => run(input.command, input.cwd, input.workspace), options)
    : await run(command, cwd)
  const captured = attachSandboxArtifacts(command, cwd, result, baseline)
  captured.changedPaths = [...new Set([...(result.changedPaths ?? []), ...(captured.changedPaths ?? [])])]
  return captured
}

export function macSandboxProfile(cwd: string, profileOrNetwork: SandboxProfileId | boolean = 'build-test', extraReadRoots: Array<string | null | undefined> = [], allowNetworkOverride?: boolean, isolatedWorkspace?: string): string {
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
    ...(!isolatedWorkspace ? [os.tmpdir(), process.env.TMPDIR, darwinUserTempDir()] : []),
    ...extraReadRoots,
  ])
  const writeRoots = uniqueExistingPaths([
    ...(definition.allowCwdWrite || isolatedWorkspace ? [cwd] : []),
    ...(!isolatedWorkspace ? [os.tmpdir(), process.env.TMPDIR, darwinUserTempDir()] : []),
  ])
  const readRules = readRoots.map((root) => `(subpath ${sbplString(root)})`).join('\n      ')
  const writeRules = writeRoots.map((root) => `(subpath ${sbplString(root)})`).join('\n      ')

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
${allowNetwork ? '(allow network*)' : ''}${isolatedWorkspace ? `
(deny file-read-data file-write* (subpath ${sbplString(isolatedWorkspace)}))
(allow file-read* file-write* (subpath ${sbplString(cwd)}))` : ''}`
}

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
  if (cachedDeveloperDir === undefined) {
    let dir: string | null
    try { dir = realpathSync('/var/db/xcode_select_link') } catch { dir = existsSync('/Library/Developer/CommandLineTools') ? '/Library/Developer/CommandLineTools' : null }
    const bundle = dir?.match(/^(.*?\.app)\/Contents\/Developer$/)
    cachedDeveloperDir = bundle ? bundle[1] : dir
  }
  return cachedDeveloperDir
}

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

function allowUnsandboxedCommands(): boolean {
  const resources = (process as NodeJS.Process & { resourcesPath?: string }).resourcesPath
  if (resources && existsSync(path.join(resources, 'app.asar'))) return false
  const value = process.env[UNSANDBOXED_ENV]?.toLowerCase()
  return value === '1' || value === 'true' || value === 'yes'
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
    } catch {  }
  }
  return out
}

export function sbplString(value: string): string {
  return JSON.stringify(value)
}
