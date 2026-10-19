import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { copyFileSync, existsSync, lstatSync, mkdirSync, mkdtempSync, readlinkSync, readdirSync, readFileSync, realpathSync, rmSync, statSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { StringDecoder } from 'node:string_decoder'
import { approvalPolicyScope, isReadOnlyShellCommand } from './approval-policy'
import { commandShell, commandShellArgs } from './cli-path'
import { canUseLinuxSandbox, linuxSandboxArgv, linuxSandboxBinary } from './linux-sandbox'
import { commandApprovalReason } from './command-policy'
import { hostGitEnv } from './host-git-env'
import { redactCommandStreams } from './command-output-redaction'
import { SECRET_MASK } from './security'
import { redactSecretValues } from './secret-detection'
import type { ToolArtifactSummary } from './types'

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

export const SANDBOX_ARTIFACT_SCHEMA = 'vontra.sandbox-artifacts/1'

export interface SandboxFileArtifact {
  path: string
  bytes: number
  sha256: string
  text?: string
}

export interface SandboxLogArtifact {
  kind: 'stdout' | 'stderr'
  bytes: number
  preview: string
}

export interface SandboxArtifactDiff {
  path: string
  status: 'added' | 'changed' | 'deleted'
  beforeSha256?: string
  afterSha256?: string
  bytes?: number
}

export interface SandboxArtifactBaseline {
  inputSnapshot: SandboxFileArtifact[]
  outputFiles: SandboxFileArtifact[]
  capturedAt: number
  /** Git tracked+untracked file fingerprints used only for mutation ownership. */
  workspaceSnapshot?: SandboxWorkspaceSnapshot
}

export interface SandboxWorkspaceSnapshot {
  root: string
  files: Array<{ path: string; fingerprint: string }>
}

export interface SandboxArtifactBundle {
  schema: typeof SANDBOX_ARTIFACT_SCHEMA
  id: string
  command: string
  cwd: string
  profile: SandboxProfileId | 'unsandboxed'
  implementation: 'macos-sandbox-exec' | 'linux-bwrap' | 'unsandboxed'
  inputSnapshot: SandboxFileArtifact[]
  outputFiles: SandboxFileArtifact[]
  screenshots: SandboxFileArtifact[]
  logs: SandboxLogArtifact[]
  diffs: SandboxArtifactDiff[]
  patch: string | null
  createdAt: number
}

export interface SandboxedCommandResult {
  code: number
  stdout: string
  stderr: string
  sandboxed: boolean
  profile: SandboxProfileId | 'unsandboxed'
  implementation: 'macos-sandbox-exec' | 'linux-bwrap' | 'unsandboxed'
  /** Number of value-level secrets removed from stdout/stderr before this result left the sandbox. */
  redactionCount?: number
  artifacts?: SandboxArtifactBundle
  /** Every git tracked/untracked workspace path changed by the command, even on failure. */
  changedPaths?: string[]
}

export interface SandboxedCommandOptions {
  cwd: string
  timeoutMs?: number
  maxBuffer?: number
  allowNetwork?: boolean
  profile?: SandboxProfileId
  signal?: AbortSignal
}

const UNSANDBOXED_ENV = 'VONTRA_ALLOW_UNSANDBOXED_COMMANDS'
let macSandboxAvailable: boolean | undefined
const INPUT_SNAPSHOT_PATHS = [
  'package.json',
  'package-lock.json',
  'pnpm-lock.yaml',
  'yarn.lock',
  'bun.lockb',
  'tsconfig.json',
  'vite.config.ts',
]
const OUTPUT_ARTIFACT_PATHS = [
  '.vontra-artifacts',
  'artifacts',
  'screenshots',
  'test-results',
  'playwright-report',
  'coverage',
  'dist',
]
const PACKAGE_INSTALL_INPUT_PATHS = [
  'package.json',
  'package-lock.json',
  'npm-shrinkwrap.json',
  'pnpm-lock.yaml',
  'yarn.lock',
  'bun.lockb',
  'pyproject.toml',
  'requirements.txt',
  'requirements-dev.txt',
  'poetry.lock',
  'uv.lock',
  'Pipfile',
  'Pipfile.lock',
]
const ARTIFACT_SKIP_DIRS = new Set(['.git', 'node_modules'])
const MAX_ARTIFACT_FILES = 120
const MAX_HASH_BYTES = 10 * 1024 * 1024
const SCREENSHOT_RE = /\.(?:png|jpe?g|webp|gif)$/i

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

/**
 * The global task-network toggle is only meaningful for a command already
 * classified as network-bearing. It must never widen build-test/untrusted-code
 * profiles merely because an opaque interpreter invocation slipped classification.
 */
export function taskSandboxNetworkOverride(profile: SandboxProfileId, enabled: boolean | undefined): boolean | undefined {
  if (profile === 'package-install') return true
  if (profile === 'network-fetch') return enabled === true
  return undefined
}

export function captureSandboxArtifactBaseline(cwd: string, now = Date.now(), captureWorkspace = true): SandboxArtifactBaseline {
  const root = realpathSync(cwd)
  return {
    inputSnapshot: collectArtifactFiles(root, INPUT_SNAPSHOT_PATHS),
    outputFiles: collectArtifactFiles(root, OUTPUT_ARTIFACT_PATHS),
    capturedAt: now,
    ...(captureWorkspace ? { workspaceSnapshot: captureWorkspaceSnapshot(root) ?? undefined } : {}),
  }
}

export function buildSandboxArtifactBundle(input: {
  cwd: string
  command: string
  result: Pick<SandboxedCommandResult, 'stdout' | 'stderr' | 'profile' | 'implementation'>
  baseline?: SandboxArtifactBaseline
  now?: number
}): SandboxArtifactBundle {
  const now = input.now ?? Date.now()
  const root = realpathSync(input.cwd)
  const baseline = input.baseline ?? captureSandboxArtifactBaseline(root, now)
  const outputFiles = collectArtifactFiles(root, OUTPUT_ARTIFACT_PATHS)
  const screenshots = outputFiles.filter((file) => SCREENSHOT_RE.test(file.path))
  const diffs = diffArtifactFiles(baseline.outputFiles, outputFiles)
  const commandStreams = redactCommandStreams(input.result.stdout, input.result.stderr)
  const logs = ([
    sandboxLogArtifact('stdout', commandStreams.stdout),
    sandboxLogArtifact('stderr', commandStreams.stderr),
  ] as SandboxLogArtifact[]).filter((log) => log.bytes > 0)

  return {
    schema: SANDBOX_ARTIFACT_SCHEMA,
    id: `sbox-${Math.max(0, Math.floor(now)).toString(36)}`,
    command: input.command,
    cwd: root,
    profile: input.result.profile,
    implementation: input.result.implementation,
    inputSnapshot: baseline.inputSnapshot,
    outputFiles,
    screenshots,
    logs,
    diffs,
    patch: buildArtifactPatch(baseline.outputFiles, outputFiles, diffs),
    createdAt: now,
  }
}

const ARTIFACT_SUMMARY_PATHS_MAX = 20

/** The timeline's compact record of a bundle: counts and paths, never file or log contents. */
export function summarizeSandboxArtifactBundle(bundle: SandboxArtifactBundle): ToolArtifactSummary {
  return {
    id: bundle.id,
    profile: bundle.profile,
    implementation: bundle.implementation,
    inputFiles: bundle.inputSnapshot.length,
    outputFiles: bundle.outputFiles.length,
    screenshots: bundle.screenshots.slice(0, ARTIFACT_SUMMARY_PATHS_MAX).map((file) => file.path),
    diffs: bundle.diffs.slice(0, ARTIFACT_SUMMARY_PATHS_MAX).map((diff) => ({ path: diff.path, status: diff.status })),
    patch: !!bundle.patch,
    logBytes: bundle.logs.reduce((sum, log) => sum + log.bytes, 0),
  }
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
  const hostCwd = realpathSync(options.cwd)
  const profile = options.profile ?? sandboxProfileForCommand(command)
  const allowNetwork = sandboxProfileAllowsNetwork(profile, options.allowNetwork)
  const cwd = prepareSandboxExecutionCwd(hostCwd, profile)
  // package-install runs in a throwaway temp dir; remove it once artifacts are captured so we
  // don't orphan a (potentially node_modules-sized) directory per install. The host cwd is never
  // deleted — only a sandbox-owned scratch dir that differs from it.
  const ephemeralCwd = cwd === hostCwd ? null : cwd
  try {
    const baseline = captureSandboxArtifactBaseline(cwd, Date.now(), SANDBOX_PROFILES[profile].allowCwdWrite)
    if (canUseMacSandbox()) {
      const result = await runShell(['/usr/bin/sandbox-exec', '-p', macSandboxProfile(cwd, profile, [], allowNetwork), commandShell(), ...commandShellArgs(), command], {
        cwd,
        timeoutMs: options.timeoutMs,
        maxBuffer: options.maxBuffer,
        sandboxed: true,
        profile,
        implementation: 'macos-sandbox-exec',
        signal: options.signal,
      })
      return attachSandboxArtifacts(command, cwd, result, baseline)
    }
    if (canUseLinuxSandbox()) {
      const definition = SANDBOX_PROFILES[profile]
      const bwrapArgv = linuxSandboxArgv(cwd, { allowCwdWrite: definition.allowCwdWrite, allowNetwork })
      const result = await runShell([linuxSandboxBinary()!, ...bwrapArgv, commandShell(), ...commandShellArgs(), command], {
        cwd,
        timeoutMs: options.timeoutMs,
        maxBuffer: options.maxBuffer,
        sandboxed: true,
        profile,
        implementation: 'linux-bwrap',
        signal: options.signal,
      })
      return attachSandboxArtifacts(command, cwd, result, baseline)
    }
    if (!allowUnsandboxedCommands()) {
      return attachSandboxArtifacts(command, cwd, {
        code: 126,
        stdout: '',
        stderr: `No supported command sandbox is available on this platform. Set ${UNSANDBOXED_ENV}=1 only for deliberate unsandboxed local development.`,
        sandboxed: false,
        profile,
        implementation: 'unsandboxed',
      }, baseline)
    }
    const result = await runShell([commandShell(), ...commandShellArgs(), command], {
      cwd,
      timeoutMs: options.timeoutMs,
      maxBuffer: options.maxBuffer,
      sandboxed: false,
      profile: 'unsandboxed',
      implementation: 'unsandboxed',
      signal: options.signal,
    })
    return attachSandboxArtifacts(command, cwd, result, baseline)
  } finally {
    if (ephemeralCwd) {
      try { rmSync(ephemeralCwd, { recursive: true, force: true }) } catch { /* best-effort cleanup */ }
    }
  }
}

function prepareSandboxExecutionCwd(hostCwd: string, profile: SandboxProfileId): string {
  if (profile !== 'package-install') return hostCwd
  const sandboxCwd = mkdtempSync(path.join(os.tmpdir(), 'vontra-package-install-'))
  for (const relPath of PACKAGE_INSTALL_INPUT_PATHS) {
    const source = path.join(hostCwd, relPath)
    if (!existsSync(source)) continue
    try {
      const stat = statSync(source)
      if (!stat.isFile()) continue
      const destination = path.join(sandboxCwd, relPath)
      mkdirSync(path.dirname(destination), { recursive: true })
      copyFileSync(source, destination)
    } catch { /* ignore unreadable optional package inputs */ }
  }
  return realpathSync(sandboxCwd)
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
  // For no-network profiles, `(allow mach-lookup)` alone would still let a sandboxed child reach
  // network-resolution daemons (mDNSResponder/DNS, the system proxy/DNS configuration services)
  // over the bootstrap port and perform name resolution / proxied network actions, partially
  // undermining the "network denied" guarantee. SBPL is last-match-wins, so re-deny just those
  // network-adjacent mach services after the broad allow — general mach-lookup (needed by dyld,
  // compilers and language runtimes) stays open so build/test/read-only runs don't break.
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

// A writable workspace still may not touch git's control files. Host-side git (status,
// diff, snapshots) runs outside the sandbox and would execute a planted core.fsmonitor,
// filter driver or hook, so a sandboxed `npm test` could otherwise escape through
// .git/config. Denies the `.git` entry itself (no swapping or creating it) and its
// config, hooks and info at any depth (nested repos, submodules). SBPL is last-match-wins.
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

// The developer dir `xcode-select` points at (Xcode.app or the Command Line Tools), read
// from its link rather than spawning xcode-select on every sandboxed command. Inside Xcode.app
// the shims also load SharedFrameworks (DVTSystemPrerequisites), so the whole bundle is read.
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

// xcrun keeps its lookup cache in the per-user temp dir (confstr DARWIN_USER_TEMP_DIR), which is
// TMPDIR for a Finder launch but not when the app starts with a scrubbed environment.
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

function runShell(argv: string[], opts: { cwd: string; timeoutMs?: number; maxBuffer?: number; sandboxed: boolean; profile: SandboxedCommandResult['profile']; implementation: SandboxedCommandResult['implementation']; signal?: AbortSignal }): Promise<SandboxedCommandResult> {
  return new Promise((resolve) => {
    const [cmd, ...args] = argv
    const child = spawn(cmd, args, {
      cwd: opts.cwd,
      env: sandboxCommandEnv(opts.cwd),
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
    })
    const started = Date.now()
    const maxBuffer = opts.maxBuffer ?? 8 * 1024 * 1024
    let stdout = ''
    let stderr = ''
    let killed = false
    let aborted = false
    let overflowed = false
    let escalation: ReturnType<typeof setTimeout> | undefined
    const signalTree = (signal: NodeJS.Signals) => {
      try {
        if (process.platform !== 'win32' && child.pid) process.kill(-child.pid, signal)
        else child.kill(signal)
      } catch {
        try { child.kill(signal) } catch { /* already exited */ }
      }
    }
    const terminate = () => {
      if (escalation) clearTimeout(escalation)
      signalTree('SIGTERM')
      escalation = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) {
          signalTree('SIGKILL')
        }
      }, 1000)
      escalation.unref()
    }
    const timer = setTimeout(() => {
      killed = true
      terminate()
    }, opts.timeoutMs ?? 120_000)
    const onAbort = () => {
      if (aborted) return
      aborted = true
      clearTimeout(timer)
      terminate()
    }
    opts.signal?.addEventListener('abort', onAbort, { once: true })
    if (opts.signal?.aborted) onAbort()
    // Per-stream decoders buffer partial multi-byte sequences across chunk boundaries, so emoji/CJK
    // split across two 'data' events don't decode to U+FFFD replacement chars.
    const outDecoder = new StringDecoder('utf8')
    const errDecoder = new StringDecoder('utf8')
    const append = (kind: 'stdout' | 'stderr', chunk: Buffer) => {
      const text = (kind === 'stdout' ? outDecoder : errDecoder).write(chunk)
      if (kind === 'stdout') stdout = (stdout + text).slice(0, maxBuffer)
      else stderr = (stderr + text).slice(0, maxBuffer)
      if (stdout.length + stderr.length >= maxBuffer && !overflowed) {
        overflowed = true
        // Cancel the unrelated timeout timer so it doesn't double-fire later, then run the same
        // SIGTERM-now/SIGKILL-after-grace-period escalation the timeout path uses, so a child that
        // traps/ignores SIGTERM doesn't linger until the original (possibly much longer) timeout.
        // Note: `child.killed` reflects whether a signal was *sent*, not whether the process has
        // actually exited yet — checking `exitCode`/`signalCode` (only set once 'exit' fires) is
        // what correctly detects a child that is still alive after ignoring SIGTERM.
        clearTimeout(timer)
        terminate()
      }
    }
    child.stdout.on('data', (chunk) => append('stdout', chunk))
    child.stderr.on('data', (chunk) => append('stderr', chunk))
    child.on('error', (error) => {
      clearTimeout(timer)
      if (escalation) clearTimeout(escalation)
      opts.signal?.removeEventListener('abort', onAbort)
      resolve({ code: 127, stdout, stderr: error.message, sandboxed: opts.sandboxed, profile: opts.profile, implementation: opts.implementation })
    })
    child.on('close', (code, signal) => {
      clearTimeout(timer)
      if (escalation) clearTimeout(escalation)
      opts.signal?.removeEventListener('abort', onAbort)
      const timed = killed && !aborted ? `Timed out after ${Math.round((Date.now() - started) / 1000)}s.` : ''
      const cancelled = aborted ? 'Cancelled by the user.' : ''
      const sig = signal ? `Terminated by ${signal}.` : ''
      resolve({
        code: code ?? (aborted ? 130 : killed ? 124 : 1),
        stdout,
        stderr: [stderr.trim(), cancelled, timed, sig].filter(Boolean).join('\n'),
        sandboxed: opts.sandboxed,
        profile: opts.profile,
        implementation: opts.implementation,
      })
    })
  })
}

function attachSandboxArtifacts(command: string, cwd: string, result: SandboxedCommandResult, baseline: SandboxArtifactBaseline): SandboxedCommandResult {
  const streams = redactCommandStreams(result.stdout, result.stderr)
  const sanitized = { ...result, ...streams }
  return {
    ...sanitized,
    artifacts: buildSandboxArtifactBundle({ cwd, command, result: sanitized, baseline }),
    changedPaths: changedWorkspacePaths(cwd, baseline.workspaceSnapshot),
  }
}

function gitOutput(args: string[], cwd: string): Buffer | null {
  const result = spawnSync('git', args, {
    cwd,
    encoding: 'buffer',
    maxBuffer: 32 * 1024 * 1024,
    timeout: 20_000,
    env: hostGitEnv({ ...process.env, GIT_OPTIONAL_LOCKS: '0' }),
  })
  return result.status === 0 && Buffer.isBuffer(result.stdout) ? result.stdout : null
}

function captureWorkspaceSnapshot(cwd: string): SandboxWorkspaceSnapshot | null {
  const rootRaw = gitOutput(['rev-parse', '--show-toplevel'], cwd)
  if (!rootRaw) return null
  const rootText = rootRaw.toString('utf8').trim()
  if (!rootText) return null
  let root: string
  try { root = realpathSync(rootText) } catch { return null }
  const listed = gitOutput(['ls-files', '-c', '-o', '--exclude-standard', '-z'], root)
  if (!listed) return null
  const files: Array<{ path: string; fingerprint: string }> = []
  for (const rel of listed.toString('utf8').split('\0').filter(Boolean)) {
    if (rel.startsWith('/') || rel.split('/').includes('..')) continue
    const full = path.join(root, ...rel.split('/'))
    let stat
    try { stat = lstatSync(full) } catch { continue }
    let fingerprint: string
    if (stat.isSymbolicLink()) {
      try { fingerprint = `link:${workspaceContentFingerprint(readlinkSync(full))}` } catch { continue }
    } else if (stat.isFile()) {
      try {
        fingerprint = stat.size <= MAX_HASH_BYTES
          ? `file:${workspaceContentFingerprint(readFileSync(full))}`
          : `large:${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`
      } catch { continue }
    } else {
      continue
    }
    files.push({ path: rel.replace(/\\/g, '/'), fingerprint })
  }
  return { root, files: files.sort((a, b) => a.path.localeCompare(b.path)) }
}

function workspaceContentFingerprint(content: string | Buffer): string {
  return createHash('sha256').update(content).digest('hex')
}

function changedWorkspacePaths(cwd: string, before: SandboxWorkspaceSnapshot | undefined): string[] {
  if (!before) return []
  const after = captureWorkspaceSnapshot(before.root)
  if (!after || after.root !== before.root) return []
  const previous = new Map(before.files.map((file) => [file.path, file.fingerprint]))
  const current = new Map(after.files.map((file) => [file.path, file.fingerprint]))
  const changed = new Set<string>()
  for (const [file, fingerprint] of current) if (previous.get(file) !== fingerprint) changed.add(file)
  for (const file of previous.keys()) if (!current.has(file)) changed.add(file)
  return [...changed].sort().map((repoPath) => {
    const relative = path.relative(cwd, path.join(before.root, ...repoPath.split('/')))
    return relative.split(path.sep).join('/') || '.'
  })
}

function collectArtifactFiles(cwd: string, roots: string[]): SandboxFileArtifact[] {
  const out: SandboxFileArtifact[] = []
  for (const root of roots) {
    if (out.length >= MAX_ARTIFACT_FILES) break
    collectArtifactPath(cwd, path.join(cwd, root), out)
  }
  return out.sort((a, b) => a.path.localeCompare(b.path))
}

function collectArtifactPath(cwd: string, fullPath: string, out: SandboxFileArtifact[]): void {
  if (out.length >= MAX_ARTIFACT_FILES) return
  let stat
  try { stat = statSync(fullPath) } catch { return }
  if (stat.isDirectory()) {
    const name = path.basename(fullPath)
    if (ARTIFACT_SKIP_DIRS.has(name)) return
    let entries
    try { entries = readdirSync(fullPath, { withFileTypes: true }) } catch { return }
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (out.length >= MAX_ARTIFACT_FILES) break
      collectArtifactPath(cwd, path.join(fullPath, entry.name), out)
    }
    return
  }
  if (!stat.isFile()) return
  const relPath = normalizeArtifactPath(cwd, fullPath)
  if (!relPath) return
  out.push({
    path: relPath,
    bytes: stat.size,
    sha256: hashArtifactFile(fullPath, stat.size),
    ...artifactText(fullPath, stat.size),
  })
}

function normalizeArtifactPath(cwd: string, fullPath: string): string | null {
  const relative = path.relative(cwd, fullPath)
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return null
  return relative.split(path.sep).join('/')
}

function hashArtifactFile(fullPath: string, bytes: number): string {
  const hash = createHash('sha256')
  if (bytes > MAX_HASH_BYTES) {
    hash.update(`large:${bytes}:${statSync(fullPath).mtimeMs}`)
  } else {
    hash.update(readFileSync(fullPath))
  }
  return hash.digest('hex')
}

function artifactText(fullPath: string, bytes: number): { text?: string } {
  if (bytes > 200_000) return {}
  let buf
  try { buf = readFileSync(fullPath) } catch { return {} }
  if (buf.includes(0)) return {}
  const { text } = redactSecretValues(buf.toString('utf8'), SECRET_MASK)
  return { text }
}

function sandboxLogArtifact(kind: SandboxLogArtifact['kind'], text: string): SandboxLogArtifact {
  return {
    kind,
    bytes: Buffer.byteLength(text),
    preview: text.replace(/\s+/g, ' ').trim().slice(0, 400),
  }
}

function diffArtifactFiles(before: SandboxFileArtifact[], after: SandboxFileArtifact[]): SandboxArtifactDiff[] {
  const previous = new Map(before.map((file) => [file.path, file]))
  const current = new Map(after.map((file) => [file.path, file]))
  const diffs: SandboxArtifactDiff[] = []
  for (const file of after) {
    const old = previous.get(file.path)
    if (!old) {
      diffs.push({ path: file.path, status: 'added', afterSha256: file.sha256, bytes: file.bytes })
    } else if (old.sha256 !== file.sha256 || old.bytes !== file.bytes) {
      diffs.push({ path: file.path, status: 'changed', beforeSha256: old.sha256, afterSha256: file.sha256, bytes: file.bytes })
    }
  }
  for (const file of before) {
    if (!current.has(file.path)) diffs.push({ path: file.path, status: 'deleted', beforeSha256: file.sha256, bytes: file.bytes })
  }
  return diffs.sort((a, b) => a.path.localeCompare(b.path))
}

function buildArtifactPatch(before: SandboxFileArtifact[], after: SandboxFileArtifact[], diffs: SandboxArtifactDiff[]): string | null {
  const previous = new Map(before.map((file) => [file.path, file]))
  const current = new Map(after.map((file) => [file.path, file]))
  const patches: string[] = []
  for (const diff of diffs) {
    const oldFile = previous.get(diff.path)
    const newFile = current.get(diff.path)
    const oldText = diff.status === 'added' ? '' : oldFile?.text
    const newText = diff.status === 'deleted' ? '' : newFile?.text
    if (oldText == null || newText == null) continue
    patches.push(formatArtifactPatch(diff.path, diff.status, oldText, newText))
  }
  return patches.length ? patches.join('\n') : null
}

function formatArtifactPatch(pathName: string, status: SandboxArtifactDiff['status'], oldText: string, newText: string): string {
  const oldLines = patchLines(oldText)
  const newLines = patchLines(newText)
  return [
    `diff --git a/${pathName} b/${pathName}`,
    `--- ${status === 'added' ? '/dev/null' : `a/${pathName}`}`,
    `+++ ${status === 'deleted' ? '/dev/null' : `b/${pathName}`}`,
    '@@',
    ...oldLines.map((line) => `-${line}`),
    ...newLines.map((line) => `+${line}`),
  ].join('\n')
}

function patchLines(text: string): string[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  if (lines.at(-1) === '') lines.pop()
  return lines
}

export function sandboxCommandEnv(cwd: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    PATH: process.env.PATH || '/usr/bin:/bin:/usr/sbin:/sbin',
    SHELL: commandShell(),
    HOME: cwd,
    TMPDIR: process.env.TMPDIR || os.tmpdir(),
    LANG: process.env.LANG || 'C.UTF-8',
    LC_ALL: process.env.LC_ALL || process.env.LANG || 'C.UTF-8',
  }
  return env
}

function allowUnsandboxedCommands(): boolean {
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
    } catch { /* ignore missing roots */ }
  }
  return out
}

export function sbplString(value: string): string {
  return JSON.stringify(value)
}
