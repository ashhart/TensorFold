import { chmodSync, closeSync, existsSync, mkdirSync, mkdtempSync, openSync, readdirSync, readSync, realpathSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { canUseMacSandbox, sbplString, uniqueExistingPaths } from './sandbox'
import type { Provider } from './types'

// Confinement for delegated CLIs in planning-only mode. Each run starts in an empty private
// per-thread folder under the OS temp dir (no username in the path, which CLIs report to their
// model as the working directory), with the CLI's own file, shell and plugin
// tools switched off by its flags, inside a macOS seatbelt profile that only lets it read
// system paths, its install, its auth/config dirs and the planning folder.
//
// Nesting: a seatbelt can't be applied inside another (sandbox_apply fails, exit 71). Codex
// wraps the commands it runs in its own seatbelt, so under this profile any command it still
// tries fails closed. That is intended: planning runs must not execute anything. It also
// means the app itself must not run inside an App Sandbox; canUseMacSandbox() probes that,
// and the guard refuses CLI planning runs when the probe fails.

export interface CliConfinement {
  planningDir: string
  tmpDir: string
  // Denied last, so the workspace stays unreadable even under an allowed install root.
  denyRoots?: string[]
}

export interface PlanningCommand {
  command: string
  args: string[]
  cwd: string | undefined
  env: NodeJS.ProcessEnv
}

const confinements = new WeakMap<Provider, CliConfinement>()

/** A copy of `provider` whose CLI spawns are confined to `confinement`. */
export function confineProvider(provider: Provider, confinement: CliConfinement): Provider {
  const copy = { ...provider }
  confinements.set(copy, confinement)
  return copy
}

export function cliConfinement(provider: Provider | null | undefined): CliConfinement | null {
  return provider ? confinements.get(provider) ?? null : null
}

export function planningSandboxAvailable(): boolean {
  return canUseMacSandbox()
}

const PLANNING_PREFIX = 'tf-plan-'
const threadDirs = new Map<string, CliConfinement>()
const ownDirs = new Set<string>()
let exitCleanup = false

/** The OS temp dir, e.g. /private/var/folders/xx/<random>/T: per user, without the username. */
export function planningRoot(): string {
  return realpathSync(os.tmpdir())
}

function processIsAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch (error: any) { return error?.code !== 'ESRCH' }
}

// Folders left by a crashed app run; the owner pid is in the name, as for Claude prompt files.
function removeStalePlanningDirs(root: string): void {
  let names: string[] = []
  try { names = readdirSync(root) } catch { return }
  for (const name of names) {
    const owner = new RegExp(`^${PLANNING_PREFIX}(\\d+)-`).exec(name)
    if (owner && Number(owner[1]) !== process.pid && !processIsAlive(Number(owner[1]))) rmSync(path.join(root, name), { recursive: true, force: true })
  }
}

/** A private (0700) planning folder: reused per thread for the app's lifetime, one-shot without a key. */
export function preparePlanningDir(key: string | null, root = planningRoot()): CliConfinement {
  const id = key ? `${root}\0${key}` : ''
  const cached = id ? threadDirs.get(id) : undefined
  if (cached && existsSync(cached.tmpDir)) return cached
  removeStalePlanningDirs(root)
  const planningDir = realpathSync(mkdtempSync(path.join(root, `${PLANNING_PREFIX}${process.pid}-`)))
  chmodSync(planningDir, 0o700)
  mkdirSync(path.join(planningDir, 'tmp'), { mode: 0o700 })
  const confinement = { planningDir, tmpDir: path.join(planningDir, 'tmp') }
  if (id) threadDirs.set(id, confinement)
  ownDirs.add(planningDir)
  if (!exitCleanup) {
    exitCleanup = true
    process.once('exit', () => { for (const dir of ownDirs) rmSync(dir, { recursive: true, force: true }) })
  }
  return confinement
}

export function removePlanningDir(confinement: CliConfinement, root = planningRoot()): void {
  let base: string
  try { base = realpathSync(root) } catch { return }
  if (path.dirname(confinement.planningDir) !== base || !path.basename(confinement.planningDir).startsWith(PLANNING_PREFIX)) return
  rmSync(confinement.planningDir, { recursive: true, force: true })
  ownDirs.delete(confinement.planningDir)
  for (const [id, cached] of threadDirs) if (cached.planningDir === confinement.planningDir) threadDirs.delete(id)
}

const SYSTEM_READ_ROOTS = [
  '/System', '/usr', '/bin', '/sbin', '/Library', '/dev',
  '/private/etc', '/private/var/db', '/private/var/select', '/private/var/run',
  '/opt/homebrew', '/usr/local',
]

// Codex feature switches; `-c` overrides tolerate names an older Codex doesn't know.
const CODEX_FEATURES_OFF = [
  'shell_tool', 'unified_exec', 'apps', 'plugins', 'multi_agent', 'browser_use', 'browser_use_external',
  'computer_use', 'in_app_browser', 'image_generation', 'view_image', 'hooks', 'shell_snapshot',
  'memories', 'skill_search', 'tool_suggest', 'workspace_dependencies', 'code_mode_host',
  // Keep the turn in this (sandboxed) process, never the shared unsandboxed daemon.
  'daemon_auto_start',
]
const CODEX_FLAGS = [
  ...CODEX_FEATURES_OFF.flatMap((feature) => ['-c', `features.${feature}=false`]),
  '-c', 'web_search="disabled"', '-c', 'include_apply_patch_tool=false',
]
const CLAUDE_FLAGS = ['--tools', '', '--strict-mcp-config', '--safe-mode', '--disable-slash-commands']
const PI_FLAGS = ['--no-tools', '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-context-files', '--no-approve']

/** The CLI's own switches for no file, shell, plugin or MCP tools, merged into its argv. */
export function planningCliArgs(transport: Provider['transport'], args: string[]): string[] {
  if (transport === 'codex-cli') {
    const at = args[0] === 'exec' || args[0] === 'app-server' ? 1 : 0
    return [...args.slice(0, at), ...CODEX_FLAGS, ...args.slice(at)]
  }
  if (transport === 'claude-cli') return [...CLAUDE_FLAGS, ...args]
  if (transport === 'pi-cli') {
    const kept = args.filter((arg, i) => arg !== '--tools' && args[i - 1] !== '--tools')
    return [...PI_FLAGS, ...kept]
  }
  if (transport === 'cursor-cli') return args.includes('--mode=ask') ? args : ['--mode=ask', ...args]
  return args
}

/** Auth and config folders the CLI reads and writes (sessions, token refresh). */
export function cliStateDirs(transport: Provider['transport'], env: NodeJS.ProcessEnv, home = os.homedir()): string[] {
  const under = (...parts: string[]) => path.join(home, ...parts)
  const dirs: Array<string | undefined> = [under('Library', 'Keychains')]
  if (transport === 'codex-cli') dirs.push(env.CODEX_HOME || under('.codex'))
  if (transport === 'claude-cli') dirs.push(env.CLAUDE_CONFIG_DIR || under('.claude'), env.CLAUDE_HOME)
  if (transport === 'pi-cli') dirs.push(env.PI_CODING_AGENT_DIR || under('.pi'))
  if (transport === 'cursor-cli') dirs.push(under('.cursor'), under('.config', 'cursor'))
  if (transport === 'gemini-cli') dirs.push(under('.gemini'))
  return dirs.filter((dir): dir is string => !!dir)
}

function onPath(name: string, env: NodeJS.ProcessEnv): string | null {
  if (name.includes('/')) return name
  for (const dir of String(env.PATH ?? '').split(':').filter(Boolean)) {
    const full = path.join(dir, name)
    if (existsSync(full)) return full
  }
  return null
}

function interpreter(file: string, env: NodeJS.ProcessEnv): string | null {
  let head = ''
  try {
    const fd = openSync(file, 'r')
    try {
      const buf = Buffer.alloc(256)
      head = buf.subarray(0, readSync(fd, buf, 0, 256, 0)).toString('utf8')
    } finally { closeSync(fd) }
  } catch { return null }
  const shebang = /^#!\s*(\S+)(?:\s+(?:-S\s+)?(\S+))?/.exec(head)
  if (!shebang) return null
  return shebang[1].endsWith('/env') && shebang[2] ? onPath(shebang[2], env) : shebang[1]
}

/** Install prefixes for the binary, its symlink target and any script interpreter. */
export function cliInstallRoots(bin: string, env: NodeJS.ProcessEnv, home = os.homedir()): string[] {
  const roots: string[] = []
  const tooBroad = (dir: string) => dir === '/' || dir === home || home.startsWith(`${dir}/`) || path.dirname(dir) === home
  // Both the link (an npm or nvm prefix) and its target (a versioned install) count.
  const add = (file: string | null) => {
    if (!file) return
    const candidates = new Set([path.resolve(file)])
    try { candidates.add(realpathSync(file)) } catch { /* missing binary: spawn reports it */ }
    for (const p of candidates) {
      const prefix = path.dirname(path.dirname(p))
      roots.push(tooBroad(prefix) ? (tooBroad(path.dirname(p)) ? p : path.dirname(p)) : prefix)
    }
  }
  const resolved = onPath(bin, env)
  add(resolved)
  if (resolved && existsSync(resolved)) add(interpreter(realpathSync(resolved), env))
  return roots
}

/** Seatbelt profile: everything but file access is allowed; reads and writes are allowlisted. */
export function planningSandboxProfile(input: { readRoots: string[]; writeRoots: string[]; readFiles?: string[]; denyRoots?: string[]; home?: string }): string {
  const reads = uniqueExistingPaths(input.readRoots).map((root) => `(subpath ${sbplString(root)})`)
  const files = uniqueExistingPaths(input.readFiles ?? []).map((file) => `(literal ${sbplString(file)})`)
  const writes = uniqueExistingPaths(input.writeRoots).map((root) => `(subpath ${sbplString(root)})`)
  const denies = uniqueExistingPaths(input.denyRoots ?? []).map((root) => `(subpath ${sbplString(root)})`)
  // Claude keeps its account state in ~/.claude.json and rewrites it atomically beside it.
  const home = realpathSync(input.home ?? os.homedir())
  const claudeState = `(regex ${sbplString(`^${home.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/\\.claude\\.json`)})`
  return `(version 1)
(allow default)
(deny file-read* file-write*)
(allow file-read-metadata)
(allow file-read*
      (literal "/")
      ${[...reads, ...files, claudeState].join('\n      ')})
(allow file-write*
      ${[...writes, claudeState].join('\n      ')})
${denies.length ? `(deny file-read* file-write*\n      ${denies.join('\n      ')})\n` : ''}`
}

/** The seatbelt profile for one confined CLI spawn. */
export function planningProfileFor(input: { transport: Provider['transport']; command: string; args: string[]; env: NodeJS.ProcessEnv; confinement: CliConfinement; home?: string }): string {
  const home = input.home ?? os.homedir()
  const stateDirs = cliStateDirs(input.transport, input.env, home)
  const needed = uniqueExistingPaths([...stateDirs, input.confinement.planningDir])
  const settingsAt = input.args.indexOf('--settings')
  const settingsFile = settingsAt >= 0 && !String(input.args[settingsAt + 1] ?? '').trim().startsWith('{') ? input.args[settingsAt + 1] : null
  return planningSandboxProfile({
    readRoots: [...SYSTEM_READ_ROOTS, ...cliInstallRoots(input.command, input.env, home), ...stateDirs, input.confinement.planningDir],
    readFiles: settingsFile ? [settingsFile] : [],
    writeRoots: [...stateDirs, input.confinement.planningDir, '/dev'],
    // A workspace that holds the CLI's own dirs (e.g. the home folder) is already outside
    // the allowlist; denying it outright would also cut off auth.
    denyRoots: uniqueExistingPaths(input.confinement.denyRoots ?? []).filter((deny) => !needed.some((dir) => dir === deny || dir.startsWith(`${deny}/`))),
    home,
  })
}

/** Wrap a CLI spawn in the planning profile when `provider` is confined; else pass through. */
export function planningCommand(provider: Provider, command: string, args: string[], base: { cwd?: string; env: NodeJS.ProcessEnv; home?: string }): PlanningCommand {
  const confinement = cliConfinement(provider)
  if (!confinement) return { command, args, cwd: base.cwd, env: base.env }
  if (!planningSandboxAvailable()) throw new Error('Planning-only mode needs the macOS sandbox (sandbox-exec), which is not available here, so nothing was sent.')
  // Claude Code keeps task files under /tmp/claude-<uid> unless CLAUDE_CODE_TMPDIR moves them.
  const tmp = confinement.tmpDir
  const env: NodeJS.ProcessEnv = { ...base.env, TMPDIR: tmp, TMP: tmp, TEMP: tmp, CLAUDE_CODE_TMPDIR: tmp, PWD: confinement.planningDir }
  const profile = planningProfileFor({ transport: provider.transport, command, args, env, confinement, home: base.home })
  return {
    command: '/usr/bin/sandbox-exec',
    args: ['-p', profile, onPath(command, env) ?? command, ...planningCliArgs(provider.transport, args)],
    cwd: confinement.planningDir,
    env,
  }
}
