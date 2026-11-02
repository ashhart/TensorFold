import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Dirs where user-installed CLIs commonly live but the GUI-launched PATH omits.
export function knownBinDirs(home = os.homedir(), platform = process.platform, env: NodeJS.ProcessEnv = process.env): string[] {
  if (platform === 'win32') {
    const dirs = [
      path.win32.join(env.APPDATA || path.win32.join(home, 'AppData', 'Roaming'), 'npm'),
      path.win32.join(env.LOCALAPPDATA || path.win32.join(home, 'AppData', 'Local'), 'Microsoft', 'WinGet', 'Links'),
      path.win32.join(home, '.local', 'bin'), path.win32.join(home, '.cargo', 'bin'),
      path.win32.join(home, 'scoop', 'shims'), path.win32.join(home, '.volta', 'bin'),
      path.win32.join(env.ProgramFiles || 'C:\\Program Files', 'nodejs'),
      path.win32.join(env.SystemRoot || 'C:\\Windows', 'System32'), env.SystemRoot || 'C:\\Windows',
    ]
    return dirs
  }
  return [
    '/opt/homebrew/bin', '/opt/homebrew/sbin', '/usr/local/bin',
    `${home}/.local/bin`, `${home}/.bun/bin`, `${home}/.cargo/bin`,
    `${home}/.deno/bin`, `${home}/.npm-global/bin`, `${home}/.volta/bin`,
    `${home}/.codex/bin`, `${home}/.claude/bin`,
    '/usr/bin', '/bin', '/usr/sbin', '/sbin',
  ]
}

// Merge PATH-style `:`-separated dir lists, de-duplicating while preserving first-seen order. Pure.
export function computeAugmentedPath(...lists: Array<string | null | undefined>): string {
  return mergeCliPaths(process.platform, ...lists)
}

export function mergeCliPaths(platform: string, ...lists: Array<string | null | undefined>): string {
  const separator = platform === 'win32' ? ';' : ':'
  const seen = new Set<string>()
  const out: string[] = []
  for (const list of lists) {
    for (const dir of (list || '').split(separator)) {
      const d = dir.trim().replace(/^"(.*)"$/, '$1')
      const key = platform === 'win32' ? d.toLowerCase().replace(/[\\/]+$/, '') : d
      if (d && !seen.has(key)) { seen.add(key); out.push(d) }
    }
  }
  return out.join(separator)
}

// Capture the PATH a login+interactive shell would have (picks up brew shellenv in ~/.zprofile etc).
export function loginShellPath(): string {
  if (process.platform === 'win32') return ''
  const shell = process.env.SHELL || '/bin/zsh'
  try {
    const res = spawnSync(shell, ['-lic', 'printf "%s" "$PATH"'], { encoding: 'utf8', timeout: 5000 })
    return res.status === 0 && typeof res.stdout === 'string' ? res.stdout.trim() : ''
  } catch {
    return ''
  }
}

let cachedCommandShell: string | undefined

// The shell that runs agent commands: zsh where present (macOS behavior), then bash/sh for Linux.
export function commandShell(): string {
  if (cachedCommandShell) return cachedCommandShell
  if (process.platform === 'win32') return (cachedCommandShell = path.win32.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'cmd.exe'))
  for (const candidate of ['/bin/zsh', '/bin/bash', '/bin/sh', process.env.SHELL]) {
    if (candidate && existsSync(candidate)) return (cachedCommandShell = candidate)
  }
  return (cachedCommandShell = '/bin/sh')
}

// zsh keeps the login shell for macOS PATH parity; bash/sh run plain, as a login bash re-clobbers PATH.
export function commandShellArgs(): string[] {
  if (process.platform === 'win32') return ['/d', '/s', '/c']
  return commandShell().endsWith('zsh') ? ['-lc'] : ['-c']
}

export function ensureCliPath(): void {
  const delimiter = path.delimiter
  process.env.PATH = computeAugmentedPath(process.env.PATH, loginShellPath(), knownBinDirs().join(delimiter))
}

export function resolveCliBinary(
  explicit: string | undefined | null,
  names: string[],
  opts: { dirs?: string[]; exists?: (p: string) => boolean; platform?: NodeJS.Platform; env?: NodeJS.ProcessEnv } = {},
): string {
  const cli = explicit?.trim()
  if (cli) return cli
  const platform = opts.platform ?? process.platform
  const env = opts.env ?? process.env
  const dirs = opts.dirs ?? [...(env.PATH ?? '').split(platform === 'win32' ? ';' : ':').filter(Boolean), ...knownBinDirs(os.homedir(), platform, env)]
  const exists = opts.exists ?? existsSync
  for (const name of names) {
    for (const dir of dirs) {
      const extensions = platform === 'win32' && !path.win32.extname(name)
        ? ['.exe', '.com', '.cmd', '.bat', ''] : ['']
      for (const extension of extensions) {
        const full = platform === 'win32' ? path.win32.join(dir, name + extension) : `${dir}/${name}`
        if (exists(full)) return full
      }
    }
  }
  return names[0] ?? ''
}

const SAFE_ARGV_ID_RE = /^[A-Za-z0-9_.@:/-]{1,256}$/
export function isSafeArgvId(value: string | null | undefined): boolean {
  return !!value && !value.startsWith('-') && SAFE_ARGV_ID_RE.test(value)
}
