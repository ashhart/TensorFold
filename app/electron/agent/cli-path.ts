import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import os from 'node:os'

// A Finder/dock-launched macOS app inherits a stripped PATH (/usr/bin:/bin:/usr/sbin:/sbin) that
// omits Homebrew (/opt/homebrew/bin) and ~/.local/bin, so `spawn('codex')` / `spawn('claude')` (and
// git, npm, MCP stdio servers, …) fail with ENOENT even though they work in a terminal. These
// helpers recover the real PATH and resolve a CLI's absolute path so spawns work in the packaged app.

/** Dirs where user-installed CLIs commonly live but the GUI-launched PATH omits. */
export function knownBinDirs(home = os.homedir()): string[] {
  return [
    '/opt/homebrew/bin', '/opt/homebrew/sbin', '/usr/local/bin',
    `${home}/.local/bin`, `${home}/.bun/bin`, `${home}/.cargo/bin`,
    `${home}/.deno/bin`, `${home}/.npm-global/bin`, `${home}/.volta/bin`,
    `${home}/.codex/bin`, `${home}/.claude/bin`,
    '/usr/bin', '/bin', '/usr/sbin', '/sbin',
  ]
}

/** Merge PATH-style `:`-separated dir lists, de-duplicating while preserving first-seen order. Pure. */
export function computeAugmentedPath(...lists: Array<string | null | undefined>): string {
  const seen = new Set<string>()
  const out: string[] = []
  for (const list of lists) {
    for (const dir of (list || '').split(':')) {
      const d = dir.trim()
      if (d && !seen.has(d)) { seen.add(d); out.push(d) }
    }
  }
  return out.join(':')
}

/** Capture the PATH a login+interactive shell would have (picks up brew shellenv in ~/.zprofile etc). */
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

/** The shell that runs agent commands: zsh where present (macOS behavior), then bash/sh for Linux. */
export function commandShell(): string {
  if (cachedCommandShell) return cachedCommandShell
  for (const candidate of ['/bin/zsh', '/bin/bash', '/bin/sh', process.env.SHELL]) {
    if (candidate && existsSync(candidate)) return (cachedCommandShell = candidate)
  }
  return (cachedCommandShell = '/bin/sh')
}

/** zsh keeps the login shell for macOS PATH parity; bash/sh run plain, as a login bash re-clobbers PATH. */
export function commandShellArgs(): string[] {
  return commandShell().endsWith('zsh') ? ['-lc'] : ['-c']
}

/**
 * Augment process.env.PATH (in place) so GUI-launched processes can find CLIs the stripped PATH
 * omits. Existing entries win; the login-shell PATH and known dirs are appended. Safe to call once
 * at startup. No-op on Windows.
 */
export function ensureCliPath(): void {
  if (process.platform === 'win32') return
  process.env.PATH = computeAugmentedPath(process.env.PATH, loginShellPath(), knownBinDirs().join(':'))
}

/**
 * Resolve a CLI to an absolute path: an explicit configured path wins; otherwise probe known install
 * dirs; otherwise fall back to the bare name (relying on a — hopefully augmented — PATH). `dirs`/`exists`
 * are injectable for testing.
 */
export function resolveCliBinary(
  explicit: string | undefined | null,
  names: string[],
  opts: { dirs?: string[]; exists?: (p: string) => boolean } = {},
): string {
  const cli = explicit?.trim()
  if (cli) return cli
  const dirs = opts.dirs ?? knownBinDirs()
  const exists = opts.exists ?? existsSync
  for (const name of names) {
    for (const dir of dirs) {
      const full = `${dir}/${name}`
      if (exists(full)) return full
    }
  }
  return names[0] ?? ''
}

/**
 * Validate that a value pushed into a child-process argv element is a safe identifier, not a
 * disguised flag. Resume cursors and model ids originate from CLI output / provider config and
 * are passed back as separate argv tokens (array-form spawn — no shell injection). But a crafted
 * value like `--mode=yolo` would be honored by the host CLI as a flag. Reject anything that is
 * empty, exceeds the max length, starts with `-`, or contains characters outside the safe set.
 */
const SAFE_ARGV_ID_RE = /^[A-Za-z0-9_.@:/-]{1,256}$/
export function isSafeArgvId(value: string | null | undefined): boolean {
  return !!value && !value.startsWith('-') && SAFE_ARGV_ID_RE.test(value)
}
