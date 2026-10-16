import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import type { CliDefinition, CliToolStatus } from './types/cli'
import type { Provider } from './types/base'
import { knownBinDirs } from './cli-path'
import { versionAdvisory } from './cli-version-advisory'

// Built-in CLIs (adapted from Clyde's Agent.builtins). Colors are Clyde's hex tints; icons map
// to TensorFold's icon set (src/lib/icons.tsx). resumeCommand is used by Phase 2 (tmux reattach).
export function builtinClis(): CliDefinition[] {
  return [
    { id: 'shell', name: 'Shell', short: 'zsh', command: null, resumeCommand: null,
      color: '#8B8B97', icon: 'terminal', builtin: true },
    { id: 'claude', name: 'Claude Code', short: 'claude', command: 'claude', resumeCommand: 'claude --resume',
      color: '#D97757', icon: 'spark', builtin: true },
    { id: 'codex', name: 'Codex', short: 'codex', command: 'codex', resumeCommand: 'codex resume --last',
      color: '#2FB9B0', icon: 'code', builtin: true },
    { id: 'cursor', name: 'Cursor Agent', short: 'cursor', command: 'cursor-agent', resumeCommand: null,
      color: '#8B8B97', icon: 'spark', builtin: true },
    { id: 'gemini', name: 'Gemini', short: 'gemini', command: 'gemini', resumeCommand: null,
      color: '#6E7BF2', icon: 'globe', builtin: true },
    { id: 'grok', name: 'Grok', short: 'grok', command: 'grok', resumeCommand: null,
      color: '#4DA3FF', icon: 'spark', builtin: true },
    { id: 'hermes', name: 'Hermes', short: 'hermes', command: 'hermes', resumeCommand: 'hermes --resume',
      color: '#C9A54E', icon: 'spark', builtin: true },
    { id: 'opencode', name: 'OpenCode', short: 'opencode', command: 'opencode', resumeCommand: 'opencode --continue',
      color: '#E06C9F', icon: 'code', builtin: true },
    { id: 'kilo', name: 'Kilo', short: 'kilo', command: 'kilo', resumeCommand: null,
      color: '#B48AF7', icon: 'code', builtin: true },
    { id: 'pi', name: 'Pi', short: 'pi', command: 'pi', resumeCommand: 'pi --resume',
      color: '#5BC873', icon: 'cpu', builtin: true },
  ]
}

function providerToolId(provider: Provider): string | null {
  if (provider.transport === 'codex-cli') return 'codex'
  if (provider.transport === 'claude-cli') return 'claude'
  if (provider.transport === 'cursor-cli') return 'cursor'
  if (provider.transport === 'gemini-cli') return 'gemini'
  if (provider.transport === 'pi-cli') return 'pi'
  if (provider.transport === 'opencode') return 'opencode'
  if (provider.transport === 'kilo') return 'kilo'
  return null
}

function providerOverrideFor(cli: CliDefinition, providers: Provider[]): { path: string | null; providerIds: string[] } {
  const matches = providers.filter((provider) => providerToolId(provider) === cli.id)
  const configured = matches.map((provider) => cleanConfiguredPath(provider.cliPath)).find((value): value is string => !!value)
  return { path: configured ?? null, providerIds: matches.map((provider) => provider.id) }
}

function cleanConfiguredPath(value?: string | null): string | null {
  const trimmed = value?.trim()
  if (!trimmed) return null
  if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) return trimmed.slice(1, -1)
  return trimmed
}

function shellQuotePath(path: string): string {
  return /^[A-Za-z0-9_/:=+,.@%-]+$/.test(path) ? path : `'${path.replace(/'/g, "'\\''")}'`
}

function binaryPathPresent(
  binary: string,
  opts: { dirs?: string[]; exists?: (p: string) => boolean } = {},
): boolean {
  const dirs = opts.dirs ?? knownBinDirs()
  const exists = opts.exists ?? existsSync
  if (binary.includes('/')) return exists(binary)
  return dirs.some((dir) => exists(`${dir}/${binary}`))
}

export function effectiveCliCommand(cli: CliDefinition, configuredPath?: string | null): string | null {
  if (!cli.command) return null
  const configured = cleanConfiguredPath(configuredPath)
  if (!configured) return cli.command
  const [, ...args] = cli.command.trim().split(/\s+/)
  return [shellQuotePath(configured), ...args].join(' ')
}

// Is a CLI's binary on disk? A plain shell (command === null) is always available; otherwise probe
// the known install dirs for the first word of the launch command. `dirs`/`exists` are injectable
// for testing. Pure (no process spawn — relies on the PATH dirs ensureCliPath() already augments).
export function cliBinaryPresent(
  cli: CliDefinition,
  opts: { dirs?: string[]; exists?: (p: string) => boolean } = {},
): boolean {
  if (!cli.command) return true // plain shell — nothing to resolve
  const dirs = opts.dirs ?? knownBinDirs()
  const bin = cli.command.trim().split(/\s+/)[0]
  if (!bin) return true
  return binaryPathPresent(bin, { ...opts, dirs })
}

// Filter a CLI list to the ones whose binary is installed. The plain shell is always kept (it's the
// fallback launcher), even if it somehow sorts out. Order is preserved. Pure + injectable.
export function installedClis(
  clis: CliDefinition[],
  opts: { dirs?: string[]; exists?: (p: string) => boolean } = {},
): CliDefinition[] {
  return clis.filter((c) => !c.command || cliBinaryPresent(c, opts))
}

export function cliToolStatuses(
  clis: CliDefinition[],
  providers: Provider[] = [],
  opts: {
    dirs?: string[]
    exists?: (p: string) => boolean
    overrides?: Record<string, { binaryPath?: string }>
    // Probed installed versions, and out-of-band "latest" + update commands per CLI id. When a
    // latest version is present the host computes a version advisory (VST-037).
    versions?: Record<string, string | null>
    // Why a probe failed, per CLI id; a CLI with a problem is 'broken' and reports no version.
    problems?: Record<string, string | null>
    latestVersions?: Record<string, string | null>
    updateCommands?: Record<string, string | null>
  } = {},
): CliToolStatus[] {
  return clis.map((cli) => {
    const providerOverride = providerOverrideFor(cli, providers)
    const configuredPath = providerOverride.path ?? cleanConfiguredPath(opts.overrides?.[cli.id]?.binaryPath) ?? null
    const effectiveCommand = effectiveCliCommand(cli, configuredPath)
    const installed = !effectiveCommand || (configuredPath ? binaryPathPresent(configuredPath, opts) : cliBinaryPresent(cli, opts))
    const problem = installed && effectiveCommand ? opts.problems?.[cli.id] ?? null : null
    const version = problem ? null : opts.versions?.[cli.id] ?? null
    const latestVersion = opts.latestVersions?.[cli.id] ?? null
    const updateCommand = opts.updateCommands?.[cli.id] ?? null
    // Compute an advisory only when there's a latest version to compare against; otherwise
    // leave it absent so the doctor shows nothing rather than a false "up to date".
    const advisory = effectiveCommand && latestVersion
      ? versionAdvisory({ name: cli.name, installedVersion: version, latestVersion, updateCommand, installed })
      : undefined
    return {
      ...cli,
      version,
      state: !installed ? 'missing' as const : problem ? 'broken' as const : 'installed' as const,
      problem,
      effectiveCommand,
      configuredPath,
      providerIds: providerOverride.providerIds,
      installed,
      latestVersion,
      updateCommand,
      advisory,
    }
  })
}

export function makeCustomCli(name: string, command: string): CliDefinition {
  const trimmed = command.trim()
  return {
    id: randomUUID(),
    name: name.trim() || trimmed,
    short: trimmed.split(/\s+/)[0] || name.trim().toLowerCase(),
    command: trimmed,
    resumeCommand: null,
    color: '#8AB4F8',
    icon: 'terminal',
    builtin: false,
  }
}

// The command to type into an interactive shell. Permission modes belong to the guarded engine
// adapters, not this terminal launcher: silently appending bypass flags here would turn a future
// UI toggle into an approval-boundary escape. Returns null for a plain shell.
export function launchString(cli: CliDefinition): string | null {
  if (!cli.command) return null
  return cli.command
}

// The command to type into a RESTORED shell so the CLI reattaches to its prior session
// (claude → `claude --continue`, codex → `codex resume --last`). When a CLI has no
// resumeCommand we fall back to launchString — a plain shell stays null, and any other CLI
// simply relaunches fresh (a custom CLI re-runs its own command). Pure; mirrors launchString.
export function resumeString(cli: CliDefinition): string | null {
  if (cli.resumeCommand) return cli.resumeCommand
  return launchString(cli)
}
