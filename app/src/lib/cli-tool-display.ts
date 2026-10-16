// How a provider CLI row reads in Settings: one state, a quiet version and the probe's error when broken.
import type { CliToolState, CliToolStatus } from './types'

export function cliToolState(tool: Pick<CliToolStatus, 'command' | 'installed' | 'state'>): CliToolState {
  if (!tool.command) return 'installed'
  return tool.state ?? (tool.installed ? 'installed' : 'missing')
}

/** "2.1.280 (Claude Code)" and "codex-cli 0.142.5" read as "v2.1.280" and "v0.142.5"; null when no version is in it. */
export function cliVersionLabel(version: string | null | undefined): string | null {
  const match = version?.match(/\bv?(\d+\.\d+(?:\.\d+)*(?:-[0-9A-Za-z.]+)?)/)
  return match ? `v${match[1]}` : null
}

/** The status column text and its hover title. */
export function cliToolStatusText(tool: Pick<CliToolStatus, 'command' | 'installed' | 'state' | 'version' | 'problem'>): { text: string; title: string } {
  if (!tool.command) return { text: 'Built in', title: 'Built in' }
  const state = cliToolState(tool)
  if (state === 'missing') return { text: 'Missing', title: 'Not found on PATH or the configured path' }
  if (state === 'broken') return { text: 'Broken', title: tool.problem || '--version printed no version' }
  const version = cliVersionLabel(tool.version)
  return version ? { text: version, title: tool.version?.trim() || version } : { text: 'Installed', title: 'Installed' }
}
