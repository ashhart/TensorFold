// How a provider CLI row reads in Settings: one state, a quiet version and the probe's error when broken.
import type { CliToolState, CliToolStatus } from './types'
import { authoredText } from './preferences/language-packs'
import { currentUiLanguage } from './preferences/current-locale'

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
  const tx = (source: string) => authoredText(currentUiLanguage(), source)
  if (!tool.command) {
    const builtIn = currentUiLanguage() === 'en' ? 'Built in' : tx('Built-in tools')
    return { text: builtIn, title: builtIn }
  }
  const state = cliToolState(tool)
  if (state === 'missing') return { text: tx('Missing'), title: tx('Not found on PATH or the configured path') }
  if (state === 'broken') return { text: tx('Broken'), title: tool.problem || tx('--version printed no version') }
  const version = cliVersionLabel(tool.version)
  return version ? { text: version, title: tool.version?.trim() || version } : { text: tx('Installed'), title: tx('Installed') }
}
