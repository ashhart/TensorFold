import type { Settings } from '../../lib/types'

export const ASSISTANT_SOURCES = ['email', 'slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'whatsapp', 'imessage'] as const
export type AssistantSourceId = typeof ASSISTANT_SOURCES[number]

export const SOURCE_LABEL: Record<AssistantSourceId, string> = {
  email: 'Email', slack: 'Slack', discord: 'Discord', mattermost: 'Mattermost', matrix: 'Matrix', googlechat: 'Google Chat', whatsapp: 'WhatsApp', imessage: 'iMessage',
}

const NOUN: Partial<Record<AssistantSourceId, string>> = { email: 'source', slack: 'workspace', discord: 'bot', mattermost: 'server', matrix: 'account', googlechat: 'space' }

// One-line connector status for the Assistant's source rows. It reflects whether the source has
// synced or has data, not just its toggle, so a working source never reads "Not configured".
export function sourceDetail(settings: Settings | null | undefined, id: AssistantSourceId): string {
  if (!settings) return ''
  if (id === 'whatsapp') {
    const paired = settings.whatsapp.sources.filter((s) => s.paired)
    if (paired.length) return `${paired.length} paired`
    return settings.whatsapp.sources.length ? 'Pairing…' : 'Not configured'
  }
  if (id === 'imessage') {
    if (settings.imessage.lastError) return `Error: ${settings.imessage.lastError.slice(0, 40)}`
    if (settings.imessage.lastSyncAt) return 'Connected · synced'
    if (settings.imessage.enabled) return 'Enabled, not synced yet'
    return 'Not configured'
  }
  const block = settings[id] as { sources: { enabled?: boolean }[]; lastSyncAt?: number | null }
  const on = block.sources.filter((s) => s.enabled)
  if (!on.length) return 'Not configured'
  const noun = NOUN[id] ?? 'source'
  const count = `${on.length} ${noun}${on.length === 1 ? '' : 's'}`
  return block.lastSyncAt ? `${count} · synced` : count
}

export function connectedSources(settings: Settings | null | undefined): AssistantSourceId[] {
  return ASSISTANT_SOURCES.filter((id) => { const d = sourceDetail(settings, id); return !!d && d !== 'Not configured' })
}
