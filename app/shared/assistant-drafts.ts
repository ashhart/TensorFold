import type { ComposerDraftScope } from './composer-drafts'

export const ASSISTANT_DRAFT_SOURCES = ['all', 'email', 'slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'whatsapp', 'imessage', 'docs', 'calendar', 'telegram', 'controller', 'webhook', 'browser'] as const

export interface AssistantDraftEvent {
  id: string
  connectorId: string
  threadKey?: string
  sources?: Array<{ connectorId: string; sourceId?: string | null; sourceRevision?: string }>
}

function record(value: unknown): Record<string, any> { return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, any> : {} }

export function assistantSourceConfiguration(settings: unknown, connectorId: string, sourceId: string): Record<string, any> | null {
  const root = record(settings)
  const section = record(root[connectorId === 'docs' ? 'documents' : connectorId])
  return Array.isArray(section.sources) ? section.sources.find((source: any) => source && source.id === sourceId) ?? null : null
}

function safeUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value) return null
  try { const url = new URL(value); return `${url.protocol}//${url.host}${url.pathname}` } catch { return null }
}

/** Only account/location metadata, never tokens, passwords, message contents or filenames. */
function accountIdentity(source: Record<string, any> | null): unknown {
  if (!source) return null
  const selected: Record<string, unknown> = {}
  for (const key of ['id', 'kind', 'path', 'host', 'user', 'mailbox', 'port', 'tls', 'fromAddress', 'phoneNumber']) {
    const value = source[key]
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' || value === null) selected[key] = value
  }
  if (source.url) selected.url = safeUrl(source.url)
  // Host-authored rotation metadata distinguishes a replaced API-token account without
  // reading or hashing its credential. WhatsApp Signal rotations are intentionally excluded.
  if (source.secretRotations) selected.secretRotations = Object.fromEntries(Object.entries(record(source.secretRotations)).filter(([, value]) => value === null || typeof value === 'number').sort(([a], [b]) => a.localeCompare(b)))
  return selected
}

export function assistantDraftScope(selectedSource: string, event: AssistantDraftEvent | null, settings: unknown): ComposerDraftScope {
  if (!(ASSISTANT_DRAFT_SOURCES as readonly string[]).includes(selectedSource)) throw new Error('Unknown assistant draft source.')
  if (!event) {
    const root = record(settings)
    const sections = selectedSource === 'all' ? ASSISTANT_DRAFT_SOURCES.filter((source) => source !== 'all') : [selectedSource]
    const accounts = sections.map((source) => {
      const section = record(root[source === 'docs' ? 'documents' : source])
      const configured = Array.isArray(section.sources) ? section.sources.map((value: unknown) => accountIdentity(record(value))).sort((a: unknown, b: unknown) => JSON.stringify(a).localeCompare(JSON.stringify(b))) : source === 'telegram' ? [accountIdentity({ id: 'telegram-bot', secretRotations: root.botControl?.telegram?.secretRotations })] : []
      return [source, configured]
    })
    const identity = `assistant-overview:${JSON.stringify([selectedSource, accounts])}`
    if (identity.length > 16_000) throw new Error('Assistant account overview identity is too large to store safely.')
    return { surface: 'assistant', conversationId: identity, hostId: null, incognito: false }
  }
  if (!(ASSISTANT_DRAFT_SOURCES as readonly string[]).includes(event.connectorId) || event.connectorId === 'all') throw new Error('Unknown assistant conversation source.')
  const refs = (event.sources ?? []).filter((source) => source.connectorId === event.connectorId && source.sourceId)
  const accountIds = new Set(refs.map((source) => source.sourceId!))
  // Several live connectors carry their account only in the existing routing key.
  if (['slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'whatsapp'].includes(event.connectorId)) {
    const match = event.threadKey?.match(/^[^:]+:([^:]+):/)
    if (match) accountIds.add(match[1])
  }
  const accounts = [...accountIds].sort().map((id) => {
    const configured = assistantSourceConfiguration(settings, event.connectorId, id)
    if (!configured) throw new Error('Assistant conversation account is not configured.')
    return [id, refs.find((source) => source.sourceId === id)?.sourceRevision ?? null, accountIdentity(configured)]
  })
  const fallback = event.connectorId === 'email' && accounts.length === 0 ? event.id : null
  const identity = JSON.stringify([event.connectorId, event.threadKey?.trim() || `event:${event.id}`, accounts, fallback])
  if (identity.length > 16_000) throw new Error('Assistant conversation identity is too large to store safely.')
  return { surface: 'assistant', conversationId: `assistant:${identity}`, hostId: null, incognito: false }
}

export function isAssistantOverviewDraft(scope: ComposerDraftScope, settings: unknown): boolean {
  if (scope.surface !== 'assistant' || scope.hostId !== null || scope.incognito !== false) return false
  return (ASSISTANT_DRAFT_SOURCES as readonly string[]).some((source) => {
    try { return scope.conversationId === assistantDraftScope(source, null, settings).conversationId }
    catch { return false }
  })
}
