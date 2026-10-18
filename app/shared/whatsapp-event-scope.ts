import type { ConnectorEvent, WhatsAppBridgeSource } from '../electron/agent/types'

/** Exact account/contact identities; group IDs can never identify the linked phone. */
export function normalizeWhatsAppJid(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const input = value.trim()
  const digits = input.replace(/^\+/, '')
  if (/^\d{5,20}$/.test(digits)) return `${digits}@s.whatsapp.net`
  const match = /^(\d{5,25})(?::\d{1,5})?@(s\.whatsapp\.net|lid)$/.exec(input)
  return match ? `${match[1]}@${match[2]}` : null
}

/** Group conversation grants are separate from PN/LID account identity and self-chat. */
export function normalizeWhatsAppConversationJid(value: unknown): string | null {
  const contact = normalizeWhatsAppJid(value)
  if (contact) return contact
  if (typeof value !== 'string') return null
  const group = /^(\d{5,25}(?:-\d{1,25})?)@g\.us$/.exec(value.trim())
  return group ? `${group[1]}@g.us` : null
}

/** Reapply current account/contact grants whenever previously cached content is surfaced. */
export function isWhatsAppEventInCurrentScope(event: ConnectorEvent, sources: readonly WhatsAppBridgeSource[]): boolean {
  if (event.connectorId !== 'whatsapp' || typeof event.threadKey !== 'string') return false
  const jid = normalizeWhatsAppConversationJid(event.payload?.jid)
  if (!jid) return false
  const source = sources.find((candidate) => event.threadKey === `whatsapp:${candidate.id}:${jid}`)
  if (!source?.enabled) return false
  // A source can later link a different phone. Unbound older events cannot prove account
  // ownership and remain on disk, but are never used as current-context permissions.
  const own = normalizeWhatsAppJid(source.phoneNumber)
  const authenticated = normalizeWhatsAppJid(event.payload?.authenticatedAccountJid)
  if (!own || !authenticated || authenticated !== own) return false
  if (jid === own) return source.allowSelfChat === true
  // The bridge canonicalizes its socket-authenticated own LID to the PN before
  // storing a self-chat event. A cached flag never makes an unrelated LID our own.
  if (event.payload?.selfChat === true) return false
  return (source.allowedChatJids ?? []).some((allowed) => normalizeWhatsAppConversationJid(allowed) === jid)
}
