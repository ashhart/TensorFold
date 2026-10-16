// Connector action safety (VST-013). Two pure pieces:
//  1) Capability modes + configured-verdicts — a connector only counts as "configured"
//     when it has PROVEN the capabilities its mode claims (read → a successful sync or
//     indexed sample; write → a recorded successful outbound action). Configuration
//     without proof is 'unproven', named per missing capability — never silently healthy.
//  2) Safe outbound previews — before an outbound connector action is approved, render
//     exactly what would leave the machine (target + bounded, redacted content excerpt)
//     so approval is an informed decision about the actual payload.

import { redactText } from './export-redaction'
import type { ConnectorId } from './types'

export type ConnectorCapability = 'read' | 'write'

// What each connector family's mode CLAIMS. A read+write connector must prove both
// before it renders configured; read-only families never claim write.
const CONNECTOR_MODES: Record<ConnectorId, ConnectorCapability[]> = {
  email: ['read', 'write'],
  slack: ['read', 'write'],
  discord: ['read', 'write'],
  mattermost: ['read', 'write'],
  matrix: ['read', 'write'],
  googlechat: ['read', 'write'],
  whatsapp: ['read', 'write'],
  imessage: ['read'],
  telegram: ['read', 'write'],
  docs: ['read'],
  calendar: ['read', 'write'],
  browser: ['read'],
  controller: ['write'],
  webhook: ['read', 'write'],
}

export function connectorModeCapabilities(connectorId: ConnectorId): ConnectorCapability[] {
  return CONNECTOR_MODES[connectorId] ?? ['read']
}

export interface ConnectorCapabilityEvidence {
  // Read proof: a completed sync or at least one indexed sample item.
  syncedAt?: number | null
  sampleItemCount?: number
  // Write proof: a recorded successful outbound action (send/draft delivered without error).
  lastOutboundOkAt?: number | null
}

export interface ConnectorConfiguredVerdict {
  configured: boolean
  missing: ConnectorCapability[]
  detail: string
}

/**
 * "Configured" is an earned state (VST-013): every capability the connector's mode claims
 * needs live evidence. The verdict names exactly what is unproven so the fix is obvious.
 */
export function connectorConfiguredVerdict(connectorId: ConnectorId, evidence: ConnectorCapabilityEvidence): ConnectorConfiguredVerdict {
  const modes = connectorModeCapabilities(connectorId)
  const missing: ConnectorCapability[] = []
  if (modes.includes('read') && !(evidence.syncedAt || (evidence.sampleItemCount ?? 0) > 0)) missing.push('read')
  if (modes.includes('write') && !evidence.lastOutboundOkAt) missing.push('write')
  if (!missing.length) return { configured: true, missing, detail: `Proven: ${modes.join(' + ')}.` }
  const parts = missing.map((capability) => capability === 'read'
    ? 'read unproven — run a sync or setup test'
    : 'write unproven — no successful outbound action recorded yet')
  return { configured: false, missing, detail: parts.join('; ') + '.' }
}

export interface OutboundActionPreview {
  connectorId: ConnectorId
  action: string
  // Who/where this goes (recipient, channel, calendar, URL) — never guessed.
  target: string
  // Bounded, redacted excerpt of what would actually be sent.
  excerpt: string
  text: string
}

const PREVIEW_EXCERPT_CHARS = 280

function excerptOf(value: unknown): string {
  const raw = typeof value === 'string' ? value : value == null ? '' : JSON.stringify(value)
  const redacted = redactText(raw).text.replace(/\s+/g, ' ').trim()
  return redacted.length > PREVIEW_EXCERPT_CHARS ? `${redacted.slice(0, PREVIEW_EXCERPT_CHARS)}…` : redacted
}

/**
 * Render what an outbound connector tool call WOULD do (VST-013), for the approval item.
 * Returns null for non-outbound tools — inbound/read tools need no outbound preview.
 */
export function outboundActionPreview(toolName: string, args: Record<string, unknown>): OutboundActionPreview | null {
  const arg = (key: string) => args?.[key]
  switch (toolName) {
    case 'email_compose': {
      const target = String(arg('to') ?? arg('recipient') ?? 'unspecified recipient')
      const subject = excerptOf(arg('subject'))
      const body = excerptOf(arg('body') ?? arg('text'))
      return preview('email', 'compose email', target, body, `To ${target}${subject ? ` — "${subject}"` : ''}`)
    }
    case 'message_compose': {
      const channel = String(arg('channel') ?? arg('target') ?? arg('to') ?? 'unspecified channel')
      const body = excerptOf(arg('text') ?? arg('body') ?? arg('message'))
      return preview('slack', 'compose message', channel, body, `To ${channel}`)
    }
    case 'calendar_create':
    case 'calendar_update': {
      const title = excerptOf(arg('title') ?? arg('summary'))
      const when = [arg('start'), arg('end')].filter(Boolean).map(String).join(' → ')
      // Attendees are the TARGET of this action — the approver must see who gets invited
      // (same rule as the email recipient above). Bounded, not redacted.
      const attendees = Array.isArray(arg('attendees')) ? (arg('attendees') as unknown[]).slice(0, 5).map(String).join(', ').slice(0, 200) : ''
      return preview('calendar', toolName === 'calendar_create' ? 'create event' : 'update event', attendees || 'calendar', [title, when].filter(Boolean).join(' · '), `${title || 'event'}${attendees ? ` with ${attendees}` : ''}`)
    }
    case 'browser_automation': {
      const goal = excerptOf(arg('goal') ?? arg('task'))
      const url = String(arg('url') ?? '')
      return preview('browser', 'browser automation', url || 'current page', goal, url ? `On ${url}` : 'On the current page')
    }
    default:
      return null
  }
}

function preview(connectorId: ConnectorId, action: string, target: string, excerpt: string, headline: string): OutboundActionPreview {
  // Targets stay VISIBLE (bounded, not redacted): who/where something goes is exactly
  // what the approver is deciding about. Only content excerpts get the redaction pass.
  const boundedTarget = target.replace(/\s+/g, ' ').trim().slice(0, 200) || 'unspecified'
  return {
    connectorId,
    action,
    target: boundedTarget,
    excerpt,
    text: [
      `Outbound preview — ${action}. ${headline}.`,
      excerpt ? `Content: ${excerpt}` : 'No content payload.',
    ].join('\n'),
  }
}
