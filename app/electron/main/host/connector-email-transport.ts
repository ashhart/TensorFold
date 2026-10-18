import { emailDraftEvent, writeEmailOutboxMessage } from '../../agent/email-connector'
import { sendEmailOverSmtp } from '../../agent/smtp-send'
import { auditPreview } from '../../agent/security'
import { vontraPath } from '../../store'
import { auditEvent } from './audit'
import { store } from '../state'
import type { ConnectorEvent, EmailDraftInput } from '../../agent/types'
import type { ConnectorEventSpine } from './connector-event-spine'

export interface ConnectorEmailTransport {
  recordEmailDraft: (input: EmailDraftInput, sources?: ConnectorEvent['sources']) => Promise<ConnectorEvent>
  recordEmailEvent: (event: ConnectorEvent) => Promise<ConnectorEvent>
  isEmailSendRequest: (event: ConnectorEvent) => boolean
  deliverApprovedEmailEvent: (event: ConnectorEvent) => Promise<NonNullable<ConnectorEvent['delivery']>>
}

export function createEmailTransport(deps: {
  spine: Pick<ConnectorEventSpine, 'rememberConnectorEvents'>
}): ConnectorEmailTransport {
  const { rememberConnectorEvents } = deps.spine
  async function recordEmailDraft(input: EmailDraftInput, sources: ConnectorEvent['sources'] = []): Promise<ConnectorEvent> {
    const event = emailDraftEvent(input, Date.now(), sources)
    await rememberConnectorEvents([event])
    await auditEvent('email.compose', { eventId: event.id, action: event.action, approvalId: event.approvalId, sourceIds: sources.map(source => source.id), to: auditPreview(input.to) })
    // Recording a draft never sends it, even when an old broad saved allow rule exists.
    return store.connectorEvents.find(candidate => candidate.id === event.id) ?? event
  }
  async function recordEmailEvent(event: ConnectorEvent): Promise<ConnectorEvent> {
    await rememberConnectorEvents([event])
    await auditEvent('email.event', { eventId: event.id, action: event.action, threadKey: event.threadKey })
    return store.connectorEvents.find(candidate => candidate.id === event.id) ?? event
  }
  function isEmailSendRequest(event: ConnectorEvent): boolean {
    return event.connectorId === 'email' && ['send_requested', 'reply_requested'].includes(event.action)
  }
  async function deliverApprovedEmailEvent(event: ConnectorEvent): Promise<NonNullable<ConnectorEvent['delivery']>> {
    if (!isEmailSendRequest(event) || event.approvalStatus !== 'approved' || event.requiresApproval || event.delivery?.status !== 'sending' || !event.outbound || event.delivery.reviewDigest !== event.reviewDigest) throw new Error('Email requires a durable exact-content approval before transport.')
    const snapshot = event.outbound
    const account = snapshot.accountId ? store.settings.email.sources.find(source => source.id === snapshot.accountId) : null
    if (snapshot.accountId && !account) return { transport: 'smtp', status: 'failed', deliveredAt: null, error: 'The reviewed email account is no longer available. Nothing was sent.' }
    // SMTP and local-outbox writers receive a detached approved payload, never the live
    // mutable event object. Preserve selected account identity instead of picking a default.
    const transportEvent: ConnectorEvent = { ...event, payload: { to: snapshot.recipient, subject: snapshot.subject, body: snapshot.body, inReplyTo: snapshot.inReplyTo, threadKey: snapshot.threadKey } }
    if (account?.kind === 'imap') {
      const selected = { ...account }
      if (!selected.host || !selected.user || !selected.password) return { transport: 'smtp', status: 'failed', deliveredAt: null, error: 'The reviewed email account is missing SMTP configuration. Nothing was sent.' }
      const result = await sendEmailOverSmtp(transportEvent, selected)
      await auditEvent('email.transport.smtp', { eventId: event.id, approvalId: event.approvalId, messageId: auditPreview(result.messageId) })
      return { transport: 'smtp', status: 'delivered', path: `smtp:${selected.id}`, sentId: result.messageId, deliveredAt: result.deliveredAt, error: null }
    }
    const result = await writeEmailOutboxMessage(transportEvent, vontraPath('email-outbox'))
    await auditEvent('email.transport.local_outbox', { eventId: event.id, approvalId: event.approvalId })
    // A file has been queued for an external mail client; no recipient delivery occurred.
    return { transport: 'local_outbox', status: 'queued', path: result.path, deliveredAt: null, error: null }
  }
  return { recordEmailDraft, recordEmailEvent, isEmailSendRequest, deliverApprovedEmailEvent }
}
