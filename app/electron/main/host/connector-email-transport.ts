import { emailDraftEvent, writeEmailOutboxMessage } from '../../agent/email-connector'
import { sendEmailOverSmtp } from '../../agent/smtp-send'
import { auditPreview } from '../../agent/security'
import { vontraPath } from '../../store'
import { auditEvent } from './audit'
import { store } from '../state'
import { recordConnectorInteraction } from '../../agent/connector-spine'
import type { ConnectorEvent, EmailDraftInput } from '../../agent/types'
import type { ConnectorEventSpine } from './connector-event-spine'

export interface ConnectorEmailTransport {
  recordEmailDraft: (input: EmailDraftInput, sources?: ConnectorEvent['sources']) => Promise<ConnectorEvent>
  recordEmailEvent: (event: ConnectorEvent) => Promise<ConnectorEvent>
  isEmailSendRequest: (event: ConnectorEvent) => boolean
  transportApprovedEmailEvent: (event: ConnectorEvent) => Promise<void>
}

export function createEmailTransport(deps: {
  spine: Pick<ConnectorEventSpine, 'rememberConnectorEvents' | 'recordEventDelivery'>
}): ConnectorEmailTransport {
  const { rememberConnectorEvents, recordEventDelivery } = deps.spine

  async function recordEmailDraft(input: EmailDraftInput, sources: ConnectorEvent['sources'] = []): Promise<ConnectorEvent> {
    const event = emailDraftEvent(input, Date.now(), sources)
    await rememberConnectorEvents([event])
    await auditEvent('email.compose', {
      eventId: event.id,
      action: event.action,
      threadKey: event.threadKey,
      approvalId: event.approvalId,
      requiresApproval: event.requiresApproval,
      sourceIds: (event.sources ?? []).map((source) => source.id),
      to: auditPreview(input.to),
      subject: auditPreview(input.subject),
    })
    const recorded = store.connectorEvents.find((candidate) => candidate.id === event.id) ?? event
    await transportApprovedEmailEvent(recorded)
    return recorded
  }

  async function recordEmailEvent(event: ConnectorEvent): Promise<ConnectorEvent> {
    await rememberConnectorEvents([event])
    await auditEvent('email.event', {
      eventId: event.id,
      action: event.action,
      threadKey: event.threadKey,
      sourceIds: (event.sources ?? []).map((source) => source.id),
    })
    return store.connectorEvents.find((candidate) => candidate.id === event.id) ?? event
  }

  function isEmailSendRequest(event: ConnectorEvent): boolean {
    return event.connectorId === 'email' && ['send_requested', 'reply_requested'].includes(event.action)
  }

  async function transportApprovedEmailEvent(event: ConnectorEvent): Promise<void> {
    if (!isEmailSendRequest(event) || event.approvalStatus !== 'approved' || event.requiresApproval) return
    // If the send was composed against an IMAP account, deliver over SMTP so the
    // reply actually leaves the inbox; otherwise fall back to the local outbox.
    const sourceId = event.sources?.[0]?.sourceId ?? null
    const account = sourceId ? store.settings.email.sources.find((s) => s.id === sourceId && s.kind === 'imap') : null
    if (account) {
      try {
        const result = await sendEmailOverSmtp(event, account)
        recordEventDelivery(event, {
          transport: result.transport,
          status: 'delivered',
          path: `smtp:${account.host}`,
          deliveredAt: result.deliveredAt,
          error: null,
        })
        await auditEvent('email.transport.smtp', {
          eventId: event.id,
          approvalId: event.approvalId,
          to: auditPreview(result.envelope.to),
          messageId: auditPreview(result.messageId),
        })
        rememberConnectorEvents([recordConnectorInteraction({
          connectorId: 'email',
          pluginName: 'Email',
          action: 'email_smtp_sent',
          direction: 'outbound',
          target: result.envelope.to,
          output: {
            title: 'Email sent via SMTP',
            summary: `${event.title} -> ${result.envelope.to}`,
            threadKey: event.threadKey,
          },
        })])
      } catch (e: any) {
        const error = String(e?.message ?? e)
        recordEventDelivery(event, { transport: 'smtp', status: 'failed', path: null, deliveredAt: null, error })
        await auditEvent('email.transport.smtp_failed', {
          eventId: event.id,
          approvalId: event.approvalId,
          error: auditPreview(error),
        })
        throw e
      }
      return
    }
    // File-based email sources: write the approved draft to the local outbox.
    try {
      const result = await writeEmailOutboxMessage(event, vontraPath('email-outbox'))
      recordEventDelivery(event, {
        transport: result.transport,
        status: 'delivered',
        path: result.path,
        deliveredAt: result.deliveredAt,
        error: null,
      })
      await auditEvent('email.transport.local_outbox', {
        eventId: event.id,
        approvalId: event.approvalId,
        path: auditPreview(result.path),
        sourceIds: (event.sources ?? []).map((source) => source.id),
      })
      rememberConnectorEvents([recordConnectorInteraction({
        connectorId: 'email',
        pluginName: 'Email',
        action: 'email_outbox_delivered',
        direction: 'internal',
        target: result.path,
        output: {
          title: 'Email written to local outbox',
          summary: `${event.title} -> ${result.path}`,
          threadKey: event.threadKey,
        },
      })])
    } catch (e: any) {
      const error = String(e?.message ?? e)
      recordEventDelivery(event, {
        transport: 'local_outbox',
        status: 'failed',
        path: null,
        deliveredAt: null,
        error,
      })
      await auditEvent('email.transport.local_outbox_failed', {
        eventId: event.id,
        approvalId: event.approvalId,
        error: auditPreview(error),
      })
      throw e
    }
  }

  return { recordEmailDraft, recordEmailEvent, isEmailSendRequest, transportApprovedEmailEvent }
}
