import { personalMessageDraftEvent } from '../../agent/tool-labels'
import { emailDraftEvent } from '../../agent/email-connector'
import type { Automation, AutomationRun, ConnectorEvent } from '../../agent/types'

export async function deliverAutomationResult(definition: Automation, run: AutomationRun, automationId: string, rememberConnectorEvents: (events: ConnectorEvent[]) => void | Promise<void>): Promise<void> {
  run.deliveryStatus = 'inbox'
  if (definition.deliverTo !== 'inbox' && !definition.previewOnly) {
    const target = definition.deliveryTarget
    if (!target || target.connectorId !== definition.deliverTo || !target.recipient.trim() || (target.connectorId !== 'telegram' && !target.accountId)) {
      // Existing addressless definitions remain safe/local until a user chooses a
      // destination. Successful generation is not a claim of message delivery.
      run.deliveryStatus = 'blocked'
      run.deliveryError = 'Choose an exact account and recipient before reviewing external delivery; this result remains in the inbox.'
    } else {
      const event = target.connectorId === 'email'
        ? emailDraftEvent({ to: target.recipient, subject: target.subject || `${definition.name} result`, body: run.output, mode: 'send' })
        : personalMessageDraftEvent({ connectorId: target.connectorId, target: target.recipient, body: run.output, accountId: target.accountId ?? undefined })
      event.approvalId = `automation-${automationId}-${run.id}`
      event.payload = { ...event.payload, accountId: target.accountId, automationId: automationId, runId: run.id }
      event.payloadKeys = [...event.payloadKeys, 'accountId', 'automationId', 'runId']
      await rememberConnectorEvents([event])
      run.deliveryEventId = event.id
      run.deliveryStatus = 'pending_approval'
    }
  }
}
