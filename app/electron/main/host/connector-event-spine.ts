import { randomUUID } from 'node:crypto'
import { connectorProjectMemoryCandidateInputs } from '../../agent/connector-feed'
import { createProjectMemoryCandidate } from '../../agent/project-memory'
import { approvalPolicyScope, requiresFreshConnectorApproval, type ApprovalPolicyScope } from '../../agent/approval-policy'
import { auditPreview, redactSettings } from '../../agent/security'
import { persistConnectorEvents, persistProjectMemoryCandidates } from './persistence'
import { auditEvent } from './audit'
import { store } from '../state'
import type { ConnectorEvent } from '../../agent/types'
import type { ConnectorRuntimeCallbacks } from './connector-runtime'

export function publicConnectorSettings() {
  return redactSettings(store.settings)
}

export function connectorPolicyForEvent(event: ConnectorEvent): ApprovalPolicyScope {
  return approvalPolicyScope('connector', `${event.connectorId}:${event.action}`, {
    connectorId: event.connectorId,
    action: event.action,
    actor: event.actor,
    target: event.threadKey,
    threadKey: event.threadKey,
    title: event.title,
    payloadKeys: event.payloadKeys,
  })
}

export async function resolveSavedConnectorApprovalPolicy(
  event: ConnectorEvent,
  callbacks: Pick<ConnectorRuntimeCallbacks, 'findApprovalRule'> & {
    audit?: (data: Record<string, unknown>) => void | Promise<void>
    now?: () => number
  },
): Promise<ConnectorEvent> {
  if (!event.requiresApproval && (event.approvalStatus ?? null) !== 'pending') return event
  const policy = connectorPolicyForEvent(event)
  const rule = await callbacks.findApprovalRule(policy.scope, policy.key, { workspace: `connector:${event.connectorId}` })
  if (!rule) return event
  const approved = rule.decision === 'allow'
  // A durable rule can remember a denial, but it must not silently authorize a
  // decision-sensitive external mutation whose content may have changed since the
  // rule was created (messages, calendar changes, purchases, permissions, etc.).
  if (approved && requiresFreshConnectorApproval(policy)) return event
  const status = approved ? 'approved' : 'denied'
  const summary = `${event.summary}\n\nAuto-${status} by saved policy: ${policy.explanation}`
  await callbacks.audit?.({
    eventId: event.id,
    connectorId: event.connectorId,
    action: event.action,
    policyKey: policy.key,
    policyLabel: policy.label,
    decision: rule.decision,
  })
  return {
    ...event,
    requiresApproval: false,
    approvalStatus: status,
    resolvedAt: callbacks.now?.() ?? Date.now(),
    resolvedBy: 'saved-policy',
    summary,
  }
}

export interface ConnectorEventSpine {
  rememberConnectorEvents: (events: ConnectorEvent[]) => Promise<void>
  recordCalendarEvent: (event: ConnectorEvent) => Promise<ConnectorEvent>
  recordMessageEvent: (event: ConnectorEvent) => Promise<ConnectorEvent>
  recordEventDelivery: (event: ConnectorEvent, delivery: ConnectorEvent['delivery']) => Promise<void>
  rememberConnectorMemoryCandidates: (events: ConnectorEvent[]) => void
  applySavedConnectorApprovalPolicy: (event: ConnectorEvent) => Promise<ConnectorEvent>
}

export function createConnectorEventSpine(
  deps: { callbacks: ConnectorRuntimeCallbacks; syncQueue?: unknown },
): ConnectorEventSpine {
  const { callbacks } = deps

  async function applySavedConnectorApprovalPolicy(event: ConnectorEvent): Promise<ConnectorEvent> {
    return resolveSavedConnectorApprovalPolicy(event, {
      findApprovalRule: callbacks.findApprovalRule,
      audit: (data) => auditEvent('connector.approval_policy_matched', data),
    })
  }

  function rememberConnectorMemoryCandidates(events: ConnectorEvent[]) {
    const defaultProjectId = store.settings.defaultCwd
      ? store.projects.find((project) => !project.archivedAt && project.cwd === store.settings.defaultCwd)?.id ?? null
      : null
    const inputs = connectorProjectMemoryCandidateInputs(events, {
      projects: store.projects,
      defaultProjectId,
      existingCandidates: store.projectMemoryCandidates,
      now: Date.now(),
    })
    if (!inputs.length) return
    const now = Date.now()
    const seen = new Set(store.projectMemoryCandidates.map((candidate) => `${candidate.projectId}:${candidate.sourceRef ?? ''}`))
    const candidates = inputs
      .filter((input) => {
        const key = `${input.projectId}:${input.sourceRef ?? ''}`
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })
      .map((input) => createProjectMemoryCandidate(input, now, `mc_${randomUUID().slice(0, 8)}`))
    if (!candidates.length) return
    store.projectMemoryCandidates = [...candidates, ...store.projectMemoryCandidates].slice(0, 300)
    persistProjectMemoryCandidates()
  }

  function commitConnectorEvents(events: ConnectorEvent[]) {
    if (!events.length) return
    // Dedup by stable message id AT INSERT TIME against the CURRENT store (not a snapshot a
    // sync took before its await): closes the window where two concurrent syncs double-ingest
    // the same messages, and drops WhatsApp replays the push bridge can re-deliver.
    const seenIds = new Set<string>()
    for (const e of store.connectorEvents) {
      const mid = typeof e.payload?.messageId === 'string' ? e.payload.messageId : null
      if (mid) seenIds.add(mid)
    }
    const fresh: ConnectorEvent[] = []
    for (const event of events) {
      const mid = typeof event.payload?.messageId === 'string' ? event.payload.messageId : null
      if (mid) {
        if (seenIds.has(mid)) continue
        seenIds.add(mid)
      }
      fresh.push(event)
    }
    if (!fresh.length) return
    store.connectorEvents = [...fresh, ...store.connectorEvents].slice(0, 500)
    persistConnectorEvents()
    rememberConnectorMemoryCandidates(fresh)
    for (const event of fresh) {
      callbacks.remotePublish(event)
      callbacks.emitOutboundWebhook(event)
    }
  }

  let connectorApprovalTail: Promise<void> = Promise.resolve()
  async function rememberConnectorEvents(events: ConnectorEvent[]): Promise<void> {
    if (!events.length) return
    const needsPolicy = events.some((event) => event.requiresApproval || (event.approvalStatus ?? null) === 'pending')
    if (!needsPolicy) {
      commitConnectorEvents(events)
      return
    }
    const batch = events.slice()
    const run = connectorApprovalTail.then(async () => {
      const resolved: ConnectorEvent[] = []
      for (const event of batch) resolved.push(await applySavedConnectorApprovalPolicy(event))
      commitConnectorEvents(resolved)
    })
    connectorApprovalTail = run.catch((error) => {
      console.warn('[connector] saved approval policy lookup failed; events remain pending:', auditPreview(error))
      commitConnectorEvents(batch)
    })
    await connectorApprovalTail
  }

  // Record a delivery result with a read-modify-write against the CURRENT store, by event id.
  // Mutating the live event object across the send's await is unsafe: a concurrent
  // rememberConnectorEvents can slice it past the 500-event cap, so the in-place mutation
  // would be lost from disk. Re-locate by id (or re-insert if it was evicted) so the
  // delivery/approval result is never dropped.
  //
  // This MUST serialize through the same connectorApprovalTail chain as
  // commitConnectorEvents: otherwise a delivery write interleaving with a pending
  // rememberConnectorEvents (whose commitConnectorEvents reassigns the store from a
  // fresh list computed before the delivery ran) would silently drop the delivery
  // result from persistence (last-write-wins between the two queued writes).
  function recordEventDelivery(event: ConnectorEvent, delivery: ConnectorEvent['delivery']): Promise<void> {
    const run = connectorApprovalTail.then(() => {
      const idx = store.connectorEvents.findIndex((e) => e.id === event.id)
      if (idx >= 0) {
        store.connectorEvents = store.connectorEvents.map((e) => (e.id === event.id ? { ...e, delivery } : e))
      } else {
        store.connectorEvents = [{ ...event, delivery }, ...store.connectorEvents].slice(0, 500)
      }
      persistConnectorEvents()
    })
    connectorApprovalTail = run.catch((error) => {
      console.warn('[connector] recordEventDelivery failed:', auditPreview(error))
    })
    return connectorApprovalTail
  }

  async function recordCalendarEvent(event: ConnectorEvent): Promise<ConnectorEvent> {
    await rememberConnectorEvents([event])
    await auditEvent('calendar.event', {
      eventId: event.id,
      action: event.action,
      threadKey: event.threadKey,
      approvalId: event.approvalId,
      requiresApproval: event.requiresApproval,
      sourceIds: (event.sources ?? []).map((source) => source.id),
      title: auditPreview(event.title),
    })
    return store.connectorEvents.find((candidate) => candidate.id === event.id) ?? event
  }

  async function recordMessageEvent(event: ConnectorEvent): Promise<ConnectorEvent> {
    await rememberConnectorEvents([event])
    await auditEvent('message.compose', {
      eventId: event.id,
      connectorId: event.connectorId,
      action: event.action,
      threadKey: event.threadKey,
      approvalId: event.approvalId,
      requiresApproval: event.requiresApproval,
      target: auditPreview(event.actor),
    })
    return store.connectorEvents.find((candidate) => candidate.id === event.id) ?? event
  }

  return {
    rememberConnectorEvents,
    recordCalendarEvent,
    recordMessageEvent,
    recordEventDelivery,
    rememberConnectorMemoryCandidates,
    applySavedConnectorApprovalPolicy,
  }
}
