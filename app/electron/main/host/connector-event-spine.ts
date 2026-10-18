import { sealConnectorOutboundEvent, isOutboundSendRequest } from './connector-outbound-delivery'
import { saveConnectorEvents } from '../../store'
import { randomUUID } from 'node:crypto'
import { connectorProjectMemoryCandidateInputs } from '../../agent/connector-feed'
import { createProjectMemoryCandidate } from '../../agent/project-memory'
import { projectMemoryAutoSourceSuppressed } from './project-memory-auto-policy'
import { approvalPolicyScope, requiresFreshConnectorApproval, type ApprovalPolicyScope } from '../../agent/approval-policy'
import { auditPreview, redactSettings } from '../../agent/security'
import { persistProjectMemoryCandidates } from './persistence'
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
  event = sealConnectorOutboundEvent(event)
  if (!event.requiresApproval && (event.approvalStatus ?? null) !== 'pending') return event
  const policy = connectorPolicyForEvent(event)
  const rule = await callbacks.findApprovalRule(policy.scope, policy.key, { workspace: `connector:${event.connectorId}` })
  if (!rule) return event
  const approved = rule.decision === 'allow'
  // A durable rule can remember a denial, but it must not silently authorize a
  // decision-sensitive external mutation whose content may have changed since the
  // rule was created (messages, calendar changes, purchases, permissions, etc.).
  if (approved && (isOutboundSendRequest(event) || requiresFreshConnectorApproval(policy))) return event
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
        if (projectMemoryAutoSourceSuppressed(store.projectMemory, input)) return false
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

  async function commitConnectorEvents(events: ConnectorEvent[]): Promise<void> {
    if (!events.length) return
    // Dedup by stable message id AT INSERT TIME against the CURRENT store (not a snapshot a
    // sync took before its await): closes the window where two concurrent syncs double-ingest
    // the same messages, and drops WhatsApp replays the push bridge can re-deliver.
    const seenEventIds = new Set(store.connectorEvents.map(event => event.id))
    const seenIds = new Set<string>()
    for (const e of store.connectorEvents) {
      const mid = typeof e.payload?.messageId === 'string' ? e.payload.messageId : null
      if (mid) seenIds.add(mid)
    }
    const fresh: ConnectorEvent[] = []
    let replayedHostEvent = false
    for (const event of events) {
      // A repeated host event must not resurrect a consumed send approval.
      if (seenEventIds.has(event.id)) { replayedHostEvent = true; continue }
      seenEventIds.add(event.id)
      const mid = typeof event.payload?.messageId === 'string' ? event.payload.messageId : null
      if (mid) {
        if (seenIds.has(mid)) continue
        seenIds.add(mid)
      }
      fresh.push(event)
    }
    if (!fresh.length) {
      // Retry persistence of an unchanged known event without restoring an older
      // pending body or erasing a consumed delivery marker.
      if (replayedHostEvent) await saveConnectorEvents(structuredClone(store.connectorEvents))
      return
    }
    store.connectorEvents = [...fresh, ...store.connectorEvents].slice(0, 500)
    await saveConnectorEvents(structuredClone(store.connectorEvents))
    rememberConnectorMemoryCandidates(fresh)
    for (const event of fresh) {
      callbacks.remotePublish(event)
      callbacks.emitOutboundWebhook(event)
    }
  }

  let connectorApprovalTail: Promise<void> = Promise.resolve()
  function rememberConnectorEvents(events: ConnectorEvent[]): Promise<void> {
    if (!events.length) return Promise.resolve()
    events = events.map(sealConnectorOutboundEvent)
    const batch = events.slice()
    const run = connectorApprovalTail.then(async () => {
      const resolved: ConnectorEvent[] = []
      try {
        for (const event of batch) resolved.push(await applySavedConnectorApprovalPolicy(event))
      } catch (error) {
        console.warn('[connector] saved approval policy lookup failed; events remain pending:', auditPreview(error))
        resolved.splice(0, resolved.length, ...batch)
      }
      await commitConnectorEvents(resolved)
    })
    connectorApprovalTail = run.catch((error) => {
      console.warn('[connector] event persistence failed:', auditPreview(error))
    })
    return run
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
    rememberConnectorMemoryCandidates,
    applySavedConnectorApprovalPolicy,
  }
}
