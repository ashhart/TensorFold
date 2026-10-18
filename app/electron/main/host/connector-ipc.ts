import { composingDraftStream, resolveAdapter, sendViaChannel, streamDraftViaChannel, type ChannelApprovalRequest } from '../../agent/channel-adapter'
import { draftEditIntervalForPlatform, makeDraftEditThrottle } from '../../agent/draft-throttle'
import { parseThreadAddress } from '../../agent/thread-address'
import { buildAssistantControlSnapshot, recordConnectorInteraction } from '../../agent/connector-spine'
import { getWhatsAppBridgeRuntime } from '../../agent/whatsapp-bridge'
import { auditPreview } from '../../agent/security'
import { connectorDecisionQueue, createReviewedConnectorDelivery, isOutboundSendRequest, sealConnectorOutboundEvent } from './connector-outbound-delivery'
import { auditActor, auditEvent, requireSecurityRole } from './audit'
import { handle as registerHandler } from './ipc'
import { store } from '../state'
import { connectorPolicyForEvent } from './connector-event-spine'
import type {
  ConnectorEvent,
  ConnectorInteractionInput,
} from '../../agent/types'

// The shared type lives in ./connector-ipc/types.ts; re-exported here for consumers
// that import from the umbrella.
export type { ConnectorIpcDeps } from './connector-ipc/types'
import type { ConnectorIpcDeps } from './connector-ipc/types'

// Per-connector handler registrations live in ./connector-ipc/*.ts.
import { registerEmailIpc } from './connector-ipc/email'
import { registerIMessageIpc } from './connector-ipc/imessage'
import { registerWhatsAppIpc } from './connector-ipc/whatsapp'
import { registerDiscordIpc } from './connector-ipc/discord'
import { registerMattermostIpc } from './connector-ipc/mattermost'
import { registerMatrixIpc } from './connector-ipc/matrix'
import { registerGoogleChatIpc } from './connector-ipc/googlechat'
import { registerSlackIpc } from './connector-ipc/slack'
import { registerCalendarIpc } from './connector-ipc/calendar'
import { registerDocumentIpc } from './connector-ipc/documents'
import { registerBrowserIpc } from './connector-ipc/browser'

export function registerAllConnectorIpc(deps: ConnectorIpcDeps): void {
  const { spine, emailTransport, channels, callbacks } = deps
  const { rememberConnectorEvents } = spine
  const { requestChannelApproval, resolveChannelApprovalByThread, channelAdapters } = channels
  const reviewedDelivery = createReviewedConnectorDelivery({ adapters: channelAdapters, sendEmail: (event) => emailTransport.deliverApprovedEmailEvent(event) })
  const readonlyActions = new Set(['list', 'read', 'thread', 'summary', 'access'])
  const handle: typeof registerHandler = (channel, listener) => {
    registerHandler(channel, async (event, ...args) => {
      // Reads remain available while an update closes mutable operation admission.
      const readonly = channel === 'assistant:snapshot' || readonlyActions.has(channel.split(':')[1])
      const reservation = readonly ? null : callbacks.reserveOperation?.()
      try { return await listener(event, ...args) } finally { reservation?.release() }
    })
  }
  const guardedDeps = { ...deps, handle }

  // Unified outbound reply: route a reply to whichever channel the thread belongs to. On a
  // channel that can edit a sent message (Telegram, Slack, Discord), the reply animates a brief
  // "composing" draft that finalizes into the real message — a live preview — instead of a single
  // silent send. The reply text is a finished string here (no token stream), so streamDraftViaChannel
  // opens a placeholder then commits the true text via finalizeDraft; its intermediate-edit throttle
  // is sized to THAT channel's edit rate limit (editMessageText / chat.update / message PATCH) so a
  // fast finalize can't trip it. Channels that can't edit a message (iMessage/WhatsApp) degrade to
  // the unchanged single send.
  handle('connector:reply', async (_e, { threadKey, text, inReplyTo }: { threadKey: string; text: string; inReplyTo?: string }) => {
    requireSecurityRole('owner', 'send a connector reply')
    const address = parseThreadAddress(threadKey)
    const adapter = resolveAdapter(channelAdapters, address)
    const streamed = Boolean(adapter?.supportsDraftStreaming && adapter.sendDraft && adapter.finalizeDraft)
    // Space intermediate draft edits by the THIS channel's own edit rate limit (Telegram 1.5s,
    // Slack 1.2s, Discord 1s) so a fast finalize/stream can't trip that platform's edit ceiling.
    const result = streamed
      ? await streamDraftViaChannel(channelAdapters, threadKey, composingDraftStream(text, inReplyTo), makeDraftEditThrottle(draftEditIntervalForPlatform(address?.platform)))
      : await sendViaChannel(channelAdapters, threadKey, { text, inReplyTo })
    await auditEvent('connector.reply', { threadKey: auditPreview(threadKey), ok: result.ok, chars: String(text ?? '').length, streamed })
    return result
  })

  // Rich in-channel approval: prompt the user IN the channel (Slack/Discord/Telegram get
  // inline buttons; the rest get a plain-text "/approve" prompt + the in-app gate) and
  // block until they answer. Returns the normalized ChannelApprovalResult.
  handle('connector:requestApproval', async (_e, { threadKey, request }: { threadKey: string; request: ChannelApprovalRequest }) => {
    requireSecurityRole('owner', 'request an in-channel approval')
    const result = await requestChannelApproval(threadKey, request)
    await auditEvent('connector.approval_channel', { threadKey: auditPreview(threadKey), ok: result.ok, decision: result.ok ? result.decision : null, fallbackToApp: result.ok ? false : Boolean(result.fallbackToApp) })
    return result
  })
  handle('assistant:snapshot', async () => {
    // Legacy pending drafts acquire a host-authored exact review snapshot on read.
    store.connectorEvents = store.connectorEvents.map(event => isOutboundSendRequest(event) && !event.delivery ? sealConnectorOutboundEvent(event) : event)
    const defaultProjectId = store.settings.defaultCwd
      ? store.projects.find((project) => !project.archivedAt && project.cwd === store.settings.defaultCwd)?.id ?? null
      : null
    return buildAssistantControlSnapshot(store.connectorEvents, store.settings.email, store.settings.botControl, store.settings.controller, store.settings.calendar, store.settings.documents, store.settings.slack, store.settings.browser, store.settings.imessage, store.settings.whatsapp, store.settings.discord, store.settings.mattermost, store.settings.matrix, store.settings.googlechat, {
      projects: store.projects,
      missions: store.missions,
      existingMemoryCandidates: store.projectMemoryCandidates,
      defaultProjectId,
      whatsappRuntime: getWhatsAppBridgeRuntime(),
    })
  })

  handle('assistant:interaction', async (_e, input: ConnectorInteractionInput) => {
    requireSecurityRole('operator', 'record assistant connector interactions')
    const event = recordConnectorInteraction(input)
    await rememberConnectorEvents([event])
    return event
  })

  handle('assistant:resolveApproval', async (_e, { eventId, decision, reviewDigest }: { eventId: string; decision: 'approve' | 'deny'; reviewDigest?: string }) => {
    requireSecurityRole('operator', 'resolve assistant connector approvals')
    if (!['approve', 'deny'].includes(decision)) throw new Error('Choose approve or deny.')
    const candidate = store.connectorEvents.find((item) => item.id === eventId || item.approvalId === eventId)
    if (!candidate) throw new Error('Connector approval not found.')
    const outbound = isOutboundSendRequest(candidate)
    if (decision === 'approve' && outbound) requireSecurityRole('owner', 'approve an outbound connector send')
    // Settled/in-flight decisions return immediately, including reentrant callbacks. A
    // sending or uncertain record never becomes pending just because a user clicks again.
    if (candidate.delivery || (!candidate.requiresApproval && candidate.approvalStatus !== 'pending')) return candidate
    return connectorDecisionQueue.run(candidate.id, async () => {
      const event = store.connectorEvents.find((item) => item.id === candidate.id)
      if (!event) throw new Error('Connector approval not found.')
      if (event.delivery || (!event.requiresApproval && event.approvalStatus !== 'pending')) return event
      const policy = connectorPolicyForEvent(event)
      let resolved: ConnectorEvent
      if (isOutboundSendRequest(event)) {
        // External sends are single-content approvals, never silently remembered allow
        // rules for future bodies/recipients. The protected one-shot marker is durable
        // before the exact approved snapshot reaches any adapter.
        resolved = decision === 'approve'
          ? await reviewedDelivery.approve(event, reviewDigest, auditActor())
          : { ...event, approvalStatus: 'denied', requiresApproval: false, resolvedAt: Date.now(), resolvedBy: auditActor() }
        if (decision === 'deny') await reviewedDelivery.durable(resolved)
      } else {
        const savedRule = await callbacks.rememberApprovalRule('connector', policy.key, decision === 'approve' ? 'allow' : 'deny', 'always', {
          label: policy.label, explanation: policy.explanation, workspace: `connector:${event.connectorId}`,
        })
        if (!savedRule) throw new Error('Connector approval policy could not be saved with a bounded workspace scope.')
        if (!store.connectorEvents.includes(event)) throw new Error('Connector approval changed while saving the decision. Review the current draft again.')
        resolved = { ...event, approvalStatus: decision === 'approve' ? 'approved' : 'denied', requiresApproval: false, resolvedAt: Date.now(), resolvedBy: auditActor() }
        await reviewedDelivery.durable(resolved)
      }
      if (event.threadKey) resolveChannelApprovalByThread(event.threadKey, decision)
      await auditEvent('connector.approval_resolved', { eventId: resolved.id, connectorId: resolved.connectorId, action: resolved.action, decision, delivery: resolved.delivery?.status ?? null, policyKey: policy.key })
      await rememberConnectorEvents([recordConnectorInteraction({
        connectorId: resolved.connectorId, pluginName: resolved.pluginName,
        action: decision === 'approve' ? 'approval_approved' : 'approval_denied', direction: 'internal', target: resolved.threadKey,
        input: { title: resolved.title, action: resolved.action },
      })])
      return resolved
    })
  })

  // Per-connector handler registrations.
  registerEmailIpc(guardedDeps)
  registerIMessageIpc(guardedDeps)
  registerWhatsAppIpc(guardedDeps)
  registerDiscordIpc(guardedDeps)
  registerMattermostIpc(guardedDeps)
  registerMatrixIpc(guardedDeps)
  registerGoogleChatIpc(guardedDeps)
  registerSlackIpc(guardedDeps)
  registerCalendarIpc(guardedDeps)
  registerDocumentIpc(guardedDeps)
  registerBrowserIpc(guardedDeps)
}
