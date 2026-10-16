import { randomUUID } from 'node:crypto'
import type { ConnectorEvent, WebhookEndpoint } from '../../agent/types'
import { recordConnectorInteraction } from '../../agent/connector-spine'
import { auditPreview, isSafeExternalUrl, assertOutboundWebhookUrlResolved, isSecretMask } from '../../agent/security'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { handle } from './ipc'
import { persistWebhooks } from './persistence'
import { deliverOutboundWebhook, recordWebhookDelivery, webhookSnapshot as redactedWebhookSnapshot } from './webhooks'
import { stampWebhookSecretRotation } from '../../agent/secret-rotation'
import { normalizeWebhookReplayWindowSeconds } from '../../agent/webhook-server'
import { assertUniqueWebhookPath, assertWebhookApprovalCurrent, canonicalWebhookPath, captureWebhookApproval, nextWebhookApproval, nextWebhookApprovalRequired, resolveWebhookAutomationId, sanitizeWebhookSave, stampWebhookApproval } from './public-config-policy'
import { createWebhookMutationQueue } from './webhook-mutation-queue'

interface WebhookIpcDeps {
  webhookServer: {
    status(): Parameters<typeof redactedWebhookSnapshot>[0]
    syncFromEndpoints(): Promise<void>
  }
  rememberConnectorEvents(events: ConnectorEvent[]): void
}

export function registerWebhookIpc({ webhookServer, rememberConnectorEvents }: WebhookIpcDeps) {
  const webhookSnapshot = () => redactedWebhookSnapshot(webhookServer.status())
  const webhookMutations = createWebhookMutationQueue()

  handle('webhooks:get', async () => webhookSnapshot())
  handle('webhooks:save', async (_e, requested: Partial<WebhookEndpoint>) => {
    requireSecurityRole('owner', 'save webhooks')
    return webhookMutations.run(async () => {
    const w = sanitizeWebhookSave(requested)
    const now = Date.now()
    if (w.id) {
      const ex = store.webhooks.find((x) => x.id === w.id)
      if (!ex) throw new Error('Webhook not found.')
      const secretRef = isSecretMask(w.secretRef) ? ex.secretRef : (w.secretRef ?? ex.secretRef)
      const { id: _id, ...config } = w
      // Build and validate a copy before touching live state. Host-owned evidence, identity,
      // timestamps, signature mode, and secret-rotation metadata never come from this patch.
      let next: WebhookEndpoint = {
        ...ex,
        ...config,
        id: ex.id,
        createdAt: ex.createdAt,
        deliveries: ex.deliveries,
        signatureMode: 'hmac-sha256',
        secretRotations: ex.secretRotations,
        secretRef,
        updatedAt: now,
      }
      next.replayWindowSeconds = normalizeWebhookReplayWindowSeconds(next.replayWindowSeconds)
      if (next.direction === 'outbound') {
        next.url = (next.url ?? '').trim()
        if (!next.url) throw new Error('Outbound webhooks require a URL.')
        next.path = undefined
        next.eventKinds = [...new Set(next.eventKinds ?? [])]
        next.approvalRequired = nextWebhookApprovalRequired(ex, next.direction, Object.prototype.hasOwnProperty.call(config, 'approvalRequired') ? config.approvalRequired : undefined)
        next.runAutomationId = null
        next.approvedAt = nextWebhookApproval(ex, next)
        if (!isSafeExternalUrl(next.url, { allowHttpLocalhost: true })) throw new Error('Outbound webhook URL must use https, except localhost during development.')
        await assertOutboundWebhookUrlResolved(next.url)
      } else {
        next.path = next.path ? canonicalWebhookPath(next.path) : canonicalWebhookPath(`/hooks/${Math.random().toString(36).slice(2, 8)}`)
        assertUniqueWebhookPath(store.webhooks, next.path, ex.id)
        next.url = undefined
        next.eventKinds = []
        next.approvalRequired = false
        next.approvedAt = null
        next.runAutomationId = resolveWebhookAutomationId(next, store.automations)
      }
      next = stampWebhookSecretRotation(next, ex, now)
      // Delivery records can arrive from the outbound event spine outside renderer IPC while DNS
      // validation is pending. Re-read that host-owned evidence at commit time.
      next.deliveries = ex.deliveries
      Object.assign(ex, next)
      persistWebhooks()
      await webhookServer.syncFromEndpoints()
      await auditEvent('webhook.save', { id: ex.id, enabled: ex.enabled, direction: ex.direction, target: auditPreview(ex.direction === 'outbound' ? ex.url : ex.path) })
      return webhookSnapshot()
    }
    const direction = w.direction || 'inbound'
    const outboundUrl = direction === 'outbound' ? (w.url ?? '').trim() : ''
    if (direction === 'outbound' && !outboundUrl) throw new Error('Outbound webhooks require a URL.')
    if (direction === 'outbound' && !isSafeExternalUrl(outboundUrl, { allowHttpLocalhost: true })) {
      throw new Error('Outbound webhook URL must use https, except localhost during development.')
    }
    if (direction === 'outbound') {
      await assertOutboundWebhookUrlResolved(outboundUrl)
    }
    let inboundPath = direction === 'inbound' && w.path ? canonicalWebhookPath(w.path) : canonicalWebhookPath(`/hooks/${Math.random().toString(36).slice(2, 8)}`)
    if (direction === 'inbound') assertUniqueWebhookPath(store.webhooks, inboundPath)
    const runAutomationId = resolveWebhookAutomationId({ direction, triggerPolicy: w.triggerPolicy ?? 'notify', runAutomationId: w.runAutomationId }, store.automations)
    let endpoint: WebhookEndpoint = {
      id: 'wh_' + Math.random().toString(36).slice(2, 9),
      name: w.name || 'Webhook', direction, enabled: w.enabled ?? false, scope: w.scope || {},
      path: direction === 'inbound' ? inboundPath : undefined,
      url: direction === 'outbound' ? outboundUrl : undefined,
      secretRef: isSecretMask(w.secretRef) ? '' : (w.secretRef || ''),
      signatureMode: 'hmac-sha256', replayWindowSeconds: normalizeWebhookReplayWindowSeconds(w.replayWindowSeconds),
      capturePolicy: w.capturePolicy || 'redacted_payload', triggerPolicy: w.triggerPolicy || 'notify',
      runAutomationId,
      eventKinds: direction === 'outbound' ? (w.eventKinds ?? []) : [],
      approvalRequired: direction === 'outbound' ? (w.approvalRequired ?? true) : false,
      approvedAt: null,
      deliveries: [],
      createdAt: now, updatedAt: now,
    }
    endpoint = stampWebhookSecretRotation(endpoint, null, now)
    store.webhooks.unshift(endpoint)
    persistWebhooks()
    await webhookServer.syncFromEndpoints()
    await auditEvent('webhook.create', { id: endpoint.id, direction: endpoint.direction, target: auditPreview(endpoint.direction === 'outbound' ? endpoint.url : endpoint.path) })
    return webhookSnapshot()
    })
  })
  handle('webhooks:approve', async (_e, requestedId: string) => {
    requireSecurityRole('owner', 'approve outbound webhook destinations')
    return webhookMutations.run(async () => {
    const { id } = sanitizeWebhookSave({ id: requestedId })
    const endpoint = store.webhooks.find((candidate) => candidate.id === id)
    if (!endpoint) throw new Error('Webhook not found.')
    if (endpoint.direction !== 'outbound' || endpoint.approvalRequired === false) throw new Error('This webhook does not require outbound destination approval.')
    const url = (endpoint.url ?? '').trim()
    if (!url || !isSafeExternalUrl(url, { allowHttpLocalhost: true })) throw new Error('Outbound webhook URL is not approvable.')
    const expected = captureWebhookApproval(endpoint)
    await assertOutboundWebhookUrlResolved(url)
    const current = store.webhooks.find((candidate) => candidate.id === endpoint.id)
    if (current !== endpoint) throw new Error('Webhook destination changed while approval was being checked. Review and approve it again.')
    assertWebhookApprovalCurrent(current, expected)
    const now = Date.now()
    Object.assign(endpoint, stampWebhookApproval(endpoint, now))
    persistWebhooks()
    await webhookServer.syncFromEndpoints()
    await auditEvent('webhook.approve', { id: endpoint.id, target: auditPreview(url) })
    return webhookSnapshot()
    })
  })
  handle('webhooks:delete', async (_e, id: string) => {
    requireSecurityRole('owner', 'delete webhooks')
    return webhookMutations.run(async () => {
    store.webhooks = store.webhooks.filter((w) => w.id !== id)
    persistWebhooks()
    await webhookServer.syncFromEndpoints()
    return webhookSnapshot()
    })
  })
  handle('webhooks:test', async (_e, id: string) => {
    requireSecurityRole('operator', 'test webhooks')
    return webhookMutations.run(async () => {
    const w = store.webhooks.find((x) => x.id === id)
    if (!w) return webhookSnapshot()
    if (w.direction === 'outbound') {
      const record = await deliverOutboundWebhook(w, 'test', { title: 'Test delivery', summary: `Sample outbound payload for ${w.name}`, sample: true })
      recordWebhookDelivery(w.id, record)
      await auditEvent('webhook.outbound.test', { id: w.id, status: record.status, httpStatus: record.httpStatus })
    } else {
      const event = recordConnectorInteraction({ connectorId: 'webhook', pluginName: w.name, action: 'webhook_test', direction: 'inbound', target: w.path, input: { title: 'Test delivery', summary: `Sample inbound payload for ${w.name}`, sample: true } })
      rememberConnectorEvents([event])
      recordWebhookDelivery(w.id, {
        id: `whd_${randomUUID()}`,
        direction: 'inbound',
        eventKind: 'test',
        urlOrPath: w.path || '',
        status: 'delivered',
        attempts: 1,
        httpStatus: 200,
        error: null,
        idempotencyKey: `whin_${randomUUID()}`,
        payloadPreview: auditPreview(JSON.stringify({ title: 'Test delivery', summary: `Sample inbound payload for ${w.name}`, sample: true }), 900),
        createdAt: Date.now(),
        finishedAt: Date.now(),
      })
    }
    return webhookSnapshot()
    })
  })
}
