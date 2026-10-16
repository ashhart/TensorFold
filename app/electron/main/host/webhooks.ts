import { app } from 'electron'
import { randomUUID } from 'node:crypto'
import type { WebhookDeliveryRecord, WebhookEndpoint, WebhookEventKind, WebhookServerStatus } from '../../agent/types'
import { auditPreview, isSafeExternalUrl, assertOutboundWebhookUrlResolved, SECRET_MASK } from '../../agent/security'
import { discardResponseBody, pinnedFetch } from '../../agent/pinned-fetch'
import { signWebhook } from '../../agent/webhook-server'
import { store } from '../state'
import { auditEvent } from './audit'
import { persistWebhooks } from './persistence'

export function compactWebhookData(kind: WebhookEventKind, data: unknown): Record<string, unknown> {
  const obj = data && typeof data === 'object' ? data as Record<string, any> : {}
  if (kind === 'agent') return {
    type: obj.type,
    threadId: obj.threadId,
    status: obj.status,
    item: obj.item ? { kind: obj.item.kind, title: obj.item.title, path: obj.item.path, state: obj.item.state, meta: obj.item.meta } : undefined,
  }
  if (kind === 'room') return {
    roomId: obj.roomId,
    type: obj.type,
    seat: obj.seat,
    tick: obj.tick,
    text: auditPreview(obj.text, 1000),
  }
  if (kind === 'ticket') return {
    id: obj.id,
    projectId: obj.projectId,
    title: obj.title,
    status: obj.status,
    priority: obj.priority,
    runState: obj.runState,
  }
  if (kind === 'automation') return {
    id: obj.id,
    name: obj.name,
    enabled: obj.enabled,
    lastStatus: obj.lastStatus,
    lastRunAt: obj.lastRunAt,
    nextRunAt: obj.nextRunAt,
  }
  if (kind === 'mission') return {
    id: obj.id,
    title: obj.title,
    status: obj.status,
    runBudgetClass: obj.runBudgetClass,
    heartbeat: obj.heartbeat ? { count: obj.heartbeat.count, lastBeatAt: obj.heartbeat.lastBeatAt, missedSince: obj.heartbeat.missedSince } : undefined,
    checkpointCount: Array.isArray(obj.checkpoints) ? obj.checkpoints.length : 0,
    pendingApprovalCount: Array.isArray(obj.pendingApprovals) ? obj.pendingApprovals.length : 0,
  }
  if (kind === 'connector') return {
    id: obj.id,
    connectorId: obj.connectorId,
    action: obj.action,
    direction: obj.direction,
    title: obj.title,
    summary: obj.summary,
    priority: obj.priority,
    trustLevel: obj.trustLevel,
    requiresApproval: obj.requiresApproval,
  }
  return {
    title: 'Test delivery',
    summary: 'Sample outbound webhook payload from TensorFold.',
  }
}

export function recordWebhookDelivery(endpointId: string, record: WebhookDeliveryRecord): void {
  const endpoint = store.webhooks.find((w) => w.id === endpointId)
  if (!endpoint) return
  endpoint.deliveries = [record, ...(endpoint.deliveries ?? [])].slice(0, 30)
  endpoint.updatedAt = Date.now()
  persistWebhooks()
}

export async function deliverOutboundWebhook(endpoint: WebhookEndpoint, kind: WebhookEventKind, data: unknown): Promise<WebhookDeliveryRecord> {
  const now = Date.now()
  const idempotencyKey = `whdel_${randomUUID()}`
  const url = endpoint.url || ''
  const baseRecord: WebhookDeliveryRecord = {
    id: `whd_${randomUUID()}`,
    direction: 'outbound',
    eventKind: kind,
    urlOrPath: url,
    status: 'queued',
    attempts: 0,
    httpStatus: null,
    error: null,
    idempotencyKey,
    payloadPreview: '',
    createdAt: now,
    finishedAt: null,
  }
  const payload = {
    id: idempotencyKey,
    type: kind,
    createdAt: now,
    source: 'vontra',
    endpointId: endpoint.id,
    data: compactWebhookData(kind, data),
  }
  const raw = Buffer.from(JSON.stringify(payload), 'utf8')
  baseRecord.payloadPreview = auditPreview(JSON.stringify(payload), 900)
  if (endpoint.approvalRequired !== false && !endpoint.approvedAt) {
    return { ...baseRecord, status: 'failed', error: 'Destination needs approval before outbound delivery.', finishedAt: Date.now() }
  }
  if (!endpoint.url || !endpoint.secretRef) {
    return { ...baseRecord, status: 'failed', error: 'Outbound URL and HMAC secret are required.', finishedAt: Date.now() }
  }
  if (!isSafeExternalUrl(endpoint.url, { allowHttpLocalhost: true })) {
    return { ...baseRecord, status: 'failed', error: 'Outbound webhook URL must use https, except localhost during development.', finishedAt: Date.now() }
  }
  // isSafeExternalUrl only enforces https — block private/internal/metadata hosts (SSRF) here, while
  // still permitting loopback for the documented localhost dev case. DNS-resolved (not just a
  // literal-hostname check) so a hostname that has been rebound to an internal address since
  // approval is still caught on EVERY delivery, not just at save time.
  let pinnedTarget
  try {
    pinnedTarget = await assertOutboundWebhookUrlResolved(endpoint.url)
  } catch (e: any) {
    return { ...baseRecord, status: 'failed', error: String(e?.message || e || 'Outbound webhook URL is invalid.'), finishedAt: Date.now() }
  }
  let last: WebhookDeliveryRecord = baseRecord
  for (let attempt = 1; attempt <= 3; attempt++) {
    const timestamp = String(Math.floor(Date.now() / 1000))
    try {
      const res = await pinnedFetch(pinnedTarget, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'user-agent': `TensorFold/${app.getVersion()}`,
          'x-vontra-timestamp': timestamp,
          'x-vontra-idempotency': idempotencyKey,
          'x-vontra-signature': signWebhook(endpoint.secretRef, timestamp, raw),
        },
        body: raw,
        redirect: 'error', // don't let a public URL 30x-redirect into an internal/metadata host
        signal: AbortSignal.timeout(8000),
      })
      last = { ...baseRecord, attempts: attempt, httpStatus: res.status, status: res.ok ? 'delivered' : 'failed', error: res.ok ? null : `HTTP ${res.status}`, finishedAt: Date.now() }
      await discardResponseBody(res)
      if (res.ok || (res.status < 500 && res.status !== 408 && res.status !== 429)) return last
    } catch (e: any) {
      last = { ...baseRecord, attempts: attempt, status: 'failed', error: String(e?.message || e || 'Delivery failed.'), finishedAt: Date.now() }
    }
    if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, 250 * attempt))
  }
  return last
}

function webhookAllowsEvent(endpoint: WebhookEndpoint, kind: WebhookEventKind): boolean {
  if (kind === 'test') return true
  const filters = endpoint.eventKinds ?? []
  return filters.length === 0 || filters.includes(kind)
}

export function emitOutboundWebhooks(kind: WebhookEventKind, data: unknown): void {
  const endpoints = store.webhooks.filter((w) => w.direction === 'outbound' && w.enabled && webhookAllowsEvent(w, kind))
  for (const endpoint of endpoints) {
    void deliverOutboundWebhook(endpoint, kind, data).then((record) => {
      recordWebhookDelivery(endpoint.id, record)
      void auditEvent('webhook.outbound.delivery', { id: endpoint.id, kind, status: record.status, httpStatus: record.httpStatus })
    })
  }
}

const redactWebhook = (w: WebhookEndpoint): WebhookEndpoint => ({ ...w, secretRef: w.secretRef ? SECRET_MASK : '' })

export function webhookSnapshot(listener: WebhookServerStatus) {
  return { endpoints: store.webhooks.map(redactWebhook), listener }
}
