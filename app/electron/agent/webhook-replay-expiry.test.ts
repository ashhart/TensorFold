import assert from 'node:assert/strict'
import { createWebhookServer, signWebhook, type WebhookDelivery } from './webhook-server'
import type { WebhookEndpoint } from './types'

const originalNow = Date.now
let now = 1_800_000_000_000
Date.now = () => now

const endpoint: WebhookEndpoint = {
  id: 'signed-hook', name: 'Local replay fixture', direction: 'inbound', enabled: true,
  scope: {}, path: '/hooks/fixture', secretRef: 'fixture-signing-key', signatureMode: 'hmac-sha256',
  replayWindowSeconds: 300, capturePolicy: 'metadata', triggerPolicy: 'run_automation',
  runAutomationId: 'fixture-automation', createdAt: 0, updatedAt: 0,
}
let invocations = 0
const server = createWebhookServer({
  getEndpoints: () => [endpoint], onConnectorEvent: () => {}, audit: () => {},
  runAutomationById: () => { invocations++ },
})

try {
  const rawBody = Buffer.from('{"title":"Fixture delivery"}')
  const timestamp = String(now / 1000 + 290)
  const delivery: WebhookDelivery = {
    pathname: endpoint.path!, method: 'POST', contentType: 'application/json', rawBody, timestamp,
    signature: signWebhook(endpoint.secretRef!, timestamp, rawBody), idempotency: 'first',
  }
  assert.equal(server.handleDelivery(delivery).code, 202)
  now += 301_000
  assert.equal(server.handleDelivery({ ...delivery, idempotency: 'replay' }).code, 200,
    'A captured future-dated delivery must remain deduped for its entire acceptance window')
  assert.equal(invocations, 1, 'A replay must never rerun the configured automation')
  now = Number(timestamp) * 1000 + 300_000
  assert.equal(server.handleDelivery(delivery).code, 200,
    'The inclusive timestamp boundary must remain covered by deduplication')
  now++
  assert.equal(server.handleDelivery(delivery).code, 401)

  const immediateTimestamp = String(now / 1000)
  const immediate = { ...delivery, timestamp: immediateTimestamp,
    signature: signWebhook(endpoint.secretRef!, immediateTimestamp, rawBody) }
  assert.equal(server.handleDelivery(immediate).code, 202)
  now += 300_000
  assert.equal(server.handleDelivery(immediate).code, 200,
    'A delivery signed at receipt time must also remain deduped at the inclusive boundary')
  assert.equal(invocations, 2)
  console.log('Webhook replay expiry: future skew and inclusive boundaries passed')
} finally {
  Date.now = originalNow
}
