import assert from 'node:assert/strict'
import { createWebhookServer } from './webhook-server'
import { signWhatsAppCloudBody } from './whatsapp-cloud-webhook'
import type { BotControlIngestInput, WhatsAppBotControlSettings } from './types'

const settings: WhatsAppBotControlSettings = {
  enabled: true,
  accessToken: 'access-token',
  phoneNumberId: '998877',
  verifyToken: 'verify-token',
  appSecret: 'meta-app-secret',
  allowedChatIds: '447700900123',
  lastEventAt: null,
  lastError: null,
}
const accepted: BotControlIngestInput[] = []
const server = createWebhookServer({
  getEndpoints: () => [],
  onConnectorEvent: () => {},
  runAutomationById: () => {},
  audit: () => {},
  getWhatsAppSettings: () => settings,
  onWhatsAppBotInput: (input) => {
    if (input.senderId !== '447700900123') throw new Error('principal not allowed')
    accepted.push(input)
  },
})

const challenge = server.handleWhatsAppChallenge(new URLSearchParams({
  'hub.mode': 'subscribe',
  'hub.verify_token': 'verify-token',
  'hub.challenge': '4321',
}))
assert.equal(challenge.code, 200)
assert.equal(challenge.body, '4321')
assert.equal(challenge.raw, true)
assert.equal(server.handleWhatsAppChallenge(new URLSearchParams({ 'hub.mode': 'subscribe', 'hub.verify_token': 'wrong', 'hub.challenge': '4321' })).code, 401)

function body(from = '447700900123', phone = '998877') {
  return Buffer.from(JSON.stringify({
    object: 'whatsapp_business_account',
    entry: [{ changes: [{ field: 'messages', value: {
      metadata: { phone_number_id: phone },
      messages: [{ id: `wamid.${from}`, from, type: 'text', text: { body: 'rooms' } }],
    } }] }],
  }))
}

const valid = body()
assert.equal(server.handleWhatsAppDelivery({ method: 'POST', contentType: 'application/json', signature: signWhatsAppCloudBody(valid, settings.appSecret), rawBody: valid }).code, 200)
assert.equal(accepted.length, 1)
assert.equal(accepted[0].senderId, '447700900123', 'principal is derived from the signed Meta payload')

const forged = body('447700900999')
assert.equal(server.handleWhatsAppDelivery({ method: 'POST', contentType: 'application/json', signature: signWhatsAppCloudBody(forged, 'attacker-secret'), rawBody: forged }).code, 401)
assert.equal(accepted.length, 1, 'a forged signature cannot reach bot ingestion')

assert.equal(server.handleWhatsAppDelivery({ method: 'POST', contentType: 'application/json', signature: signWhatsAppCloudBody(forged, settings.appSecret), rawBody: forged }).code, 200)
assert.equal(accepted.length, 1, 'a valid but non-allowlisted signed principal is rejected by the ingress callback')

const otherEndpoint = body('447700900123', 'attacker-phone')
assert.equal(server.handleWhatsAppDelivery({ method: 'POST', contentType: 'application/json', signature: signWhatsAppCloudBody(otherEndpoint, settings.appSecret), rawBody: otherEndpoint }).code, 200)
assert.equal(accepted.length, 1, 'messages for another configured phone-number principal are ignored')

await server.syncFromEndpoints()
const status = server.status()
assert.ok(status.listening && status.port, 'signed WhatsApp configuration starts the loopback webhook listener')
assert.equal(status.routes, 1)
const challengeResponse = await fetch(`http://127.0.0.1:${status.port}/integrations/whatsapp?hub.mode=subscribe&hub.verify_token=verify-token&hub.challenge=socket-ok`)
assert.equal(challengeResponse.status, 200)
assert.equal(await challengeResponse.text(), 'socket-ok', 'the Meta challenge is returned as raw text, not a JSON-quoted string')
const rejectedResponse = await fetch(`http://127.0.0.1:${status.port}/integrations/whatsapp`, {
  method: 'POST',
  headers: { 'content-type': 'application/json', 'x-hub-signature-256': signWhatsAppCloudBody(valid, 'wrong') },
  body: valid,
})
assert.equal(rejectedResponse.status, 401)
await server.stop()

console.log('whatsapp cloud server tests passed')
