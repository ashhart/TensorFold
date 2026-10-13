import assert from 'node:assert/strict'
import {
  parseWhatsAppCloudWebhook,
  signWhatsAppCloudBody,
  verifyWhatsAppCloudChallenge,
  verifyWhatsAppCloudSignature,
} from './whatsapp-cloud-webhook'

const APP_SECRET = 'meta-app-secret'
const PHONE_ID = '998877'
const body = Buffer.from(JSON.stringify({
  object: 'whatsapp_business_account',
  entry: [{ changes: [{ field: 'messages', value: {
    metadata: { phone_number_id: PHONE_ID },
    contacts: [{ wa_id: '447700900123', profile: { name: 'Maya' } }],
    messages: [{ id: 'wamid.ABC', from: '447700900123', type: 'text', text: { body: 'rooms' } }],
  } }] }],
}))

assert.equal(verifyWhatsAppCloudSignature(body, signWhatsAppCloudBody(body, APP_SECRET), APP_SECRET), true)
assert.equal(verifyWhatsAppCloudSignature(body, signWhatsAppCloudBody(body, 'wrong'), APP_SECRET), false)
assert.equal(verifyWhatsAppCloudSignature(Buffer.from('{}'), signWhatsAppCloudBody(body, APP_SECRET), APP_SECRET), false)
assert.equal(verifyWhatsAppCloudSignature(body, 'sha256=zz', APP_SECRET), false)

const challenge = new URLSearchParams({ 'hub.mode': 'subscribe', 'hub.verify_token': 'verify-me', 'hub.challenge': '12345' })
assert.equal(verifyWhatsAppCloudChallenge(challenge, 'verify-me'), '12345')
assert.equal(verifyWhatsAppCloudChallenge(challenge, 'wrong'), null)

const inputs = parseWhatsAppCloudWebhook(body, PHONE_ID)
assert.equal(inputs.length, 1)
assert.deepEqual(inputs[0], {
  channel: 'whatsapp',
  chatId: '447700900123',
  senderId: '447700900123',
  eventId: 'whatsapp:wamid.ABC',
  text: 'rooms',
  senderName: 'Maya',
  payload: { message_id: 'wamid.ABC', phone_number_id: PHONE_ID },
})
assert.deepEqual(parseWhatsAppCloudWebhook(body, 'attacker-endpoint'), [], 'signed messages for another phone-number endpoint are ignored')
assert.throws(() => parseWhatsAppCloudWebhook(Buffer.from('{}'), PHONE_ID), /Invalid WhatsApp webhook envelope/)

console.log('whatsapp cloud webhook tests passed')
