import { createHmac, timingSafeEqual } from 'node:crypto'
import type { BotControlIngestInput } from './types'

export const WHATSAPP_CLOUD_WEBHOOK_PATH = '/integrations/whatsapp'

function constantTimeTextEqual(left: string, right: string): boolean {
  const a = Buffer.from(left, 'utf8')
  const b = Buffer.from(right, 'utf8')
  return a.length === b.length && a.length > 0 && timingSafeEqual(a, b)
}

/** Meta subscription challenge. The verify token proves the configuration request's principal. */
export function verifyWhatsAppCloudChallenge(
  query: URLSearchParams,
  verifyToken: string,
): string | null {
  const mode = query.get('hub.mode') ?? ''
  const token = query.get('hub.verify_token') ?? ''
  const challenge = query.get('hub.challenge') ?? ''
  if (mode !== 'subscribe' || !verifyToken || !constantTimeTextEqual(token, verifyToken)) return null
  if (!challenge || challenge.length > 512 || /[\r\n\0]/.test(challenge)) return null
  return challenge
}

/** Meta signs the exact raw request body as `sha256=<hex>` with the App Secret. */
export function verifyWhatsAppCloudSignature(rawBody: Buffer, signature: string | undefined, appSecret: string): boolean {
  if (!appSecret || !signature?.startsWith('sha256=')) return false
  const hex = signature.slice('sha256='.length)
  if (!/^[a-f0-9]{64}$/i.test(hex)) return false
  const provided = Buffer.from(hex, 'hex')
  const expected = createHmac('sha256', appSecret).update(rawBody).digest()
  return provided.length === expected.length && timingSafeEqual(provided, expected)
}

function objectValue(value: unknown): Record<string, any> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, any> : null
}

function messageText(message: Record<string, any>): string {
  if (message.type === 'text' && typeof message.text?.body === 'string') return message.text.body.trim()
  if (message.type === 'button' && typeof message.button?.text === 'string') return message.button.text.trim()
  if (message.type === 'interactive') {
    const title = message.interactive?.button_reply?.title ?? message.interactive?.list_reply?.title
    return typeof title === 'string' ? title.trim() : ''
  }
  return ''
}

/**
 * Derive command inputs only from a Meta-signed payload. The caller must verify the raw-body
 * signature first; this parser then binds every input to the configured phone-number endpoint,
 * the signed messages[].from principal, and Meta's stable message id.
 */
export function parseWhatsAppCloudWebhook(rawBody: Buffer, phoneNumberId: string): BotControlIngestInput[] {
  let parsed: unknown
  try { parsed = JSON.parse(rawBody.toString('utf8')) } catch { throw new Error('Invalid WhatsApp webhook JSON.') }
  const root = objectValue(parsed)
  if (!root || root.object !== 'whatsapp_business_account' || !Array.isArray(root.entry)) {
    throw new Error('Invalid WhatsApp webhook envelope.')
  }

  const inputs: BotControlIngestInput[] = []
  for (const rawEntry of root.entry) {
    const entry = objectValue(rawEntry)
    if (!entry || !Array.isArray(entry.changes)) continue
    for (const rawChange of entry.changes) {
      const change = objectValue(rawChange)
      const value = objectValue(change?.value)
      if (!change || change.field !== 'messages' || !value) continue
      const endpointId = String(value.metadata?.phone_number_id ?? '')
      if (!phoneNumberId || endpointId !== phoneNumberId) continue
      const contacts = Array.isArray(value.contacts) ? value.contacts : []
      const names = new Map<string, string>()
      for (const rawContact of contacts) {
        const contact = objectValue(rawContact)
        const waId = String(contact?.wa_id ?? '')
        const name = typeof contact?.profile?.name === 'string' ? contact.profile.name.trim() : ''
        if (waId && name) names.set(waId, name.slice(0, 160))
      }
      for (const rawMessage of Array.isArray(value.messages) ? value.messages : []) {
        const message = objectValue(rawMessage)
        if (!message) continue
        const senderId = String(message.from ?? '')
        const messageId = String(message.id ?? '')
        const text = messageText(message)
        if (!/^\d{5,32}$/.test(senderId) || !messageId || messageId.length > 256 || !text) continue
        inputs.push({
          channel: 'whatsapp',
          chatId: senderId,
          senderId,
          eventId: `whatsapp:${messageId}`,
          text,
          senderName: names.get(senderId),
          payload: { message_id: messageId, phone_number_id: endpointId },
        })
      }
    }
  }
  return inputs
}

export function signWhatsAppCloudBody(rawBody: Buffer | string, appSecret: string): string {
  return `sha256=${createHmac('sha256', appSecret).update(rawBody).digest('hex')}`
}
