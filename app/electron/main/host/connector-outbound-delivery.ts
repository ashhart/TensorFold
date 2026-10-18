import { createHash, randomUUID } from 'node:crypto'
import { createSerialTaskQueue } from '../../agent/serial-task-queue'
import { sendViaChannel, resolveAdapter, type ChannelAdapter } from '../../agent/channel-adapter'
import { parseThreadAddress } from '../../agent/thread-address'
import { saveConnectorEvents } from '../../store'
import { store } from '../state'
import type { ConnectorEvent, ConnectorOutboundSnapshot } from '../../agent/types'

// All app/channel decisions for the same event share this queue. A transport result is
// not permission to retry: sending/uncertain records permanently consume this attempt.
export const connectorDecisionQueue = createSerialTaskQueue()
const PERSONAL_ACTIONS = new Set(['message_send_requested', 'message_reply_requested'])
export function isOutboundSendRequest(event: ConnectorEvent): boolean {
  return (['telegram', 'whatsapp', 'imessage'].includes(event.connectorId) && PERSONAL_ACTIONS.has(event.action))
    || (event.connectorId === 'email' && ['send_requested', 'reply_requested'].includes(event.action))
}

function accountRevision(connectorId: string, accountId: string | null): string {
  const settings = store.settings
  let identity: unknown
  if (connectorId === 'whatsapp') {
    const source = settings.whatsapp?.sources.find(item => item.id === accountId)
    if (!source || !source.enabled || !source.paired) throw new Error('Choose an enabled linked WhatsApp account before sending.')
    // Transient sync/error status is deliberately excluded; identity and consent scope
    // remain bound while a healthy socket reconnects.
    identity = { id: source.id, enabled: source.enabled, paired: source.paired, phoneNumber: source.phoneNumber, allowedChatJids: source.allowedChatJids, allowSelfChat: source.allowSelfChat, credentialRotatedAt: source.credentialRotatedAt }
  } else if (connectorId === 'telegram') {
    const source = settings.botControl?.telegram
    if (!source?.enabled || !source.botToken?.trim()) throw new Error('Connect and enable the selected Telegram account before sending.')
    const chats = String(source.allowedChatIds ?? '').split(/[\s,;]+/).filter(Boolean)
    identity = { botToken: source.botToken, allowedChatIds: chats, enabled: source.enabled }
  } else if (connectorId === 'email') {
    const source = accountId ? settings.email?.sources.find(item => item.id === accountId) : null
    if (accountId && (!source || !source.enabled)) throw new Error('Choose an enabled email account before sending.')
    identity = source ? { id: source.id, kind: source.kind, host: source.host, port: source.port, user: source.user, password: source.password, fromAddress: source.fromAddress, enabled: source.enabled } : { kind: 'local_outbox' }
  } else {
    if (!settings.imessage?.enabled) throw new Error('Enable iMessage before reviewing this message.')
    identity = { enabled: settings.imessage.enabled, allowedChatIds: settings.imessage.allowedChatIds }
  }
  return createHash('sha256').update(JSON.stringify(identity)).digest('hex')
}

function boundedText(value: unknown, label: string, max: number, allowEmpty = false): string {
  if (typeof value !== 'string' || (!allowEmpty && !value.trim()) || value.length > max || value.includes('\0')) throw new Error(`Review a valid ${label} before sending.`)
  return value
}

export function outboundSnapshot(event: ConnectorEvent): ConnectorOutboundSnapshot {
  if (!isOutboundSendRequest(event)) throw new Error('This event is not an addressed outbound send.')
  const p = event.payload ?? {}
  const connectorId = event.connectorId as ConnectorOutboundSnapshot['connectorId']
  const recipient = boundedText(connectorId === 'email' ? p.to : p.target, 'recipient', 500).trim()
  const body = boundedText(p.body, 'message body', 100_000)
  const inputThread = typeof p.threadKey === 'string' ? p.threadKey : event.threadKey
  const address = parseThreadAddress(inputThread)
  let accountId: string | null = typeof p.accountId === 'string' && p.accountId.trim() ? p.accountId.trim() : null
  if (connectorId === 'email') {
    accountId = accountId ?? event.sources?.find(source => source.connectorId === 'email')?.sourceId ?? null
    const subject = boundedText(p.subject, 'email subject', 1_000)
    if (!/^[^\s@<>;,]+@[^\s@<>;,]+\.[^\s@<>;,]+$/.test(recipient) || /[\r\n]/.test(subject)) throw new Error('Choose one exact email address and a single-line subject before sending.')
    return { version: 1, connectorId, accountId, accountRevision: accountRevision(connectorId, accountId), recipient, body, subject, threadKey: event.threadKey, ...(typeof p.inReplyTo === 'string' ? { inReplyTo: p.inReplyTo } : {}) }
  }
  const legacyGeneratedKey = inputThread === `${connectorId}:message:${recipient}`
  if (address && !legacyGeneratedKey && address.platform === connectorId && address.conversation) {
    if (address.conversation !== recipient || (accountId && address.account && accountId !== address.account)) throw new Error('The reviewed recipient/account does not match the selected conversation.')
    accountId = accountId ?? address.account
  } else if (inputThread && !legacyGeneratedKey) throw new Error('Choose a conversation that belongs to this connector.')
  if (connectorId === 'whatsapp') {
    if (!accountId || !/^[a-zA-Z0-9_-]{1,100}$/.test(accountId)) throw new Error('Choose an exact linked WhatsApp account before sending.')
    if (!/^\d+(?:-\d+)?@(s\.whatsapp\.net|g\.us|lid)$/.test(recipient)) throw new Error('Choose an exact WhatsApp contact or group ID before sending.')
  } else if (connectorId === 'telegram') {
    if (!/^-?\d+$/.test(recipient)) throw new Error('Choose an exact Telegram chat ID before sending.')
    if (accountId && accountId !== 'bot-control') throw new Error('This Telegram account is unavailable.')
    accountId = 'bot-control'
    const chats = String(store.settings.botControl?.telegram?.allowedChatIds ?? '').split(/[\s,;]+/).filter(Boolean)
    if (!chats.includes(recipient)) throw new Error('Choose a Telegram chat in the current connection scope before sending.')
  } else {
    if (accountId || !/^(?:\+?\d{5,20}|[^\s@<>;,]+@[^\s@<>;,]+\.[^\s@<>;,]+)$/.test(recipient)) throw new Error('Choose an exact iMessage phone number or email address before sending.')
    const grant = store.settings.imessage?.allowedChatIds
    if (typeof grant !== 'string') throw new Error('Repair the saved iMessage conversation scope before sending.')
    const chats = grant.split(/[\s,;]+/).filter(Boolean)
    if (chats.length && !chats.includes(recipient)) throw new Error('Choose an iMessage contact in the current connection scope before sending.')
  }
  const threadKey = accountId ? `${connectorId}:${accountId}:${recipient}` : `${connectorId}:${recipient}`
  return { version: 1, connectorId, accountId, accountRevision: accountRevision(connectorId, accountId), recipient, body, threadKey }
}

export function outboundReviewDigest(event: ConnectorEvent, outbound = outboundSnapshot(event)): string {
  return createHash('sha256').update(JSON.stringify({ eventId: event.id, action: event.action, outbound })).digest('hex')
}

/** Called on host ingest, never accepts renderer-supplied approval snapshots. */
export function sealConnectorOutboundEvent(event: ConnectorEvent): ConnectorEvent {
  if (!isOutboundSendRequest(event)) return event
  try {
    const outbound = Object.freeze(outboundSnapshot(event))
    return { ...event, outbound, reviewDigest: outboundReviewDigest(event, outbound), outboundError: undefined }
  } catch (error) {
    return { ...event, outbound: undefined, reviewDigest: undefined, outboundError: error instanceof Error ? error.message : 'Review a complete addressed message.' }
  }
}

export interface ReviewedDeliveryDeps {
  adapters: ChannelAdapter[]
  sendEmail(event: ConnectorEvent): Promise<NonNullable<ConnectorEvent['delivery']>>
  /** Durable protected write; injectable only for isolated regression tests. */
  persist?: (events: ConnectorEvent[]) => Promise<void>
  now?: () => number
}

/** One durable one-shot attempt for precisely the snapshot the owner reviewed. */
export function createReviewedConnectorDelivery(deps: ReviewedDeliveryDeps) {
  const persist = deps.persist ?? saveConnectorEvents
  const now = deps.now ?? Date.now
  async function durable(event: ConnectorEvent): Promise<void> {
    const current = store.connectorEvents.findIndex(item => item.id === event.id)
    if (current < 0) store.connectorEvents = [event, ...store.connectorEvents].slice(0, 500)
    else store.connectorEvents = store.connectorEvents.map(item => item.id === event.id ? event : item)
    // writeProtectedJson joins its per-file queue immediately but encrypts later; detach
    // now so a later UI edit cannot change the bytes that precede an external send.
    await persist(structuredClone(store.connectorEvents))
  }
  async function approve(event: ConnectorEvent, reviewDigest: unknown, actor: string): Promise<ConnectorEvent> {
    if (event.delivery) return event
    const outbound = outboundSnapshot(event)
    const digest = outboundReviewDigest(event, outbound)
    if (!event.outbound || event.reviewDigest !== digest || reviewDigest !== digest || JSON.stringify(event.outbound) !== JSON.stringify(outbound)) throw new Error('This message changed or was not fully reviewed; review its exact account, recipient and body again.')
    const transport = outbound.connectorId === 'email' ? 'email' : outbound.connectorId
    if (outbound.connectorId !== 'email' && !resolveAdapter(deps.adapters, parseThreadAddress(outbound.threadKey))) throw new Error('This message channel is unavailable; reconnect it before reviewing again.')
    const snapshot = Object.freeze(structuredClone(outbound))
    const approved: ConnectorEvent = { ...structuredClone(event), outbound: snapshot, requiresApproval: false, approvalStatus: 'approved', resolvedAt: now(), resolvedBy: actor,
      delivery: { transport, status: 'sending', attemptId: randomUUID(), reviewDigest: digest, attemptedAt: now(), deliveredAt: null, error: null } }
    try { await durable(approved) } catch (error) {
      // No adapter was called. Restore the pending object; saving the exact review may
      // be retried only after storage recovers.
      const idx = store.connectorEvents.findIndex(item => item.id === event.id)
      if (idx >= 0) store.connectorEvents[idx] = event
      throw error
    }
    const live = store.connectorEvents.find(item => item.id === event.id)
    let unchanged = false
    try { unchanged = live === approved && outboundReviewDigest(live) === digest } catch { /* revoked account/scope: nothing sent */ }
    if (!unchanged) {
      approved.delivery = { ...approved.delivery!, status: 'failed', error: 'The draft changed before dispatch. Nothing was sent.' }
      await durable(approved)
      throw new Error(approved.delivery.error!)
    }
    let result: NonNullable<ConnectorEvent['delivery']>
    try {
      if (snapshot.connectorId === 'email') {
        result = await deps.sendEmail(approved)
      } else {
        // No draft animation/fallback or automatic retry for approved actions. Partial
        // multi-message sends and timeouts are uncertain, never safe to replay.
        const sent = await sendViaChannel(deps.adapters, snapshot.threadKey, { text: snapshot.body, inReplyTo: snapshot.inReplyTo }, { preserveText: true })
        result = { transport, status: sent.ok ? 'delivered' : 'uncertain', sentId: sent.sentId ?? null, path: snapshot.threadKey, deliveredAt: sent.ok ? now() : null, error: sent.ok ? null : (sent.error ?? 'Delivery could not be confirmed; check the conversation before creating another message.') }
      }
    } catch {
      result = { transport, status: 'uncertain', deliveredAt: null, error: 'Delivery could not be confirmed; check the destination before creating another message.' }
    }
    const settled = { ...approved, delivery: { ...approved.delivery!, ...result } }
    // If this final save fails the durable sending marker remains. On restart it is
    // shown as uncertain and cannot be replayed by an approval click.
    await durable(settled)
    return settled
  }
  return { approve, durable }
}

/** Run once after loading persisted events, before admitting new connector work. */
export async function recoverInterruptedConnectorDeliveries(): Promise<void> {
  if (!store.connectorEvents.some(event => event.delivery?.status === 'sending')) return
  store.connectorEvents = store.connectorEvents.map(event => event.delivery?.status === 'sending'
    ? { ...event, delivery: { ...event.delivery, status: 'uncertain', error: 'The app stopped before delivery was confirmed. Check the destination before composing another message.' } }
    : event)
  await saveConnectorEvents(structuredClone(store.connectorEvents))
}
