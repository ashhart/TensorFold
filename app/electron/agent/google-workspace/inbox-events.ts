import type { GoogleWorkspaceAccount, GoogleWorkspaceMessage } from '../../../shared/google-workspace'
import { gmailInboxKey } from '../../../shared/gmail-inbox'
import type { ConnectorEvent } from '../types'
import type { PrivateGoogleAction } from './private-state'

export function gmailMessageEvent(account: GoogleWorkspaceAccount, message: GoogleWorkspaceMessage): ConnectorEvent {
  const id = `gmail-event:${account.id}:${account.revision}:${message.id}`
  return {
    id, connectorId: 'email', pluginName: `Gmail · ${account.email}`, action: 'message_received', direction: 'inbound',
    classification: 'communication', priority: message.unread ? 'high' : 'normal', trustLevel: 'suggest', requiresApproval: false,
    approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null,
    title: message.subject || '(no subject)', summary: message.text ?? message.snippet, actor: message.from,
    threadKey: gmailInboxKey(account.id, account.revision, message.threadId), payloadKeys: ['gmailInbox', 'messageId', 'from', 'subject'],
    payload: { gmailInbox: true, messageId: id, from: message.from, subject: message.subject, ...(message.text ? { text: message.text } : {}) },
    sources: [{
      id, connectorId: 'email', sourceId: `gmail:${account.id}`, sourceRevision: String(account.revision), itemId: message.id,
      title: message.subject, path: `https://mail.google.com/mail/u/?authuser=${encodeURIComponent(account.email)}#all/${encodeURIComponent(message.id)}`, excerpt: message.snippet
    }],
    createdAt: message.internalDate ?? (Number.isFinite(Date.parse(message.date)) ? Date.parse(message.date) : account.connectedAt),
  }
}

export function gmailReviewEvent(account: GoogleWorkspaceAccount, item: PrivateGoogleAction): ConnectorEvent | null {
  const { review, result } = item
  if (review.action.kind !== 'gmail.send' || !review.action.email.reply) return null
  const email = review.action.email, threadKey = gmailInboxKey(account.id, account.revision, email.reply!.threadId)
  return {
    id: `gmail-review:${review.id}`, connectorId: 'email', pluginName: `Gmail · ${account.email}`, action: 'reply_requested', direction: 'outbound',
    classification: 'approval', priority: 'normal', trustLevel: 'approve', requiresApproval: !result,
    approvalId: review.id, approvalStatus: result ? result.cancelled ? 'denied' : 'approved' : 'pending',
    title: email.subject, summary: email.text, actor: account.email, threadKey, payloadKeys: ['gmailInbox', 'googleReview', 'to', 'subject', 'body'],
    payload: { gmailInbox: true, accountEmail: account.email, googleReview: { id: review.id, accountId: review.accountId, revision: review.revision, digest: review.digest }, to: email.to.join(', '), subject: email.subject, body: email.text },
    outbound: { version: 1, connectorId: 'email', accountId: account.id, accountRevision: String(account.revision), recipient: email.to.join(', '), body: email.text, subject: email.subject, threadKey },
    reviewDigest: review.digest,
    ...(result ? { delivery: { transport: 'gmail', status: result.status === 'succeeded' ? 'queued' : result.status === 'uncertain' ? 'uncertain' : 'failed', deliveredAt: null, error: result.error ?? null, sentId: result.remoteId ?? null } } : {}),
    createdAt: review.createdAt ?? account.connectedAt,
  }
}
