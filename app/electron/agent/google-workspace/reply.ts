import type { GoogleWorkspaceEmailContent } from '../../../shared/google-workspace'
import { GoogleAccounts } from './accounts'
import { parseGoogleMessage } from './gmail'
import type { GoogleWorkspacePrivateState } from './private-state'
import { googleResourceId, validEmail } from './validation'

export function assertInboxReplyAllowed(state: GoogleWorkspacePrivateState, accountId: string, revision: number, email: GoogleWorkspaceEmailContent): void {
  if (!email.reply) return
  const selected = state.inbox?.find(item => item.accountId === accountId && item.revision === revision && item.enabled)
  if (!selected || email.reply.scopeId && selected.scopeId !== email.reply.scopeId || !selected.messages.some(message => message.id === email.reply!.messageId && message.threadId === email.reply!.threadId)) throw new Error('This Gmail conversation is no longer selected for Inbox. Sync and review it again.')
}

export async function canonicalGoogleReply(accounts: GoogleAccounts, accountId: string, revision: number, email: GoogleWorkspaceEmailContent): Promise<GoogleWorkspaceEmailContent> {
  if (!email.reply) return email
  const messageId = googleResourceId(email.reply.messageId, 'reply message ID')
  googleResourceId(email.reply.threadId, 'reply thread ID')
  if (typeof email.text !== 'string' || email.text.length > 128_000 || !email.text.trim()) throw new Error('Write a bounded reply before reviewing it.')
  const scopeId = await accounts.state.read(state => { assertInboxReplyAllowed(state, accountId, revision, email); return state.inbox?.find(item => item.accountId === accountId)?.scopeId })
  const original = await accounts.withAccess(accountId, 'gmail.read', async (token, _revision, signal) =>
    parseGoogleMessage(await accounts.transport.json(`https://gmail.googleapis.com/gmail/v1/users/me/messages/${messageId}?format=metadata&metadataHeaders=Subject&metadataHeaders=From&metadataHeaders=Message-ID&metadataHeaders=References`, token, undefined, 'GET', signal)), revision)
  if (original.id !== email.reply.messageId || original.threadId !== email.reply.threadId) throw new Error('The Gmail reply target changed. Read this conversation again.')
  const recipient = (original.from.match(/<([^<>]+)>/)?.[1] ?? original.from).trim()
  validEmail(recipient)
  if (!original.messageId || !/^<[^\s<>]{1,500}>$/.test(original.messageId)) throw new Error('The original email has no safe reply header. Open Gmail to reply to it.')
  const references = [...new Set([...(original.references?.match(/<[^\s<>]{1,500}>/g) ?? []).slice(-19), original.messageId])].join(' ')
  return { to: [recipient], subject: original.subject, text: email.text, reply: { threadId: email.reply.threadId, messageId: email.reply.messageId, rfcMessageId: original.messageId, references, scopeId } }
}
