import { api } from '../../lib/ipc'
import { parseGmailInboxKey } from '../../../shared/gmail-inbox'
import type { ComposerDraftScope } from '../../../shared/composer-drafts'
import type { ConnectorEvent } from '../../lib/types'
import { desktopDraftPort } from '../../lib/desktop-drafts'
import type { GoogleWorkspaceReviewedAction, GoogleWorkspaceThread } from '../../../shared/google-workspace'

export function gmailDraftBinding(key: string) {
  const target = parseGmailInboxKey(key)
  if (!target) return null
  const scope: ComposerDraftScope = { surface: 'assistant', conversationId: `gmail-draft:${target.accountId}:${target.threadId}`, hostId: null, incognito: false }
  return { scope, port: desktopDraftPort }
}

export function gmailReviewReceipt(event: ConnectorEvent): GoogleWorkspaceReviewedAction | null {
  if (!event.id.startsWith('gmail-review:') || event.payload?.gmailInbox !== true) return null
  const receipt = event.payload.googleReview as GoogleWorkspaceReviewedAction | undefined
  if (!receipt || typeof receipt.id !== 'string' || typeof receipt.accountId !== 'string' || !Number.isSafeInteger(receipt.revision) || typeof receipt.digest !== 'string' || receipt.digest !== event.reviewDigest) throw new Error('Refresh the Gmail draft before reviewing it.')
  return receipt
}

export async function resolveInboxApproval(event: ConnectorEvent, decision: 'approve' | 'deny', digest?: string): Promise<ConnectorEvent> {
  const receipt = gmailReviewReceipt(event)
  if (!receipt) return api.resolveConnectorApproval(event.id, decision, digest)
  if (decision === 'approve' && digest !== receipt.digest) throw new Error('Review the exact Gmail reply before sending.')
  const result = await (decision === 'approve' ? api.googleWorkspaceExecuteAction(receipt) : api.googleWorkspaceCancelAction(receipt))
  return {
    ...event, requiresApproval: false, approvalStatus: decision === 'approve' ? 'approved' : 'denied', delivery: {
      transport: 'gmail', status: result.status === 'succeeded' ? 'queued' : result.status === 'uncertain' ? 'uncertain' : 'failed', sentId: result.remoteId ?? null, deliveredAt: null, error: result.error ?? null,
    }
  }
}

export function gmailTaskContext(thread: GoogleWorkspaceThread): string {
  return `<untrusted_gmail_thread>\n${thread.messages.map(message => `Message ${message.id}\nFrom: ${message.from}\nSubject: ${message.subject}\n${message.text ?? message.snippet}`).join('\n\n').slice(0, 24_000)}\n</untrusted_gmail_thread>`
}
