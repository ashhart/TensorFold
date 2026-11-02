import type { GmailInboxCache } from '../../../shared/gmail-inbox'
import type { GoogleWorkspaceMessage } from '../../../shared/google-workspace'

const id = (value: unknown) => typeof value === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(value)
const text = (value: unknown, limit: number) => typeof value === 'string' && value.length <= limit && !value.includes('\0')
function message(value: GoogleWorkspaceMessage): boolean {
  return !!value && id(value.id) && id(value.threadId) && text(value.subject, 2048) && text(value.from, 2048) && text(value.to, 2048)
    && text(value.date, 2048) && text(value.snippet, 2048) && value.text === undefined && value.untrustedContent === true
    && (value.unread === undefined || typeof value.unread === 'boolean')
    && (value.internalDate === undefined || Number.isSafeInteger(value.internalDate) && value.internalDate >= 0)
    && ['messageId', 'references', 'inReplyTo'].every(key => value[key as keyof GoogleWorkspaceMessage] === undefined || text(value[key as keyof GoogleWorkspaceMessage], 2048))
    && Array.isArray(value.attachments) && value.attachments.length <= 100
    && value.attachments.every(item => item && text(item.filename, 512) && text(item.mimeType, 200) && Number.isSafeInteger(item.size) && item.size >= 0)
}
export function validInboxCache(cache: GmailInboxCache[] | undefined, accountIds: Set<string>): boolean {
  if (cache === undefined) return true
  return Array.isArray(cache) && cache.length <= 12 && new Set(cache.map(item => item?.accountId)).size === cache.length
    && cache.every(item => item && accountIds.has(item.accountId) && Number.isSafeInteger(item.revision) && item.revision > 0
      && typeof item.enabled === 'boolean' && (item.scopeId === undefined || text(item.scopeId, 128))
      && (item.syncedAt === null || Number.isSafeInteger(item.syncedAt) && item.syncedAt >= 0)
      && (item.hasMore === undefined || typeof item.hasMore === 'boolean') && (item.error === undefined || text(item.error, 2048))
      && Array.isArray(item.messages) && item.messages.length <= 20 && new Set(item.messages.map(value => value?.id)).size === item.messages.length && item.messages.every(message))
}
