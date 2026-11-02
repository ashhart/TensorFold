import type { ConnectorEvent } from '../electron/agent/types'
import type { GoogleWorkspaceActionReview, GoogleWorkspaceThread } from './google-workspace'
import type { ComposerDraftScope, ComposerDraftSnapshot, ComposerDraftWriteResult } from './composer-drafts'

export interface GmailInboxCache {
  accountId: string
  revision: number
  enabled: boolean
  scopeId?: string
  syncedAt: number | null
  messages: import('./google-workspace').GoogleWorkspaceMessage[]
  hasMore?: boolean
  error?: string
}
export interface GmailInboxAccount {
  id: string
  email: string
  revision: number
  enabled: boolean
  connected: boolean
  scopeId?: string
  syncedAt: number | null
  error?: string
  hasMore: boolean
}
export interface GmailInboxSnapshot { accounts: GmailInboxAccount[]; events: ConnectorEvent[] }
export interface GmailInboxApi {
  gmailInboxSnapshot(): Promise<GmailInboxSnapshot>
  gmailInboxEnable(input: { accountId: string; revision: number; enabled: boolean }): Promise<GmailInboxSnapshot>
  gmailInboxSync(): Promise<GmailInboxSnapshot>
  gmailInboxThread(key: string): Promise<GoogleWorkspaceThread>
  gmailInboxReply(key: string, text: string): Promise<GoogleWorkspaceActionReview>
  gmailInboxDraft(key: string): Promise<ComposerDraftSnapshot>
  gmailInboxSaveDraft(input: { key: string; expectedRevision: number; text: string }): Promise<ComposerDraftWriteResult>
}
export const GMAIL_INBOX_API_SCHEMA = {
  gmailInboxSnapshot: 'method', gmailInboxEnable: 'method', gmailInboxSync: 'method', gmailInboxThread: 'method', gmailInboxReply: 'method', gmailInboxDraft: 'method', gmailInboxSaveDraft: 'method',
} as const

export function gmailInboxKey(accountId: string, revision: number, threadId: string): string {
  return `gmail:${encodeURIComponent(accountId)}:${revision}:${encodeURIComponent(threadId)}`
}
export function parseGmailInboxKey(value: unknown): { accountId: string; revision: number; threadId: string } | null {
  if (typeof value !== 'string' || !value.startsWith('gmail:')) return null
  const parts = value.split(':')
  if (parts.length !== 4 || !/^[1-9]\d{0,14}$/.test(parts[2])) throw new Error('Choose a current Gmail conversation.')
  let accountId: string, threadId: string
  try { accountId = decodeURIComponent(parts[1]); threadId = decodeURIComponent(parts[3]) } catch { throw new Error('Choose a current Gmail conversation.') }
  if (!/^[A-Za-z0-9_-]{1,128}$/.test(accountId) || !/^[A-Za-z0-9_-]{1,128}$/.test(threadId)) throw new Error('Choose a current Gmail conversation.')
  return { accountId, revision: Number(parts[2]), threadId }
}

export function gmailDraftKey(scope: ComposerDraftScope): string | null {
  if (scope.surface !== 'assistant' || !scope.conversationId?.startsWith('gmail-draft:')) return null
  const parts = scope.conversationId.split(':')
  if (scope.hostId !== null || scope.incognito || parts.length !== 3 || !/^[A-Za-z0-9_-]{1,128}$/.test(parts[1]) || !/^[A-Za-z0-9_-]{1,128}$/.test(parts[2])) throw new Error('Choose a local Gmail draft account and conversation.')
  return gmailInboxKey(parts[1], 1, parts[2])
}

export function assertGmailInboxRoute(sourceThreadKey: unknown, remote: boolean): void {
  if (parseGmailInboxKey(sourceThreadKey) && remote) throw new Error('Gmail Inbox tasks run on this computer. Choose the local execution target before starting this task.')
}
