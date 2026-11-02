import type { GmailInboxApi } from '../../shared/gmail-inbox'
export function gmailInboxPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): GmailInboxApi {
  return {
    gmailInboxSnapshot: () => invoke('gmailInbox:status'),
    gmailInboxEnable: input => invoke('gmailInbox:enable', input),
    gmailInboxSync: () => invoke('gmailInbox:sync'),
    gmailInboxThread: key => invoke('gmailInbox:thread', key),
    gmailInboxReply: (key, text) => invoke('gmailInbox:reply', { key, text }),
    gmailInboxDraft: key => invoke('gmailInbox:draft', key),
    gmailInboxSaveDraft: input => invoke('gmailInbox:saveDraft', input),
  }
}
