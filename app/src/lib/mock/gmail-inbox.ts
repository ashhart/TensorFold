import type { GmailInboxApi } from '../../../shared/gmail-inbox'
export function createMockGmailInboxApi(): GmailInboxApi {
  const desktop = async (): Promise<never> => { throw new Error('Gmail Inbox requires a connected desktop account.') }
  return { gmailInboxSnapshot: async () => ({ accounts: [], events: [] }), gmailInboxSync: async () => ({ accounts: [], events: [] }), gmailInboxEnable: desktop, gmailInboxThread: desktop, gmailInboxReply: desktop, gmailInboxDraft: desktop, gmailInboxSaveDraft: desktop }
}
