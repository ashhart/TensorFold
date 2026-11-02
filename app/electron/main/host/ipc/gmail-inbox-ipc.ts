import { googleWorkspaceService } from '../google-workspace-runtime'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'

export function registerGmailInboxIpc(deps: { handle?: typeof defaultHandle; service?: typeof googleWorkspaceService } = {}): void {
  const handle = deps.handle ?? defaultHandle, service = deps.service ?? googleWorkspaceService
  const owner = () => requireSecurityRole('owner', 'use Gmail in Inbox')
  handle('gmailInbox:status', async () => { owner(); return service().inboxSnapshot() })
  handle('gmailInbox:enable', async (_event, input: { accountId: string; revision: number; enabled: boolean }) => { owner(); return service().enableInbox(input) })
  handle('gmailInbox:sync', async () => { owner(); return service().syncInbox() })
  handle('gmailInbox:thread', async (_event, key: string) => { owner(); return service().inboxThread(key) })
  handle('gmailInbox:reply', async (_event, input: { key: string; text: string }) => { owner(); return service().inboxReply(input?.key, input?.text) })
  handle('gmailInbox:draft', async (_event, key: string) => { owner(); return service().inboxDraft(key) })
  handle('gmailInbox:saveDraft', async (_event, input: { key: string; expectedRevision: number; text: string }) => { owner(); return service().saveInboxDraft(input) })
}
