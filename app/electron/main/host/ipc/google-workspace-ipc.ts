import { dialog } from 'electron'
import { constants, promises as fs } from 'node:fs'
import type { GoogleWorkspaceService } from '../../../agent/google-workspace'
import type { GoogleWorkspaceAction, GoogleWorkspaceAuthRequest, GoogleWorkspaceEmailContent, GoogleWorkspaceEventQuery, GoogleWorkspaceMessageQuery, GoogleWorkspaceReviewedAction } from '../../../../shared/google-workspace'
import { googleWorkspaceService } from '../google-workspace-runtime'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { requireNativeConfirmation, type ConfirmSensitiveAction } from '../sensitive-confirmation'

interface GoogleIpcDeps {
  confirmSensitiveAction: ConfirmSensitiveAction
  service?: () => GoogleWorkspaceService
  handle?: typeof defaultHandle
}

export function registerGoogleWorkspaceIpc(deps: GoogleIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const service = deps.service ?? googleWorkspaceService
  // OAuth material and account data are managed by the owner on the host Mac.
  const owner = () => requireSecurityRole('owner', 'use Google Workspace connections')
  handle('googleWorkspace:status', async () => { owner(); return service().snapshot() })
  handle('googleWorkspace:importClient', async () => {
    owner()
    const selection = await dialog.showOpenDialog({ title: 'Import your Google Desktop OAuth client', properties: ['openFile'], filters: [{ name: 'Google OAuth JSON', extensions: ['json'] }] })
    if (selection.canceled || selection.filePaths.length !== 1) return null
    const file = await fs.open(selection.filePaths[0], constants.O_RDONLY | constants.O_NOFOLLOW)
    try {
      const stat = await file.stat()
      if (!stat.isFile() || stat.size > 64 * 1024 || stat.size === 0) throw new Error('Choose a Google Desktop OAuth client JSON smaller than 64 KB.')
      const json = await file.readFile('utf8')
      if (Buffer.byteLength(json, 'utf8') > 64 * 1024) throw new Error('The selected client file changed while reading it.')
      const result = await service().configureClient(json)
      await auditEvent('google.client_configured', { configured: true })
      return result
    } finally { await file.close() }
  })
  handle('googleWorkspace:authorize', async (_e, input: GoogleWorkspaceAuthRequest) => { owner(); return service().startAuthorization(input) })
  handle('googleWorkspace:cancelAuth', async (_e, id?: string) => { owner(); service().cancelAuthorization(id); return service().snapshot() })
  handle('googleWorkspace:disconnect', async (_e, accountId: string) => {
    owner()
    await requireNativeConfirmation(deps.confirmSensitiveAction, { title: 'Disconnect Google', message: 'Remove this Google account and revoke its project grant?', detail: 'This removes its local OAuth credentials, drafts and pending actions. Google revocation affects all services and clients using this Google project grant, not only Gmail or Calendar.', confirmLabel: 'Disconnect Google' })
    await service().disconnect(accountId)
    await auditEvent('google.account_disconnected', { disconnected: true })
    return service().snapshot()
  })
  handle('googleWorkspace:messages', async (_e, p: { accountId: string; query?: GoogleWorkspaceMessageQuery }) => { owner(); return service().listMessages(p.accountId, p.query) })
  handle('googleWorkspace:message', async (_e, p: { accountId: string; id: string }) => { owner(); return service().readMessage(p.accountId, p.id) })
  handle('googleWorkspace:thread', async (_e, p: { accountId: string; id: string }) => { owner(); return service().readThread(p.accountId, p.id) })
  handle('googleWorkspace:calendars', async (_e, p: { accountId: string; query?: { limit?: number; pageToken?: string } }) => { owner(); return service().listCalendars(p.accountId, p.query) })
  handle('googleWorkspace:events', async (_e, p: { accountId: string; query: GoogleWorkspaceEventQuery }) => { owner(); return service().listEvents(p.accountId, p.query) })
  handle('googleWorkspace:availability', async (_e, p: { accountId: string; query: { calendarIds: string[]; timeMin: string; timeMax: string; timeZone?: string } }) => { owner(); return service().freeBusy(p.accountId, p.query) })
  handle('googleWorkspace:saveDraft', async (_e, p: { accountId: string; email: GoogleWorkspaceEmailContent; id?: string }) => { owner(); return service().saveDraft(p.accountId, p.email, p.id) })
  handle('googleWorkspace:drafts', async (_e, id: string) => { owner(); return service().listDrafts(id) })
  handle('googleWorkspace:deleteDraft', async (_e, p: { accountId: string; id: string }) => { owner(); return service().deleteDraft(p.accountId, p.id) })
  handle('googleWorkspace:prepareAction', async (_e, p: { accountId: string; action: GoogleWorkspaceAction }) => { owner(); return service().prepareAction(p.accountId, p.action) })
  handle('googleWorkspace:cancelAction', async (_e, receipt: GoogleWorkspaceReviewedAction) => { owner(); return service().cancelAction(receipt) })
  handle('googleWorkspace:executeAction', async (_e, receipt: GoogleWorkspaceReviewedAction) => {
    owner()
    const review = await service().getReview(receipt)
    await requireNativeConfirmation(deps.confirmSensitiveAction, { title: 'Review Google action', message: `Approve ${review.action.kind} for ${review.accountEmail}?`, detail: JSON.stringify({ previousEvent: review.previousEvent, action: review.action }, null, 2), confirmLabel: review.action.kind === 'gmail.send' ? 'Send email' : 'Apply calendar change' })
    const result = await service().executeReviewed(receipt)
    await auditEvent('google.action_reviewed', { kind: review.action.kind, status: result.status })
    return result
  })
}
