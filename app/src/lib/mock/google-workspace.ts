import type { GoogleWorkspaceApi } from '../../../shared/google-workspace-api'
import type { GoogleWorkspaceAccount, GoogleWorkspaceActionReview, GoogleWorkspaceEmailContent, GoogleWorkspaceEvent, GoogleWorkspaceLocalDraft, GoogleWorkspaceMessage, GoogleWorkspaceReviewedAction, GoogleWorkspaceSnapshot } from '../../../shared/google-workspace'

/** Browser preview only; it contains synthetic data and never calls Google. */
export function createMockGoogleWorkspaceApi(uid: () => string): GoogleWorkspaceApi {
  const state: GoogleWorkspaceSnapshot = { clientConfigured: false, accounts: [], reviews: [], recentActions: [] }
  let drafts: GoogleWorkspaceLocalDraft[] = []
  const mail: GoogleWorkspaceMessage[] = [
    { id: 'demo-mail-1', threadId: 'demo-thread-1', subject: 'Project check-in', from: 'Casey <casey@example.com>', to: 'viewer@example.com', date: '2026-10-16T09:00:00Z', snippet: 'Can we review the latest draft tomorrow?', text: 'Can we review the latest draft tomorrow?\n\nThis is synthetic browser-preview content.', attachments: [], untrustedContent: true },
    { id: 'demo-mail-2', threadId: 'demo-thread-2', subject: 'Your weekly reading list', from: 'Reading <reading@example.com>', to: 'viewer@example.com', date: '2026-10-16T08:00:00Z', snippet: 'A few interesting articles for the week.', text: 'A few interesting articles for the week.\n\nRemote images are not loaded.', htmlOmitted: true, attachments: [{ filename: 'reading-list.pdf', mimeType: 'application/pdf', size: 4096 }], untrustedContent: true },
  ]
  const events: GoogleWorkspaceEvent[] = [{ id: 'demo-event-1', calendarId: 'primary', title: 'Project review', description: 'Synthetic browser-preview event.', start: '2026-10-17T09:00:00+01:00', end: '2026-10-17T09:30:00+01:00', timeZone: 'Europe/London', status: 'confirmed', etag: '"demo-v1"', attendees: ['casey@example.com'], untrustedContent: true }]
  const snapshot = () => structuredClone(state)
  const account = (id: string, capability?: GoogleWorkspaceAccount['capabilities'][number]) => {
    const found = state.accounts.find((item) => item.id === id && item.status === 'connected')
    if (!found) throw new Error('Connect this Google account first.')
    if (capability && !found.capabilities.includes(capability)) throw new Error('Reconnect with the required Google permission.')
    return found
  }
  const review = (receipt: GoogleWorkspaceReviewedAction) => {
    const found = state.reviews.find((item) => item.id === receipt.id && item.accountId === receipt.accountId && item.revision === receipt.revision && item.digest === receipt.digest)
    if (!found || found.expiresAt <= Date.now() || account(found.accountId).revision !== found.revision) throw new Error('Review changed or expired. Prepare it again.')
    return found
  }
  return {
    googleWorkspaceStatus: async () => snapshot(),
    googleWorkspaceImportClient: async () => { state.clientConfigured = true; return snapshot() },
    googleWorkspaceAuthorize: async (input) => {
      if (!state.clientConfigured) throw new Error('Import a Desktop OAuth client first.')
      const previous = input.accountId ? account(input.accountId) : undefined
      const next: GoogleWorkspaceAccount = { id: previous?.id ?? uid(), email: previous?.email ?? (state.accounts.length ? 'second@example.com' : 'viewer@example.com'), revision: (previous?.revision ?? 0) + 1, capabilities: [...input.capabilities], status: 'connected', connectedAt: Date.now() }
      state.accounts = [...state.accounts.filter((item) => item.id !== next.id), next]
      state.reviews = state.reviews.filter((item) => item.accountId !== next.id)
      state.authorization = { id: uid(), expiresAt: Date.now() + 180000, capabilities: [...input.capabilities], status: 'complete' }
      return structuredClone(state.authorization)
    },
    googleWorkspaceCancelAuth: async () => { if (state.authorization) state.authorization.status = 'cancelled'; return snapshot() },
    googleWorkspaceDisconnect: async (id) => { account(id); state.accounts = state.accounts.filter((item) => item.id !== id); state.reviews = state.reviews.filter((item) => item.accountId !== id); state.recentActions = state.recentActions.filter((item) => item.review.accountId !== id); drafts = drafts.filter((item) => item.accountId !== id); return snapshot() },
    googleWorkspaceMessages: async (id, query = {}) => {
      account(id, 'gmail.read')
      const matches = mail.filter((item) => !query.query || `${item.subject} ${item.text}`.toLowerCase().includes(query.query.toLowerCase()))
      return { items: structuredClone(matches.slice(0, query.limit ?? 20)) }
    },
    googleWorkspaceMessage: async (id, messageId) => { account(id, 'gmail.read'); const found = mail.find((item) => item.id === messageId); if (!found) throw new Error('Message unavailable.'); return structuredClone(found) },
    googleWorkspaceThread: async (id, threadId) => { account(id, 'gmail.read'); return { id: threadId, messages: structuredClone(mail.filter((item) => item.threadId === threadId)) } },
    googleWorkspaceCalendars: async (id) => { account(id, 'calendar.read'); return { items: [{ id: 'primary', summary: 'Personal calendar', timeZone: 'Europe/London', accessRole: 'owner' }] } },
    googleWorkspaceEvents: async (id, query) => { account(id, 'calendar.read'); return { items: structuredClone(events.filter((item) => item.calendarId === query.calendarId && Date.parse(item.start) >= Date.parse(query.timeMin) && Date.parse(item.start) < Date.parse(query.timeMax))) } },
    googleWorkspaceAvailability: async (id, query) => { account(id, 'calendar.read'); return { timeMin: query.timeMin, timeMax: query.timeMax, calendars: query.calendarIds.map((calendarId) => ({ id: calendarId, busy: events.filter((event) => event.calendarId === calendarId).map((event) => ({ start: event.start, end: event.end })), errors: [] })) } },
    googleWorkspaceSaveDraft: async (id, email: GoogleWorkspaceEmailContent, draftId) => {
      account(id)
      if (!email.to.length || !email.subject.trim() || !email.text.trim()) throw new Error('Add a recipient, subject and message.')
      if (draftId && !drafts.some((draft) => draft.id === draftId && draft.accountId === id)) throw new Error('Draft unavailable for this account.')
      const draft: GoogleWorkspaceLocalDraft = { ...structuredClone(email), id: draftId ?? uid(), accountId: id, updatedAt: Date.now(), localOnly: true }
      drafts = [draft, ...drafts.filter((item) => item.id !== draft.id)]; return structuredClone(draft)
    },
    googleWorkspaceDrafts: async (id) => { account(id); return structuredClone(drafts.filter((item) => item.accountId === id)) },
    googleWorkspaceDeleteDraft: async (id, draftId) => { account(id); drafts = drafts.filter((item) => item.id !== draftId || item.accountId !== id) },
    googleWorkspacePrepareAction: async (id, action) => {
      const owner = account(id, action.kind === 'gmail.send' ? 'gmail.send' : 'calendar.write')
      const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify({ id, revision: owner.revision, action })))
      const previousEvent = action.kind === 'calendar.update' || action.kind === 'calendar.delete' ? events.find((event) => event.id === action.eventId && event.calendarId === action.calendarId) : undefined
      const next: GoogleWorkspaceActionReview = { id: uid(), accountId: id, accountEmail: owner.email, revision: owner.revision, digest: Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, '0')).join(''), expiresAt: Date.now() + 300000, action: structuredClone(action), ...(previousEvent ? { previousEvent: structuredClone(previousEvent) } : {}) }
      state.reviews.unshift(next); return structuredClone(next)
    },
    googleWorkspaceCancelAction: async (receipt) => { const found = review(receipt); state.reviews = state.reviews.filter((item) => item.id !== found.id); const result = { id: found.id, status: 'failed' as const, cancelled: true as const, error: 'Cancelled by owner.' }; state.recentActions.unshift({ review: found, result }); return result },
    googleWorkspaceExecuteAction: async (receipt) => { const found = review(receipt); state.reviews = state.reviews.filter((item) => item.id !== found.id); const result = { id: found.id, status: 'succeeded' as const, remoteId: 'browser-preview-only' }; state.recentActions.unshift({ review: found, result }); return result },
  }
}
