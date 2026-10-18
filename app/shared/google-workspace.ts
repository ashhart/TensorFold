/** Renderer-safe contracts only. OAuth credentials never cross the host boundary. */
export type GoogleWorkspaceCapability = 'gmail.read' | 'gmail.send' | 'calendar.read' | 'calendar.write'
export type GoogleWorkspaceAccountStatus = 'connected' | 'needs-reauthorization' | 'disconnected'
export interface GoogleWorkspaceAccount {
  id: string
  email: string
  revision: number
  capabilities: GoogleWorkspaceCapability[]
  status: GoogleWorkspaceAccountStatus
  connectedAt: number
  lastError?: string
}
export interface GoogleWorkspaceAuthorization {
  id: string
  expiresAt: number
  capabilities: GoogleWorkspaceCapability[]
  status: 'waiting' | 'complete' | 'cancelled' | 'failed'
  error?: string
}
export interface GoogleWorkspaceSnapshot {
  clientConfigured: boolean
  accounts: GoogleWorkspaceAccount[]
  authorization?: GoogleWorkspaceAuthorization
  reviews: GoogleWorkspaceActionReview[]
  recentActions: GoogleWorkspaceSettledAction[]
}
export interface GoogleWorkspaceAuthRequest {
  capabilities: GoogleWorkspaceCapability[]
  accountId?: string
}
export interface GoogleWorkspacePage<T> { items: T[]; nextPageToken?: string }
export interface GoogleWorkspaceMessage {
  id: string
  threadId: string
  subject: string
  from: string
  to: string
  date: string
  snippet: string
  text?: string
  htmlOmitted?: boolean
  attachments: { filename: string; mimeType: string; size: number }[]
  untrustedContent: true
}
export interface GoogleWorkspaceThread { id: string; messages: GoogleWorkspaceMessage[] }
export interface GoogleWorkspaceCalendar { id: string; summary: string; timeZone?: string; accessRole?: string }
export interface GoogleWorkspaceEvent {
  id: string
  calendarId: string
  title: string
  description?: string
  start: string
  end: string
  timeZone?: string
  status?: string
  etag?: string
  attendees: string[]
  untrustedContent: true
}
export interface GoogleWorkspaceAvailability {
  timeMin: string
  timeMax: string
  calendars: { id: string; busy: { start: string; end: string }[]; errors: string[] }[]
}
export interface GoogleWorkspaceEmailContent { to: string[]; cc?: string[]; bcc?: string[]; subject: string; text: string }
export interface GoogleWorkspaceLocalDraft extends GoogleWorkspaceEmailContent {
  id: string
  accountId: string
  updatedAt: number
  localOnly: true
}
export interface GoogleWorkspaceEventContent {
  title: string
  description?: string
  start: string
  end: string
  timeZone: string
  attendees?: string[]
  /** Explicit choice shown in the owner review; never inferred from model output. */
  sendUpdates: 'none' | 'all'
}
export type GoogleWorkspaceAction =
  | { kind: 'gmail.send'; email: GoogleWorkspaceEmailContent }
  | { kind: 'calendar.create'; calendarId: string; event: GoogleWorkspaceEventContent }
  | { kind: 'calendar.update'; calendarId: string; eventId: string; etag: string; event: GoogleWorkspaceEventContent }
  | { kind: 'calendar.delete'; calendarId: string; eventId: string; etag: string; title: string; sendUpdates: 'none' | 'all' }
export interface GoogleWorkspaceActionReview {
  id: string
  accountId: string
  accountEmail: string
  revision: number
  digest: string
  expiresAt: number
  action: GoogleWorkspaceAction
  previousEvent?: GoogleWorkspaceEvent
}
/** Only a host-verified owner review may supply this receipt. Never expose execution as an agent tool. */
export interface GoogleWorkspaceReviewedAction { id: string; accountId: string; revision: number; digest: string }
export interface GoogleWorkspaceActionResult {
  id: string
  status: 'succeeded' | 'uncertain' | 'failed'
  remoteId?: string
  error?: string
  cancelled?: true
}
export interface GoogleWorkspaceSettledAction {
  review: GoogleWorkspaceActionReview
  result: GoogleWorkspaceActionResult
}
export interface GoogleWorkspaceMessageQuery { query?: string; pageToken?: string; limit?: number }
export interface GoogleWorkspaceEventQuery {
  calendarId: string
  timeMin: string
  timeMax: string
  pageToken?: string
  limit?: number
}
