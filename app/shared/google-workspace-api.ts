import type { GoogleWorkspaceAction, GoogleWorkspaceActionResult, GoogleWorkspaceActionReview, GoogleWorkspaceAuthRequest, GoogleWorkspaceAuthorization, GoogleWorkspaceAvailability, GoogleWorkspaceCalendar, GoogleWorkspaceEmailContent, GoogleWorkspaceEvent, GoogleWorkspaceEventQuery, GoogleWorkspaceLocalDraft, GoogleWorkspaceMessage, GoogleWorkspaceMessageQuery, GoogleWorkspacePage, GoogleWorkspaceReviewedAction, GoogleWorkspaceSnapshot, GoogleWorkspaceThread } from './google-workspace'

export interface GoogleWorkspaceApi {
  googleWorkspaceStatus(): Promise<GoogleWorkspaceSnapshot>
  googleWorkspaceImportClient(): Promise<GoogleWorkspaceSnapshot | null>
  googleWorkspaceAuthorize(request: GoogleWorkspaceAuthRequest): Promise<GoogleWorkspaceAuthorization>
  googleWorkspaceCancelAuth(id?: string): Promise<GoogleWorkspaceSnapshot>
  googleWorkspaceDisconnect(accountId: string): Promise<GoogleWorkspaceSnapshot>
  googleWorkspaceMessages(accountId: string, query?: GoogleWorkspaceMessageQuery): Promise<GoogleWorkspacePage<GoogleWorkspaceMessage>>
  googleWorkspaceMessage(accountId: string, id: string): Promise<GoogleWorkspaceMessage>
  googleWorkspaceThread(accountId: string, id: string): Promise<GoogleWorkspaceThread>
  googleWorkspaceCalendars(accountId: string, query?: { limit?: number; pageToken?: string }): Promise<GoogleWorkspacePage<GoogleWorkspaceCalendar>>
  googleWorkspaceEvents(accountId: string, query: GoogleWorkspaceEventQuery): Promise<GoogleWorkspacePage<GoogleWorkspaceEvent>>
  googleWorkspaceAvailability(accountId: string, query: { calendarIds: string[]; timeMin: string; timeMax: string; timeZone?: string }): Promise<GoogleWorkspaceAvailability>
  googleWorkspaceSaveDraft(accountId: string, email: GoogleWorkspaceEmailContent, id?: string): Promise<GoogleWorkspaceLocalDraft>
  googleWorkspaceDrafts(accountId: string): Promise<GoogleWorkspaceLocalDraft[]>
  googleWorkspaceDeleteDraft(accountId: string, id: string): Promise<void>
  googleWorkspacePrepareAction(accountId: string, action: GoogleWorkspaceAction): Promise<GoogleWorkspaceActionReview>
  googleWorkspaceCancelAction(receipt: GoogleWorkspaceReviewedAction): Promise<GoogleWorkspaceActionResult>
  googleWorkspaceExecuteAction(receipt: GoogleWorkspaceReviewedAction): Promise<GoogleWorkspaceActionResult>
}

export const GOOGLE_WORKSPACE_API_SCHEMA = {
  googleWorkspaceStatus: 'method', googleWorkspaceImportClient: 'method',
  googleWorkspaceAuthorize: 'method', googleWorkspaceCancelAuth: 'method', googleWorkspaceDisconnect: 'method',
  googleWorkspaceMessages: 'method', googleWorkspaceMessage: 'method', googleWorkspaceThread: 'method',
  googleWorkspaceCalendars: 'method', googleWorkspaceEvents: 'method', googleWorkspaceAvailability: 'method',
  googleWorkspaceSaveDraft: 'method', googleWorkspaceDrafts: 'method', googleWorkspaceDeleteDraft: 'method',
  googleWorkspacePrepareAction: 'method', googleWorkspaceCancelAction: 'method', googleWorkspaceExecuteAction: 'method',
} as const
