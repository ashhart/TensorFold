import type { GoogleWorkspaceAvailability, GoogleWorkspaceCalendar, GoogleWorkspaceEvent, GoogleWorkspaceEventQuery, GoogleWorkspacePage } from '../../../shared/google-workspace'
import { GoogleAccounts } from './accounts'
import { googlePage, googleResourceId, responseArray, responseText, returnedPageToken, timeWindow, validTimeZone } from './validation'

const BASE = 'https://www.googleapis.com/calendar/v3'
export function parseGoogleEvent(data: any, calendarId: string): GoogleWorkspaceEvent {
  return { id: responseText(data?.id, 512), calendarId, title: responseText(data?.summary, 1024), description: responseText(data?.description, 32_768), start: responseText(data?.start?.dateTime ?? data?.start?.date, 64), end: responseText(data?.end?.dateTime ?? data?.end?.date, 64), timeZone: responseText(data?.start?.timeZone, 100), status: responseText(data?.status, 32), etag: responseText(data?.etag, 512), attendees: responseArray(data?.attendees, 50).map(item => responseText(item?.email, 320)).filter(Boolean), untrustedContent: true }
}
export class GoogleCalendar {
  constructor(private readonly accounts: GoogleAccounts) {}
  async listCalendars(accountId: string, options: { limit?: number; pageToken?: string } = {}): Promise<GoogleWorkspacePage<GoogleWorkspaceCalendar>> {
    const params = new URLSearchParams(googlePage(options.limit, options.pageToken))
    return this.accounts.withAccess(accountId, 'calendar.read', async (token, _revision, signal) => {
      const data = await this.accounts.transport.json(`${BASE}/users/me/calendarList?${params}`, token, undefined, 'GET', signal)
      return { items: responseArray(data.items, options.limit ?? 20).map(item => ({ id: responseText(item?.id, 512), summary: responseText(item?.summary, 1024), timeZone: responseText(item?.timeZone, 100), accessRole: responseText(item?.accessRole, 32) })), ...(returnedPageToken(data.nextPageToken) ? { nextPageToken: data.nextPageToken } : {}) }
    })
  }
  async listEvents(accountId: string, options: GoogleWorkspaceEventQuery): Promise<GoogleWorkspacePage<GoogleWorkspaceEvent>> {
    const calendar = googleResourceId(options.calendarId, 'calendar ID')
    const params = new URLSearchParams({ ...googlePage(options.limit, options.pageToken), ...timeWindow(options.timeMin, options.timeMax), singleEvents: 'true', orderBy: 'startTime' })
    return this.accounts.withAccess(accountId, 'calendar.read', async (token, _revision, signal) => {
      const data = await this.accounts.transport.json(`${BASE}/calendars/${calendar}/events?${params}`, token, undefined, 'GET', signal)
      return { items: responseArray(data.items, options.limit ?? 20).map(item => parseGoogleEvent(item, options.calendarId)), ...(returnedPageToken(data.nextPageToken) ? { nextPageToken: data.nextPageToken } : {}) }
    })
  }
  async freeBusy(accountId: string, options: { calendarIds: string[]; timeMin: string; timeMax: string; timeZone?: string }): Promise<GoogleWorkspaceAvailability> {
    if (!Array.isArray(options.calendarIds) || !options.calendarIds.length || options.calendarIds.length > 20) throw new Error('Choose between 1 and 20 calendars.')
    options.calendarIds.forEach(id => googleResourceId(id, 'calendar ID'))
    const range = timeWindow(options.timeMin, options.timeMax)
    const timeZone = options.timeZone ? validTimeZone(options.timeZone) : 'UTC'
    return this.accounts.withAccess(accountId, 'calendar.read', async (token, _revision, signal) => {
      const data = await this.accounts.transport.json(`${BASE}/freeBusy`, token, { ...range, timeZone, calendarExpansionMax: 20, groupExpansionMax: 1, items: options.calendarIds.map(id => ({ id })) }, 'POST', signal)
      return { ...range, calendars: options.calendarIds.map(id => {
        const item = Object.hasOwn(data?.calendars ?? {}, id) ? data.calendars[id] : undefined
        if (!item) return { id, busy: [], errors: ['Google did not return availability for this calendar.'] }
        if (Array.isArray(item.busy) && item.busy.length > 10_000) throw new Error('Calendar availability exceeds the supported size.')
        return { id, busy: responseArray(item.busy, 10_000).map(period => ({ start: responseText(period?.start, 64), end: responseText(period?.end, 64) })), errors: responseArray(item.errors, 20).map(error => responseText(error?.reason, 100) || 'Availability unavailable.') }
      }) }
    })
  }
}
