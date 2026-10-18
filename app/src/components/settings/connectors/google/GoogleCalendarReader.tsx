import { OriginalDiagnostic } from '../../../OriginalDiagnostic'
import { useAuthoredText } from '../../../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { api } from '../../../../lib/ipc'
import { Button } from '../../../ui'
import type { GoogleWorkspaceAccount, GoogleWorkspaceAvailability, GoogleWorkspaceCalendar, GoogleWorkspaceEvent, GoogleWorkspacePage } from '../../../../../shared/google-workspace'
import { useGoogleAccountRequest } from './useGoogleAccountRequest'
import { calendarTimeLabel, macTimeZone, timedEventError, timestampInZone } from './google-calendar-time'
import { GoogleCalendarEditor } from './GoogleCalendarEditor'
import { GoogleDateTimeField } from './GoogleDateTimeField'

export function GoogleCalendarReader({ account, refresh }: { account: GoogleWorkspaceAccount; refresh: () => Promise<void> }) {
  const tx = useAuthoredText()
  const requests = useGoogleAccountRequest(account)
  const zone = macTimeZone()
  const [calendars, setCalendars] = useState<GoogleWorkspacePage<GoogleWorkspaceCalendar>>({ items: [] })
  const [calendarId, setCalendarId] = useState('')
  const [timeMin, setTimeMin] = useState(timestampInZone(new Date(), zone))
  const [timeMax, setTimeMax] = useState(timestampInZone(new Date(Date.now() + 7 * 86_400_000), zone))
  const [events, setEvents] = useState<GoogleWorkspacePage<GoogleWorkspaceEvent> | null>(null)
  const [availability, setAvailability] = useState<GoogleWorkspaceAvailability | null>(null)
  const [editing, setEditing] = useState<GoogleWorkspaceEvent | 'new' | null>(null)
  const loadCalendars = (pageToken?: string) => requests.run('calendars', () => api.googleWorkspaceCalendars(account.id, { limit: 20, pageToken }), (page) => { setCalendars((current) => ({ ...page, items: pageToken ? [...current.items, ...page.items.filter((item) => !current.items.some((existing) => existing.id === item.id))] : page.items })); if (!pageToken && page.items[0]) setCalendarId(page.items[0].id) })
  useEffect(() => { void loadCalendars() }, [account.id, account.revision])
  const calendar = calendars.items.find((item) => item.id === calendarId)
  const rangeError = timedEventError(timeMin, timeMax, zone)
  const loadEvents = (pageToken?: string) => requests.run('events', () => api.googleWorkspaceEvents(account.id, { calendarId, timeMin, timeMax, pageToken, limit: 20 }), setEvents)
  const changedRange = () => { setEvents(null); setAvailability(null); setEditing(null) }
  return <section className="space-y-3 rounded-xl border border-line bg-card p-4" aria-label={tx("Google Calendar")}>
    <div><h3 className="text-[14px] font-medium text-ink">{tx("Calendar · ")}{account.email}</h3><p className="mt-1 text-[12px] text-ink-3">{tx("Event dates and availability are shown in your Mac time zone, ")}{zone}{tx("; all-day events retain calendar dates and an exclusive end.")}</p></div>
    <label className="block text-[13px] text-ink-2">{tx("Calendar")}<select aria-label={tx("Google calendar")} value={calendarId} disabled={!!requests.busy || !calendars.items.length} onChange={(event) => { setCalendarId(event.target.value); changedRange() }} className="mt-1 w-full rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink"><option value="">{tx("Choose a calendar")}</option>{calendars.items.map((item) => <option key={item.id} value={item.id}>{item.summary} · {item.accessRole || tx("Access unknown")}</option>)}</select></label>
    <div className="flex flex-wrap gap-2"><Button size="sm" disabled={!!requests.busy} onClick={() => { changedRange(); void loadCalendars() }}>{requests.busy === 'calendars' ? tx("Loading calendars…") : tx("Refresh calendars")}</Button>{calendars.nextPageToken && <Button size="sm" disabled={!!requests.busy} onClick={() => loadCalendars(calendars.nextPageToken)}>{tx("More calendars")}</Button>}</div>
    {!requests.busy && !calendars.items.length && <p className="text-[13px] text-ink-3">{tx("No calendars were returned; check this account’s Calendar access and retry.")}</p>}
    <div className="grid gap-3 sm:grid-cols-2"><GoogleDateTimeField label={tx("Calendar range start")} value={timeMin} timeZone={zone} disabled={!!requests.busy} onChange={(value) => { setTimeMin(value); changedRange() }} /><GoogleDateTimeField label={tx("Calendar range end")} value={timeMax} timeZone={zone} disabled={!!requests.busy} onChange={(value) => { setTimeMax(value); changedRange() }} /></div>
    {rangeError && <p className="text-[12px] text-warn">{rangeError}</p>}
    <div className="flex flex-wrap gap-2"><Button size="sm" disabled={!!requests.busy || !calendarId || !!rangeError} onClick={() => loadEvents()}>{requests.busy === 'events' ? tx("Loading events…") : tx("View events")}</Button><Button size="sm" disabled={!!requests.busy || !calendarId || !!rangeError} onClick={() => requests.run('availability', () => api.googleWorkspaceAvailability(account.id, { calendarIds: [calendarId], timeMin, timeMax, timeZone: zone }), setAvailability)}>{requests.busy === 'availability' ? tx("Checking…") : tx("Check availability")}</Button><Button size="sm" variant="primary" disabled={!!requests.busy || !calendar || !account.capabilities.includes('calendar.write') || !['owner', 'writer'].includes(calendar.accessRole ?? '')} onClick={() => setEditing('new')}>{tx("New event")}</Button></div>
    {requests.error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{requests.error}</OriginalDiagnostic>}</p>}
    {availability && <div className="space-y-2 rounded-lg bg-card-2 p-3"><h4 className="text-[13px] font-medium text-ink">{tx("Availability for the selected range")}</h4>{availability.calendars.map((item) => <div key={item.id}>{item.errors.length ? <p role="alert" className="text-[12px] text-warn">{tx("Google could not return this calendar’s availability; it must not be treated as free.")}</p> : !item.busy.length ? <p className="text-[13px] text-ink-2">{tx("No busy periods were returned for this calendar and range.")}</p> : item.busy.map((period, index) => <p key={index} className="text-[12px] text-ink-2">{tx("Busy ")}{calendarTimeLabel(period.start, zone)} {tx("to ")}{calendarTimeLabel(period.end, zone)}</p>)}</div>)}</div>}
    {events && <div className="space-y-2">{events.items.map((event) => <article key={event.id} className="rounded-lg border border-line p-3"><p className="break-words text-[13px] font-medium text-ink">{event.title || tx("Untitled event")}{event.status ? ` · ${event.status}` : ''}</p><p className="mt-1 text-[12px] text-ink-3">{calendarTimeLabel(event.start, zone)} {tx("to ")}{calendarTimeLabel(event.end, zone)}{/^\d{4}-\d{2}-\d{2}$/.test(event.end) ? tx(" · End date excluded") : ''}</p>{event.description && <p className="mt-2 whitespace-pre-wrap break-words text-[13px] text-ink-2">{event.description}</p>}{!!event.attendees.length && <p className="mt-1 break-words text-[12px] text-ink-3">{tx("Guests: ")}{event.attendees.join(', ')}</p>}<Button size="sm" className="mt-2" disabled={!!requests.busy || event.status === 'cancelled' || !account.capabilities.includes('calendar.write')} onClick={() => setEditing(event)}>{tx("Edit or cancel")}</Button></article>)}{!events.items.length && <p className="text-[13px] text-ink-3">{tx("No events match this calendar and range.")}</p>}{events.nextPageToken && <Button size="sm" disabled={!!requests.busy} onClick={() => loadEvents(events.nextPageToken)}>{tx("Next events page")}</Button>}</div>}
    {calendar && editing && <GoogleCalendarEditor key={`${account.id}:${calendar.id}:${editing === 'new' ? 'new' : editing.id}`} account={account} calendar={calendar} event={editing === 'new' ? undefined : editing} refresh={refresh} close={() => setEditing(null)} />}
  </section>
}
