import { OriginalDiagnostic } from '../../../OriginalDiagnostic'
import { useAuthoredText } from '../../../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { api } from '../../../../lib/ipc'
import { Button, Field, Input, Toggle } from '../../../ui'
import type { GoogleWorkspaceAccount, GoogleWorkspaceAction, GoogleWorkspaceCalendar, GoogleWorkspaceEvent } from '../../../../../shared/google-workspace'
import { useGoogleAccountRequest } from './useGoogleAccountRequest'
import { allDayEventError, calendarTimeLabel, macTimeZone, safeCalendarTimeZone, timedEventError, timestampInZone } from './google-calendar-time'
import { GoogleDateTimeField } from './GoogleDateTimeField'

export function GoogleCalendarEditor({ account, calendar, event, refresh, close }: { account: GoogleWorkspaceAccount; calendar: GoogleWorkspaceCalendar; event?: GoogleWorkspaceEvent; refresh: () => Promise<void>; close: () => void }) {
  const tx = useAuthoredText()
  const requests = useGoogleAccountRequest(account)
  const initialZone = safeCalendarTimeZone(event?.timeZone || calendar.timeZone)
  const [allDay, setAllDay] = useState(!!event && /^\d{4}-\d{2}-\d{2}$/.test(event.start))
  const [title, setTitle] = useState(event?.title ?? '')
  const [description, setDescription] = useState(event?.description ?? '')
  const [zone, setZone] = useState(initialZone)
  const [start, setStart] = useState(event?.start ?? timestampInZone(new Date(), initialZone))
  const [end, setEnd] = useState(event?.end ?? timestampInZone(new Date(Date.now() + 3_600_000), initialZone))
  const [attendees, setAttendees] = useState(event?.attendees.join(', ') ?? '')
  const [sendUpdates, setSendUpdates] = useState<'' | 'all' | 'none'>('')
  const [notice, setNotice] = useState('')
  const [prepared, setPrepared] = useState('')
  const guests = attendees.split(/[,;\n]/).map((item) => item.trim()).filter(Boolean)
  const timeError = allDay ? allDayEventError(start, end, zone) : timedEventError(start, end, zone)
  const formError = !title.trim() ? tx("Give this event a title.") : /[\r\n]/.test(title) ? tx("Event titles cannot contain line breaks.") : timeError || (guests.some((address) => !/^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(address)) ? tx("Use plain email addresses for guests.") : !sendUpdates ? tx("Choose the guest notification behavior explicitly.") : null)
  const writable = account.capabilities.includes('calendar.write') && ['owner', 'writer'].includes(calendar.accessRole ?? '')
  const content = { title, description, start, end, timeZone: zone, attendees: guests, sendUpdates: sendUpdates || 'none' as const }
  const action: GoogleWorkspaceAction = event ? { kind: 'calendar.update', calendarId: calendar.id, eventId: event.id, etag: event.etag ?? '', event: content } : { kind: 'calendar.create', calendarId: calendar.id, event: content }
  const fingerprint = JSON.stringify(action)
  const prepare = (next: GoogleWorkspaceAction) => requests.run('prepare', () => api.googleWorkspacePrepareAction(account.id, next), () => { setPrepared(JSON.stringify(next)); setNotice(tx("Prepared for owner review below; Google Calendar is unchanged.")); void refresh().catch(() => setNotice(tx("The review was prepared; refresh Google connections to load it before approving."))) })
  return <section className="space-y-3 rounded-xl border border-line-strong bg-card p-4" aria-label={event ? tx("Edit Google event") : tx("Create Google event")}>
    <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-[14px] font-medium text-ink">{event ? tx("Edit or cancel event") : tx("New event")} · {calendar.summary}</h3><Button size="sm" onClick={close} disabled={!!requests.busy}>{tx("Close editor")}</Button></div>
    {event && <p className="text-[12px] text-ink-3">{tx("Currently ")}{calendarTimeLabel(event.start)} {tx("to ")}{calendarTimeLabel(event.end)}{tx("; the host checks the event version again before approving changes.")}</p>}
    <Field label={tx("Title")}><Input aria-label={tx("Event title")} value={title} onChange={(change) => setTitle(change.target.value)} disabled={!!requests.busy} /></Field>
    <label className="block text-[13px] text-ink-2">{tx("Description")}<textarea aria-label={tx("Event description")} value={description} rows={3} onChange={(change) => setDescription(change.target.value)} disabled={!!requests.busy} className="mt-1 w-full resize-y rounded-lg border border-line bg-card-2 p-3 text-[13px] text-ink" /></label>
    <div className="flex items-center justify-between gap-2"><span className="text-[13px] text-ink-2">{tx("All-day event")}</span><Toggle label={tx("All-day event")} checked={allDay} disabled={!!requests.busy} onChange={(checked) => { setAllDay(checked); setStart(''); setEnd(''); setNotice(tx("Enter the new dates or times explicitly; no existing event has been converted.")) }} /></div>
    <div className="grid gap-3 sm:grid-cols-2">{allDay ? <><Field label={tx("Start date")}><Input aria-label={tx("Event start")} type="date" value={start} onChange={(change) => setStart(change.target.value)} disabled={!!requests.busy} /></Field><Field label={tx("End date, exclusive")}><Input aria-label={tx("Event end")} type="date" value={end} onChange={(change) => setEnd(change.target.value)} disabled={!!requests.busy} /></Field></> : <><GoogleDateTimeField label={tx("Event start")} value={start} onChange={setStart} timeZone={zone} disabled={!!requests.busy} /><GoogleDateTimeField label={tx("Event end")} value={end} onChange={setEnd} timeZone={zone} disabled={!!requests.busy} /></>}</div>
    <Field label={tx("Event time zone")}><Input aria-label={tx("Event time zone")} value={zone} onChange={(change) => setZone(change.target.value)} disabled={!!requests.busy} placeholder="Europe/London" /></Field>
    <p className="text-[12px] leading-relaxed text-ink-3">{tx("Mac time zone is ")}{macTimeZone()}; {allDay ? tx("all-day end is exclusive, so a one-day event ends on the following date.") : tx("include seconds and the UTC offset; the preview checks daylight-saving offsets and ambiguous times.")}</p>
    {!timeError && <p className="break-words rounded-lg bg-card-2 p-3 text-[12px] text-ink-2">{calendarTimeLabel(start, zone)} → {calendarTimeLabel(end, zone)}{allDay ? tx(" · End date excluded") : ''}</p>}
    <Field label={tx("Guest email addresses")} hint={tx("optional")}><Input aria-label={tx("Event guests")} value={attendees} onChange={(change) => setAttendees(change.target.value)} disabled={!!requests.busy} /></Field>
    <label className="block text-[13px] text-ink-2">{tx("Guest updates")}<select aria-label={tx("Event guest updates")} value={sendUpdates} onChange={(change) => setSendUpdates(change.target.value as typeof sendUpdates)} disabled={!!requests.busy} className="mt-1 block w-full rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink"><option value="">{tx("Choose notification behavior")}</option><option value="all">{tx("Ask Google to notify all guests")}</option><option value="none">{tx("Request no guest updates")}</option></select></label>
    <p className="text-[12px] leading-relaxed text-ink-3">{tx("Google may still send some notifications when no updates are requested; the exact guests and update choice appear in the review.")}</p>
    {formError && <p className="text-[12px] text-warn">{formError}</p>}{!writable && <p className="text-[12px] text-warn">{tx("This needs granted Calendar write access and a calendar where your account is an owner or writer.")}</p>}{event && !event.etag && <p className="text-[12px] text-warn">{tx("Reload this event to obtain its current version before changing it.")}</p>}
    <div className="flex flex-wrap gap-2"><Button size="sm" variant="primary" disabled={!!requests.busy || !!formError || !writable || !!event && !event.etag || prepared === fingerprint} onClick={() => prepare(action)}>{tx("Review ")}{event ? tx("event changes") : tx("new event")}</Button>{event && <Button size="sm" variant="danger" disabled={!!requests.busy || !writable || !event.etag || !sendUpdates || prepared === 'delete'} onClick={() => requests.run('prepare-delete', () => api.googleWorkspacePrepareAction(account.id, { kind: 'calendar.delete', calendarId: calendar.id, eventId: event.id, etag: event.etag!, title: event.title, sendUpdates: sendUpdates as 'all' | 'none' }), () => { setPrepared('delete'); setNotice(tx("Cancellation prepared for owner review; the event still exists.")); void refresh().catch(() => {}) })}>{tx("Review event cancellation")}</Button>}</div>
    {requests.error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{requests.error}</OriginalDiagnostic>}</p>}{notice && <p role="status" className="text-[13px] text-ink-2">{notice}</p>}
  </section>
}
