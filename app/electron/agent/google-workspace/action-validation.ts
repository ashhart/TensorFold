import type { GoogleWorkspaceAction, GoogleWorkspaceEmailContent, GoogleWorkspaceEventContent } from '../../../shared/google-workspace'
import { boundedString, emailList, eventWindow, googleResourceId, validTimeZone } from './validation'

export function validatedEmail(input: GoogleWorkspaceEmailContent): GoogleWorkspaceEmailContent {
  if (!input || typeof input !== 'object') throw new Error('Provide an email to review.')
  const to = emailList(input.to, true); const cc = emailList(input.cc); const bcc = emailList(input.bcc)
  if (to.length + cc.length + bcc.length > 50) throw new Error('An email can have at most 50 recipients.')
  const subject = boundedString(input.subject, 'email subject', 200, true)
  if (/[\r\n]/.test(subject)) throw new Error('Email subjects cannot contain line breaks.')
  const text = boundedString(input.text, 'email body', 128_000, true)
  if (!subject.trim() && !text.trim()) throw new Error('Write an email subject or message before reviewing it.')
  return { to, ...(cc.length ? { cc } : {}), ...(bcc.length ? { bcc } : {}), subject, text }
}
function sendUpdates(value: unknown): 'none' | 'all' {
  if (value !== 'none' && value !== 'all') throw new Error('Choose whether Google should notify attendees.')
  return value
}
export function validatedEvent(input: GoogleWorkspaceEventContent): GoogleWorkspaceEventContent {
  if (!input || typeof input !== 'object') throw new Error('Provide an event to review.')
  const title = boundedString(input.title, 'event title', 1024)
  if (/[\r\n]/.test(title)) throw new Error('Event titles cannot contain line breaks.')
  const { start, end } = eventWindow(input.start, input.end)
  const timeZone = validTimeZone(input.timeZone)
  const description = input.description === undefined ? undefined : boundedString(input.description, 'event description', 32_768, true)
  const attendees = emailList(input.attendees)
  return { title, ...(description === undefined ? {} : { description }), start, end, timeZone, ...(attendees.length ? { attendees } : {}), sendUpdates: sendUpdates(input.sendUpdates) }
}
function etag(value: unknown): string {
  const text = boundedString(value, 'event version', 512)
  if (!/^"[\x21\x23-\x7e]+"$/.test(text)) throw new Error('Read the current event before updating or deleting it.')
  return text
}
export function validatedGoogleAction(input: GoogleWorkspaceAction): GoogleWorkspaceAction {
  if (!input || typeof input !== 'object') throw new Error('Choose an operation to review.')
  switch (input.kind) {
    case 'gmail.send': return { kind: 'gmail.send', email: validatedEmail(input.email) }
    case 'calendar.create':
      googleResourceId(input.calendarId, 'calendar ID')
      return { kind: input.kind, calendarId: input.calendarId, event: validatedEvent(input.event) }
    case 'calendar.update':
      googleResourceId(input.calendarId, 'calendar ID'); googleResourceId(input.eventId, 'event ID')
      return { kind: input.kind, calendarId: input.calendarId, eventId: input.eventId, etag: etag(input.etag), event: validatedEvent(input.event) }
    case 'calendar.delete':
      googleResourceId(input.calendarId, 'calendar ID'); googleResourceId(input.eventId, 'event ID')
      return { kind: input.kind, calendarId: input.calendarId, eventId: input.eventId, etag: etag(input.etag), title: boundedString(input.title, 'event title', 1024), sendUpdates: sendUpdates(input.sendUpdates) }
    default: throw new Error('Google operation is not supported.')
  }
}
export function emailMime(email: GoogleWorkspaceEmailContent, from: string): string {
  const headers = [`From: ${from}`, `To: ${email.to.join(', ')}`, ...(email.cc?.length ? [`Cc: ${email.cc.join(', ')}`] : []), ...(email.bcc?.length ? [`Bcc: ${email.bcc.join(', ')}`] : [])]
  // Short encoded words fold long Unicode subjects without introducing untrusted headers.
  const words: string[] = []
  let chunk = ''
  for (const character of email.subject) {
    if (Buffer.byteLength(chunk + character, 'utf8') > 42) { words.push(`=?UTF-8?B?${Buffer.from(chunk).toString('base64')}?=`); chunk = '' }
    chunk += character
  }
  if (chunk) words.push(`=?UTF-8?B?${Buffer.from(chunk).toString('base64')}?=`)
  headers.push(`Subject: ${words.join('\r\n ')}`, 'MIME-Version: 1.0', 'Content-Type: text/plain; charset=UTF-8', 'Content-Transfer-Encoding: base64')
  const body = Buffer.from(email.text, 'utf8').toString('base64').match(/.{1,76}/g)?.join('\r\n') ?? ''
  return Buffer.from(`${headers.join('\r\n')}\r\n\r\n${body}\r\n`, 'utf8').toString('base64url')
}
export function googleEventBody(event: GoogleWorkspaceEventContent): Record<string, unknown> {
  const allDay = /^\d{4}-\d{2}-\d{2}$/.test(event.start)
  return { summary: event.title, description: event.description ?? '', start: allDay ? { date: event.start } : { dateTime: event.start, timeZone: event.timeZone }, end: allDay ? { date: event.end } : { dateTime: event.end, timeZone: event.timeZone }, attendees: (event.attendees ?? []).map(email => ({ email })) }
}
