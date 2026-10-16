// Pure core for two-way calendar autonomy: build RFC 5545 VEVENT/VCALENDAR
// payloads for create / reschedule (update) / cancel, and assemble per-event
// meeting-prep context. No side effects — every clock/uid source is injected so
// this is fully unit-testable. The connector (calendar-connector.ts) and the
// `calendar:create` / `calendar:update` IPC handlers write the returned .ics
// string to disk; this module only produces the text + the SEQUENCE bookkeeping.

import type { CalendarEvent, CalendarMutationInput } from './types'

export type IcsMethod = 'REQUEST' | 'CANCEL'
export type IcsAction = 'create' | 'reschedule' | 'cancel'

export interface IcsBuildInput {
  /** create | reschedule (mutate DTSTART/DTEND of an existing UID) | cancel */
  action: IcsAction
  title: string
  /** epoch ms; required unless allDay+start given, cancel keeps prior start */
  startAt?: number | null
  endAt?: number | null
  allDay?: boolean
  location?: string
  description?: string
  organizer?: string
  attendees?: string[]
  /** stable iCalendar UID. For reschedule/cancel pass the existing event's uid. */
  uid?: string | null
  /** iCalendar SEQUENCE of the last known revision; the builder increments it. */
  sequence?: number | null
}

export interface IcsBuildResult {
  /** full VCALENDAR document, CRLF-folded, ready to write to an .ics file. */
  ics: string
  uid: string
  sequence: number
  method: IcsMethod
  status: 'CONFIRMED' | 'CANCELLED'
}

export interface IcsDeps {
  now?: () => number
  /** returns a fresh globally-unique id (uid host part / new-event uid). */
  uuid?: () => string
}

const DEFAULT_DEPS: Required<IcsDeps> = {
  now: () => Date.now(),
  uuid: () => Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2),
}

/**
 * Build a VCALENDAR string for a create / reschedule / cancel request.
 * - create → METHOD:REQUEST, STATUS:CONFIRMED, SEQUENCE 0, fresh UID.
 * - reschedule → METHOD:REQUEST, STATUS:CONFIRMED, SEQUENCE +1, same UID.
 * - cancel → METHOD:CANCEL, STATUS:CANCELLED, SEQUENCE +1, same UID.
 */
export function buildEventIcs(input: IcsBuildInput, deps: IcsDeps = {}): IcsBuildResult {
  const d = { ...DEFAULT_DEPS, ...deps }
  const nowMs = d.now()
  const isCancel = input.action === 'cancel'
  const isNew = input.action === 'create'

  const uid = normalizeUid(input.uid) ?? `${d.uuid()}@vontra.calendar`
  const priorSeq = Number.isFinite(input.sequence as number) ? Math.max(0, Math.floor(input.sequence as number)) : 0
  const sequence = isNew ? 0 : priorSeq + 1

  const title = clean(input.title) || 'Untitled event'
  const allDay = input.allDay ?? false
  const method: IcsMethod = isCancel ? 'CANCEL' : 'REQUEST'
  const status: 'CONFIRMED' | 'CANCELLED' = isCancel ? 'CANCELLED' : 'CONFIRMED'

  const lines: string[] = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//TensorFold//Calendar Autonomy//EN',
    'CALSCALE:GREGORIAN',
    `METHOD:${method}`,
    'BEGIN:VEVENT',
    `UID:${escapeText(uid)}`,
    `SEQUENCE:${sequence}`,
    `DTSTAMP:${icsStamp(nowMs)}`,
    `SUMMARY:${escapeText(title)}`,
    `STATUS:${status}`,
  ]

  if (input.startAt != null) lines.push(dateLine('DTSTART', input.startAt, allDay))
  // For an all-day DTEND iCalendar uses the exclusive next day; callers pass the
  // event's own end and we serialize verbatim (start-only all-day events are valid).
  if (input.endAt != null) lines.push(dateLine('DTEND', input.endAt, allDay))

  const location = clean(input.location ?? '')
  if (location) lines.push(`LOCATION:${escapeText(location)}`)
  const description = (input.description ?? '').trim()
  if (description) lines.push(`DESCRIPTION:${escapeText(description)}`)

  const organizer = emailish(input.organizer ?? '')
  if (organizer) lines.push(`ORGANIZER:mailto:${organizer}`)
  for (const attendee of dedupe((input.attendees ?? []).map(emailish).filter(Boolean))) {
    const role = isCancel ? ';PARTSTAT=DECLINED' : ';RSVP=TRUE'
    lines.push(`ATTENDEE${role}:mailto:${attendee}`)
  }

  lines.push('END:VEVENT', 'END:VCALENDAR')

  return {
    ics: lines.map(foldLine).join('\r\n') + '\r\n',
    uid,
    sequence,
    method,
    status,
  }
}

/** Map a connector CalendarMutationInput into an IcsBuildInput. */
export function icsInputFromMutation(
  input: CalendarMutationInput,
  existing?: CalendarEvent | null,
  existingSequence?: number | null,
): IcsBuildInput {
  const action: IcsAction = input.mode === 'update' ? 'reschedule' : 'create'
  return {
    action,
    title: input.title || existing?.title || 'Untitled event',
    startAt: coerceMs(input.startAt) ?? existing?.startAt ?? null,
    endAt: coerceMs(input.endAt) ?? existing?.endAt ?? null,
    allDay: input.allDay ?? existing?.allDay ?? false,
    location: input.location ?? existing?.location ?? '',
    description: input.description ?? existing?.description ?? '',
    organizer: input.organizer ?? existing?.organizer ?? '',
    attendees: input.attendees ?? existing?.attendees ?? [],
    uid: (existing as { uid?: string } | null | undefined)?.uid ?? input.id ?? null,
    sequence: existingSequence ?? null,
  }
}

// ── Meeting prep ────────────────────────────────────────────────────────────

export interface PrepRelatedItem {
  /** connector source id, e.g. "email:acct:msg" or "document:src:doc" */
  id: string
  title: string
  snippet?: string
  /** epoch ms; used to order most-recent-first */
  timestamp?: number | null
  kind?: string
}

export interface MeetingPrepContext {
  eventId: string
  title: string
  when: string
  attendees: string[]
  location: string
  /** de-duplicated, most-recent-first related material for the briefing. */
  related: PrepRelatedItem[]
  /** rendered markdown brief. */
  brief: string
}

/**
 * Assemble a per-event prep briefing from the event and a pool of candidate
 * related items (emails, docs, prior events). Pure: pass the already-fetched
 * pool; ranking is deterministic (attendee/keyword overlap, then recency).
 */
export function assembleMeetingPrep(
  event: CalendarEvent,
  pool: PrepRelatedItem[],
  opts: { maxItems?: number; now?: () => number } = {},
): MeetingPrepContext {
  const maxItems = Math.max(1, Math.min(opts.maxItems ?? 5, 20))
  const attendees = dedupe((event.attendees ?? []).map(emailish).filter(Boolean))
  const keywords = tokenize(`${event.title} ${event.location} ${event.description}`)
  const handles = new Set([...attendees.map((a) => a.toLowerCase()), ...attendees.map((a) => a.split('@')[0].toLowerCase())])

  const seen = new Set<string>()
  const scored = pool
    .filter((item) => {
      if (!item || !item.id || seen.has(item.id)) return false
      seen.add(item.id)
      return true
    })
    .map((item) => ({ item, score: scoreRelated(item, handles, keywords) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || (b.item.timestamp ?? 0) - (a.item.timestamp ?? 0))
    .slice(0, maxItems)
    .map((entry) => entry.item)

  const when = formatWhen(event.startAt, event.endAt, event.allDay)
  const brief = renderBrief(event, when, attendees, scored)

  return {
    eventId: event.id,
    title: event.title || '(untitled)',
    when,
    attendees,
    location: event.location || '',
    related: scored,
    brief,
  }
}

function scoreRelated(item: PrepRelatedItem, handles: Set<string>, keywords: Set<string>): number {
  const hay = `${item.title} ${item.snippet ?? ''}`.toLowerCase()
  let score = 0
  for (const handle of handles) if (handle && hay.includes(handle)) score += 3
  for (const kw of keywords) if (hay.includes(kw)) score += 1
  return score
}

function renderBrief(event: CalendarEvent, when: string, attendees: string[], related: PrepRelatedItem[]): string {
  const lines: string[] = []
  lines.push(`# Prep: ${event.title || '(untitled)'}`)
  lines.push('')
  lines.push(`When: ${when}`)
  if (event.location) lines.push(`Where: ${event.location}`)
  lines.push(`Attendees: ${attendees.length ? attendees.join(', ') : '—'}`)
  if (event.description?.trim()) {
    lines.push('')
    lines.push('## Agenda')
    lines.push(event.description.trim())
  }
  lines.push('')
  lines.push('## Related material')
  if (!related.length) {
    lines.push('No related emails, documents, or prior meetings found.')
  } else {
    for (const item of related) {
      const meta = item.timestamp ? ` (${new Date(item.timestamp).toISOString().slice(0, 10)})` : ''
      const snip = item.snippet ? ` — ${snippet(item.snippet, 140)}` : ''
      lines.push(`- [${item.id}] ${item.title || '(untitled)'}${meta}${snip}`)
    }
  }
  return lines.join('\n')
}

// ── helpers ─────────────────────────────────────────────────────────────────

function dateLine(name: 'DTSTART' | 'DTEND', ms: number, allDay: boolean): string {
  if (allDay) return `${name};VALUE=DATE:${icsDate(ms)}`
  return `${name}:${icsStamp(ms)}`
}

function icsStamp(ms: number): string {
  const d = new Date(ms)
  return (
    `${d.getUTCFullYear()}${p2(d.getUTCMonth() + 1)}${p2(d.getUTCDate())}` +
    `T${p2(d.getUTCHours())}${p2(d.getUTCMinutes())}${p2(d.getUTCSeconds())}Z`
  )
}

function icsDate(ms: number): string {
  const d = new Date(ms)
  return `${d.getUTCFullYear()}${p2(d.getUTCMonth() + 1)}${p2(d.getUTCDate())}`
}

function p2(n: number): string {
  return String(n).padStart(2, '0')
}

function escapeText(value: string): string {
  return String(value ?? '')
    .replace(/\\/g, '\\\\')
    .replace(/\n/g, '\\n')
    .replace(/,/g, '\\,')
    .replace(/;/g, '\\;')
}

/** RFC 5545 §3.1: fold content lines longer than 75 octets. */
function foldLine(line: string): string {
  if (line.length <= 75) return line
  const out: string[] = []
  let rest = line
  out.push(rest.slice(0, 75))
  rest = rest.slice(75)
  while (rest.length > 74) {
    out.push(' ' + rest.slice(0, 74))
    rest = rest.slice(74)
  }
  if (rest.length) out.push(' ' + rest)
  return out.join('\r\n')
}

function normalizeUid(uid?: string | null): string | null {
  const v = String(uid ?? '').trim()
  return v ? v : null
}

function coerceMs(value: number | string | null | undefined): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function formatWhen(startAt: number | null, endAt: number | null, allDay: boolean): string {
  if (startAt == null) return 'unscheduled'
  if (allDay) return new Date(startAt).toISOString().slice(0, 10)
  const start = new Date(startAt).toISOString().slice(0, 16).replace('T', ' ') + ' UTC'
  if (endAt == null) return start
  const end = new Date(endAt).toISOString().slice(11, 16)
  return `${start} – ${end}`
}

function tokenize(text: string): Set<string> {
  const stop = new Set(['the', 'and', 'for', 'with', 'meeting', 'call', 'sync', 'about', 'from', 'this', 'that'])
  return new Set(
    String(text ?? '')
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => w.length >= 4 && !stop.has(w)),
  )
}

function emailish(value: string): string {
  return String(value ?? '').replace(/^mailto:/i, '').trim()
}

function clean(value: string): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

function dedupe(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const v of values) {
    const key = v.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(v)
  }
  return out
}

function snippet(value: string, max: number): string {
  const c = clean(value)
  return c.length > max ? `${c.slice(0, max - 3)}...` : c
}
