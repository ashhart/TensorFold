// Run: jiti electron/agent/calendar-ics.test.ts
// Pure core for two-way calendar: VEVENT build (create/reschedule/cancel) + meeting prep.
import assert from 'node:assert/strict'
import { buildEventIcs, icsInputFromMutation, assembleMeetingPrep } from './calendar-ics.ts'
import type { CalendarEvent } from '../../electron/agent/types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const fixedUuid = () => 'abc123'
const fixedNow = () => Date.UTC(2026, 5, 30, 9, 0, 0) // 2026-06-30T09:00:00Z

console.log('calendar-ics tests:')

// ── create ──────────────────────────────────────────────────────────────────
{
  const r = buildEventIcs(
    {
      action: 'create',
      title: 'Design review',
      startAt: Date.UTC(2026, 6, 1, 14, 0, 0),
      endAt: Date.UTC(2026, 6, 1, 15, 0, 0),
      location: 'Room 4; floor 2',
      description: 'Discuss the plan,\nbring notes',
      organizer: 'mailto:me@example.com',
      attendees: ['a@example.com', 'MAILTO:b@example.com', 'a@example.com'],
    },
    { now: fixedNow, uuid: fixedUuid },
  )
  assert.equal(r.method, 'REQUEST')
  assert.equal(r.status, 'CONFIRMED')
  assert.equal(r.sequence, 0)
  assert.equal(r.uid, 'abc123@vontra.calendar')
  assert.ok(r.ics.startsWith('BEGIN:VCALENDAR\r\n'))
  assert.ok(r.ics.includes('METHOD:REQUEST\r\n'))
  assert.ok(r.ics.includes('BEGIN:VEVENT\r\n'))
  assert.ok(r.ics.includes('UID:abc123@vontra.calendar\r\n'))
  assert.ok(r.ics.includes('DTSTART:20260701T140000Z\r\n'))
  assert.ok(r.ics.includes('DTEND:20260701T150000Z\r\n'))
  assert.ok(r.ics.includes('DTSTAMP:20260630T090000Z\r\n'))
  assert.ok(r.ics.includes('SUMMARY:Design review\r\n'))
  assert.ok(r.ics.includes('SEQUENCE:0\r\n'))
  assert.ok(r.ics.endsWith('END:VCALENDAR\r\n'))
  ok('create → REQUEST/CONFIRMED, seq 0, fresh UID, UTC stamps')

  // Special chars escaped per RFC 5545 (comma, semicolon, newline).
  assert.ok(r.ics.includes('LOCATION:Room 4\\; floor 2\r\n'))
  assert.ok(r.ics.includes('DESCRIPTION:Discuss the plan\\,\\nbring notes\r\n'))
  ok('text escaping: comma / semicolon / newline')

  // organizer + de-duplicated, mailto-normalized attendees.
  assert.ok(r.ics.includes('ORGANIZER:mailto:me@example.com\r\n'))
  const att = r.ics.split('\r\n').filter((l) => l.startsWith('ATTENDEE'))
  assert.equal(att.length, 2)
  assert.ok(att[0].includes('RSVP=TRUE') && att[0].endsWith('mailto:a@example.com'))
  assert.ok(att[1].endsWith('mailto:b@example.com'))
  ok('attendees deduped, mailto-normalized, RSVP=TRUE')
}

// ── all-day ───────────────────────────────────────────────────────────────────
{
  const r = buildEventIcs(
    { action: 'create', title: 'Holiday', allDay: true, startAt: Date.UTC(2026, 6, 4, 0, 0, 0) },
    { now: fixedNow, uuid: fixedUuid },
  )
  assert.ok(r.ics.includes('DTSTART;VALUE=DATE:20260704\r\n'))
  assert.ok(!r.ics.includes('DTEND'))
  ok('all-day → DTSTART;VALUE=DATE, no DTEND when endAt absent')
}

// ── reschedule ────────────────────────────────────────────────────────────────
{
  const r = buildEventIcs(
    {
      action: 'reschedule',
      title: 'Standup',
      startAt: Date.UTC(2026, 6, 2, 10, 0, 0),
      endAt: Date.UTC(2026, 6, 2, 10, 30, 0),
      uid: 'existing-uid@example.com',
      sequence: 3,
    },
    { now: fixedNow, uuid: fixedUuid },
  )
  assert.equal(r.method, 'REQUEST')
  assert.equal(r.status, 'CONFIRMED')
  assert.equal(r.uid, 'existing-uid@example.com')
  assert.equal(r.sequence, 4) // prior 3 + 1
  assert.ok(r.ics.includes('DTSTART:20260702T100000Z\r\n'))
  ok('reschedule → same UID, SEQUENCE incremented, new DTSTART')
}

// ── cancel ────────────────────────────────────────────────────────────────────
{
  const r = buildEventIcs(
    { action: 'cancel', title: 'Standup', uid: 'existing-uid@example.com', sequence: 4, attendees: ['a@example.com'] },
    { now: fixedNow, uuid: fixedUuid },
  )
  assert.equal(r.method, 'CANCEL')
  assert.equal(r.status, 'CANCELLED')
  assert.equal(r.uid, 'existing-uid@example.com')
  assert.equal(r.sequence, 5)
  assert.ok(r.ics.includes('METHOD:CANCEL\r\n'))
  assert.ok(r.ics.includes('STATUS:CANCELLED\r\n'))
  assert.ok(r.ics.split('\r\n').some((l) => l.startsWith('ATTENDEE') && l.includes('PARTSTAT=DECLINED')))
  ok('cancel → CANCEL/CANCELLED, seq incremented, attendee DECLINED')
}

// ── line folding ──────────────────────────────────────────────────────────────
{
  const longTitle = 'X'.repeat(200)
  const r = buildEventIcs({ action: 'create', title: longTitle }, { now: fixedNow, uuid: fixedUuid })
  const summaryLines = r.ics.split('\r\n')
  // No physical line exceeds 75 octets.
  for (const l of summaryLines) assert.ok(l.length <= 75, `line too long: ${l.length}`)
  // Continuation lines begin with a single space.
  const idx = summaryLines.findIndex((l) => l.startsWith('SUMMARY:'))
  assert.ok(summaryLines[idx + 1].startsWith(' '))
  ok('long lines folded at ≤75 octets with space continuation')
}

// ── icsInputFromMutation ──────────────────────────────────────────────────────
{
  const existing: CalendarEvent & { uid: string } = {
    id: 'ev1', uid: 'orig@example.com', sourceId: 's', sourceLabel: 'L', path: '/c.ics',
    title: 'Old title', startAt: Date.UTC(2026, 6, 1, 9, 0, 0), endAt: Date.UTC(2026, 6, 1, 10, 0, 0),
    allDay: false, location: 'HQ', description: 'agenda', organizer: 'o@example.com', attendees: ['p@example.com'], status: 'CONFIRMED',
  }
  const built = icsInputFromMutation(
    { mode: 'update', id: 'ev1', title: 'New title', startAt: '2026-07-05T11:00:00Z' },
    existing,
    2,
  )
  assert.equal(built.action, 'reschedule')
  assert.equal(built.uid, 'orig@example.com') // pulled from existing.uid, not id
  assert.equal(built.sequence, 2)
  assert.equal(built.title, 'New title')
  assert.equal(built.startAt, Date.parse('2026-07-05T11:00:00Z'))
  assert.equal(built.endAt, existing.endAt) // preserved from existing
  assert.equal(built.location, 'HQ') // inherited
  ok('icsInputFromMutation merges input over existing, keeps uid + seq')

  // create mode with string start.
  const created = icsInputFromMutation({ mode: 'create', title: 'Kickoff', startAt: '2026-08-01T00:00:00Z' })
  assert.equal(created.action, 'create')
  assert.equal(created.uid, null)
  ok('icsInputFromMutation create → no uid, action create')
}

// ── assembleMeetingPrep ───────────────────────────────────────────────────────
{
  const event: CalendarEvent = {
    id: 'ev9', sourceId: 's', sourceLabel: 'L', path: '/c.ics',
    title: 'Quarterly roadmap review', startAt: Date.UTC(2026, 6, 10, 13, 0, 0), endAt: Date.UTC(2026, 6, 10, 14, 0, 0),
    allDay: false, location: 'Boardroom', description: 'Align on roadmap priorities', organizer: 'lead@example.com',
    attendees: ['alice@example.com', 'bob@example.com'], status: 'CONFIRMED',
  }
  const pool = [
    { id: 'email:1', title: 'Re: roadmap priorities from alice', snippet: 'here are my thoughts', timestamp: Date.UTC(2026, 6, 8) },
    { id: 'doc:1', title: 'Roadmap draft', snippet: 'q3 priorities', timestamp: Date.UTC(2026, 6, 1) },
    { id: 'email:2', title: 'Lunch plans', snippet: 'want to grab food', timestamp: Date.UTC(2026, 6, 9) },
    { id: 'email:1', title: 'dup should be ignored', timestamp: Date.UTC(2026, 6, 8) },
  ]
  const prep = assembleMeetingPrep(event, pool, { maxItems: 5 })
  assert.equal(prep.eventId, 'ev9')
  assert.deepEqual(prep.attendees, ['alice@example.com', 'bob@example.com'])
  // "Lunch plans" shares no attendee/keyword → excluded; dup id excluded.
  const ids = prep.related.map((r) => r.id)
  assert.ok(ids.includes('email:1') && ids.includes('doc:1'))
  assert.ok(!ids.includes('email:2'))
  assert.equal(new Set(ids).size, ids.length) // no dups
  // attendee-mention (alice) ranks above pure keyword match.
  assert.equal(prep.related[0].id, 'email:1')
  ok('meeting prep: ranks by attendee+keyword, dedupes, drops irrelevant')

  assert.ok(prep.brief.includes('# Prep: Quarterly roadmap review'))
  assert.ok(prep.brief.includes('Where: Boardroom'))
  assert.ok(prep.brief.includes('Attendees: alice@example.com, bob@example.com'))
  assert.ok(prep.brief.includes('## Agenda'))
  assert.ok(prep.brief.includes('[email:1]'))
  ok('meeting prep brief renders markdown with agenda + related refs')

  const empty = assembleMeetingPrep(event, [], {})
  assert.ok(empty.brief.includes('No related emails, documents, or prior meetings found.'))
  assert.equal(empty.related.length, 0)
  ok('meeting prep with empty pool renders graceful fallback')
}

console.log(`\n${passed} assertions passed.`)
