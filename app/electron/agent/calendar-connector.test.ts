// Run: npx tsx electron/agent/calendar-connector.test.ts
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { calendarMutationEvent, calendarSourceId, calendarSourceRef, readCalendarEvent, scanCalendarEvents, summarizeCalendarDay } from './calendar-connector'
import type { CalendarConnectorSettings } from './types'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-calendar-'))
  const file = path.join(root, 'work.ics')
  const settings: CalendarConnectorSettings = {
    sources: [{ id: 'cal', label: 'Work Calendar', path: file, kind: 'ics_file', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
    maxEvents: 20,
    lastSyncAt: null,
  }

  await fs.writeFile(file, [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'BEGIN:VEVENT',
    'UID:review@example.test',
    'SUMMARY:Launch review',
    'DTSTART:20260620T090000Z',
    'DTEND:20260620T093000Z',
    'LOCATION:Project room',
    'DESCRIPTION:Review blockers\\nand launch risks.',
    'ORGANIZER:mailto:ops@example.test',
    'ATTENDEE:mailto:user@example.test',
    'STATUS:CONFIRMED',
    'END:VEVENT',
    'BEGIN:VEVENT',
    'UID:focus@example.test',
    'SUMMARY:Focus day',
    'DTSTART;VALUE=DATE:20260620',
    'DTEND;VALUE=DATE:20260621',
    'DESCRIPTION:Heads-down build time.',
    'END:VEVENT',
    'END:VCALENDAR',
  ].join('\n'), 'utf8')

  const { events, sources } = await scanCalendarEvents(settings, { day: '2026-06-20', limit: 10 })
  assert.equal(events.length, 2)
  assert.equal(sources[0].lastError, null)
  assert.ok(sources[0].lastSyncAt)

  const launch = events.find((event) => event.title === 'Launch review')
  assert.ok(launch)
  assert.equal(launch.location, 'Project room')
  assert.equal(launch.organizer, 'ops@example.test')
  assert.equal(launch.attendees[0], 'user@example.test')
  assert.match(launch.description, /Review blockers/)
  assert.equal(calendarSourceId(launch), `calendar:cal:${launch.id}`)
  assert.equal(calendarSourceRef(launch).connectorId, 'calendar')

  const allDay = events.find((event) => event.title === 'Focus day')
  assert.ok(allDay)
  assert.equal(allDay.allDay, true)

  const read = await readCalendarEvent(settings, calendarSourceId(launch))
  assert.equal(read.uid, 'review@example.test')
  assert.equal(read.id, launch.id)
  assert.equal(read.raw.SUMMARY, 'Launch review')

  const summary = await summarizeCalendarDay(settings, { day: '2026-06-20' })
  assert.match(summary.summary, /2 events on 2026-06-20/)
  assert.match(summary.summary, /Sources:/)
  assert.match(summary.summary, /calendar:cal:/)
  assert.equal(summary.sources.length, 2)

  const createEvent = calendarMutationEvent({ mode: 'create', sourceId: 'cal', title: 'New planning session', startAt: '2026-06-20T12:00:00Z', attendees: ['user@example.test'] }, 123)
  assert.equal(createEvent.action, 'calendar_create_requested')
  assert.equal(createEvent.requiresApproval, true)
  assert.equal(createEvent.approvalStatus, 'pending')
  assert.match(createEvent.summary, /not modified until/)

  const updateEvent = calendarMutationEvent({ mode: 'update', id: launch.id, title: 'Launch review moved', startAt: '2026-06-20T10:00:00Z' }, 124, launch)
  assert.equal(updateEvent.action, 'calendar_update_requested')
  assert.equal(updateEvent.requiresApproval, true)
  assert.equal(updateEvent.sources?.[0]?.id, calendarSourceId(launch))

  const filtered = await scanCalendarEvents(settings, { day: '2026-06-20', query: 'project', limit: 10 })
  assert.equal(filtered.events.length, 1)
  assert.equal(filtered.events[0].title, 'Launch review')

  await fs.rm(root, { recursive: true, force: true })
  console.log('calendar connector tests passed')
}

void main()
