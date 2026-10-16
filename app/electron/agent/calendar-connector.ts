import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type {
  CalendarMutationInput,
  CalendarConnectorSettings,
  CalendarEvent,
  CalendarListOptions,
  CalendarReadResult,
  CalendarSourceConfig,
  CalendarSummaryResult,
  ConnectorEvent,
  ConnectorSourceRef,
} from './types'
import { ensureSummaryCitations } from './summary-citations'

const MAX_SCAN_FILES = 1_000
const MAX_FILE_BYTES = 4_000_000

interface ParsedCalendarProperty {
  name: string
  params: string
  value: string
}

export function makeCalendarSource(target: string, now = Date.now()): CalendarSourceConfig {
  const ext = path.extname(target).toLowerCase()
  return {
    id: `cal_${randomUUID().slice(0, 8)}`,
    label: path.basename(target) || 'Calendar',
    path: target,
    kind: ext === '.ics' ? 'ics_file' : 'ics_dir',
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
  }
}

export async function scanCalendarEvents(settings: CalendarConnectorSettings, opts: CalendarListOptions = {}): Promise<{ events: CalendarEvent[]; sources: CalendarSourceConfig[] }> {
  const now = Date.now()
  const sources = settings.sources.map((source) => ({ ...source }))
  const maxEvents = Math.max(25, Math.min(settings.maxEvents || 500, 5_000))
  const events: CalendarReadResult[] = []

  for (const source of sources) {
    if (!source.enabled) continue
    try {
      const files = await collectCalendarFiles(source.path, source.kind)
      for (const file of files.slice(0, MAX_SCAN_FILES)) events.push(...await readCalendarFile(source, file))
      source.lastSyncAt = now
      source.lastError = null
      source.updatedAt = now
    } catch (e: any) {
      source.lastError = String(e?.message ?? e)
      source.updatedAt = now
    }
  }

  const window = calendarWindow(opts)
  let filtered: CalendarEvent[] = events.sort((a, b) => (a.startAt ?? Number.MAX_SAFE_INTEGER) - (b.startAt ?? Number.MAX_SAFE_INTEGER))
  if (window.from != null) filtered = filtered.filter((event) => (event.endAt ?? event.startAt ?? 0) >= window.from!)
  if (window.to != null) filtered = filtered.filter((event) => (event.startAt ?? 0) < window.to!)
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    filtered = filtered.filter((event) => [event.title, event.location, event.description, event.organizer, event.attendees.join(' ')].some((value) => value.toLowerCase().includes(q)))
  }
  const limit = Math.max(1, Math.min(Number(opts.limit) || maxEvents, maxEvents))
  return { events: filtered.slice(0, limit), sources }
}

export async function readCalendarEvent(settings: CalendarConnectorSettings, id: string): Promise<CalendarReadResult> {
  const target = normalizeCalendarId(id)
  for (const source of settings.sources.filter((s) => s.enabled)) {
    const files = await collectCalendarFiles(source.path, source.kind)
    for (const file of files.slice(0, MAX_SCAN_FILES)) {
      const events = await readCalendarFile(source, file)
      const found = events.find((event) => event.id === target)
      if (found) return found
    }
  }
  throw new Error('Calendar event not found. Check the source and try again.')
}

export async function summarizeCalendarDay(settings: CalendarConnectorSettings, opts: CalendarListOptions = {}): Promise<CalendarSummaryResult> {
  const window = calendarWindow({ ...opts, day: opts.day ?? dayLabel(opts.from ?? Date.now()) })
  const { events } = await scanCalendarEvents(settings, { ...opts, from: window.from ?? undefined, to: window.to ?? undefined, limit: opts.limit ?? 40 })
  if (!events.length) return ensureSummaryCitations({ summary: `No calendar events found for ${dayLabel(window.from ?? Date.now())}.\n\nSources: none.`, sources: [] })
  const sources = events.map(calendarSourceRef)
  const lines = events.map((event) => `- [${calendarSourceId(event)}] ${timeRange(event)} ${event.title || '(untitled)'}${event.location ? ` @ ${event.location}` : ''}`)
  return ensureSummaryCitations({
    summary: `${events.length} event${events.length === 1 ? '' : 's'} on ${dayLabel(window.from ?? Date.now())}.\n\n${lines.join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  })
}

export function calendarSourceId(event: Pick<CalendarEvent, 'sourceId' | 'id'>): string {
  return `calendar:${event.sourceId}:${event.id}`
}

export function calendarSourceRef(event: CalendarEvent): ConnectorSourceRef {
  return {
    id: calendarSourceId(event),
    connectorId: 'calendar',
    sourceId: event.sourceId,
    itemId: event.id,
    title: event.title || '(untitled)',
    path: event.path,
    excerpt: [event.location, event.description].filter(Boolean).join(' | ').slice(0, 240),
    createdAt: event.startAt,
  }
}

export function calendarMutationEvent(input: CalendarMutationInput, now = Date.now(), existing?: CalendarEvent | null): ConnectorEvent {
  const mode = input.mode === 'update' ? 'update' : 'create'
  const title = clean(input.title || existing?.title || 'Calendar event')
  const sourceId = input.sourceId || existing?.sourceId || null
  const start = parseMutationBoundary(input.startAt) ?? existing?.startAt ?? null
  const end = parseMutationBoundary(input.endAt) ?? existing?.endAt ?? null
  const allDay = input.allDay ?? existing?.allDay ?? false
  const location = clean(input.location ?? existing?.location ?? '')
  const attendees = (input.attendees ?? existing?.attendees ?? []).map(clean).filter(Boolean)
  const description = clean(input.description ?? existing?.description ?? '')
  const action = mode === 'update' ? 'calendar_update_requested' : 'calendar_create_requested'
  const summaryParts = [
    `${mode === 'update' ? 'Update' : 'Create'} "${title}"`,
    start ? `when ${formatMutationWhen(start, end, allDay)}` : '',
    location ? `at ${location}` : '',
    attendees.length ? `with ${attendees.join(', ')}` : '',
    description ? `notes: ${snippet(description, 160)}` : '',
  ].filter(Boolean)
  const sources = existing ? [calendarSourceRef(existing)] : undefined
  return {
    id: randomUUID(),
    connectorId: 'calendar',
    pluginName: 'Calendar',
    action,
    direction: 'outbound',
    classification: 'approval',
    priority: 'high',
    trustLevel: 'approve',
    requiresApproval: true,
    approvalId: `approval-${randomUUID().slice(0, 8)}`,
    approvalStatus: 'pending',
    resolvedAt: null,
    resolvedBy: null,
    title,
    summary: `${summaryParts.join('; ')}. Calendar file is not modified until the request is approved.`,
    actor: attendees[0] || input.organizer || existing?.organizer || 'Calendar',
    threadKey: mode === 'update'
      ? calendarSourceId({ sourceId: existing?.sourceId ?? sourceId ?? 'calendar', id: existing?.id ?? input.id ?? 'event' })
      : `calendar:create:${sourceId || 'default'}:${randomUUID().slice(0, 8)}`,
    payloadKeys: ['title', 'start_at', 'end_at', 'all_day', 'location', 'description', 'attendees', ...(sourceId ? ['source_id'] : []), ...(mode === 'update' ? ['event_id'] : [])],
    sources,
    createdAt: now,
  }
}

async function collectCalendarFiles(root: string, kind: CalendarSourceConfig['kind']): Promise<string[]> {
  const stat = await fs.stat(root)
  if (kind === 'ics_file' || stat.isFile()) {
    if (path.extname(root).toLowerCase() !== '.ics') throw new Error('Calendar source is not an .ics file.')
    return [root]
  }
  if (!stat.isDirectory()) throw new Error('Calendar source is not a folder.')
  const files: string[] = []
  const walk = async (dir: string, depth: number) => {
    if (depth > 6 || files.length >= MAX_SCAN_FILES) return
    const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (files.length >= MAX_SCAN_FILES) break
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        if (entry.name === '.git' || entry.name === 'node_modules') continue
        await walk(full, depth + 1)
      } else if (entry.isFile() && path.extname(entry.name).toLowerCase() === '.ics') {
        files.push(full)
      }
    }
  }
  await walk(root, 0)
  return files
}

async function readCalendarFile(source: CalendarSourceConfig, file: string): Promise<CalendarReadResult[]> {
  const stat = await fs.stat(file)
  if (stat.size > MAX_FILE_BYTES) throw new Error('Calendar file is too large to scan.')
  const text = await fs.readFile(file, 'utf8')
  return parseCalendarEvents(text, source, file)
}

function parseCalendarEvents(text: string, source: CalendarSourceConfig, file: string): CalendarReadResult[] {
  const unfolded = text.replace(/\r?\n[ \t]/g, '')
  const blocks = unfolded.match(/BEGIN:VEVENT[\s\S]*?END:VEVENT/g) ?? []
  return blocks.map((block, index) => eventFromBlock(block, source, file, index)).filter(Boolean) as CalendarReadResult[]
}

function eventFromBlock(block: string, source: CalendarSourceConfig, file: string, index: number): CalendarReadResult | null {
  const props = block.split(/\r?\n/).map(parseProperty).filter(Boolean) as ParsedCalendarProperty[]
  const raw: Record<string, string> = {}
  for (const prop of props) raw[prop.name] ??= decodeIcsText(prop.value)
  const uid = raw.UID || `${file}:${index}`
  const startProp = props.find((prop) => prop.name === 'DTSTART')
  const endProp = props.find((prop) => prop.name === 'DTEND')
  const startAt = parseIcsDate(startProp)
  const endAt = parseIcsDate(endProp)
  const title = raw.SUMMARY || '(untitled)'
  const id = stableEventId(source.id, uid, startProp?.value ?? '', title)
  return {
    id,
    uid,
    sourceId: source.id,
    sourceLabel: source.label,
    path: file,
    title,
    startAt,
    endAt,
    allDay: isAllDay(startProp),
    location: raw.LOCATION || '',
    description: raw.DESCRIPTION || '',
    organizer: emailish(raw.ORGANIZER || ''),
    attendees: props.filter((prop) => prop.name === 'ATTENDEE').map((prop) => emailish(prop.value)).filter(Boolean),
    status: raw.STATUS || '',
    raw,
  }
}

export function parseProperty(line: string): ParsedCalendarProperty | null {
  // RFC 5545 param values may be DQUOTE-wrapped and contain ';' or ':' literally
  // (e.g. ATTENDEE;CN="Doe; John":mailto:j@example.test). Split name/params/value honoring
  // quotes so an embedded ';' isn't read as a param boundary and an embedded ':'
  // isn't read as the value boundary.
  let inQuotes = false
  let colon = -1
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"') inQuotes = !inQuotes
    else if (ch === ':' && !inQuotes) { colon = i; break }
  }
  if (colon < 0) return null
  const head = line.slice(0, colon)
  const value = line.slice(colon + 1)
  // Split the head on UNQUOTED ';' only.
  const segments: string[] = []
  let current = ''
  inQuotes = false
  for (let i = 0; i < head.length; i++) {
    const ch = head[i]
    if (ch === '"') { inQuotes = !inQuotes; current += ch; continue }
    if (ch === ';' && !inQuotes) { segments.push(current); current = ''; continue }
    current += ch
  }
  segments.push(current)
  const [name, ...params] = segments
  return { name: name.toUpperCase(), params: params.join(';'), value }
}

function parseIcsDate(prop?: ParsedCalendarProperty): number | null {
  if (!prop) return null
  const value = prop.value.trim()
  const match = value.match(/^(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2})(Z)?)?$/)
  if (!match) {
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  const [, y, mo, d, h = '00', mi = '00', s = '00', z] = match
  const nums = [y, mo, d, h, mi, s].map(Number)
  if (z) return Date.UTC(nums[0], nums[1] - 1, nums[2], nums[3], nums[4], nums[5])
  return new Date(nums[0], nums[1] - 1, nums[2], nums[3], nums[4], nums[5]).getTime()
}

function isAllDay(prop?: ParsedCalendarProperty): boolean {
  return !!prop && (prop.params.toUpperCase().includes('VALUE=DATE') || !prop.value.includes('T'))
}

function calendarWindow(opts: CalendarListOptions): { from: number | null; to: number | null } {
  if (opts.day) {
    const start = startOfDay(opts.day)
    return { from: start, to: start + 24 * 60 * 60_000 }
  }
  return { from: parseBoundary(opts.from), to: parseBoundary(opts.to) }
}

function parseBoundary(value: CalendarListOptions['from']): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return startOfDay(value)
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function parseMutationBoundary(value: CalendarMutationInput['startAt']): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : parseBoundary(value)
  }
  return null
}

function startOfDay(day: string | number): number {
  const date = typeof day === 'number' ? new Date(day) : new Date(`${day}T00:00:00`)
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()
}

function dayLabel(value: string | number): string {
  const date = typeof value === 'string' ? new Date(`${value}T00:00:00`) : new Date(value)
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

function timeRange(event: CalendarEvent): string {
  if (!event.startAt) return 'no time'
  if (event.allDay) return 'all day'
  const start = new Date(event.startAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const end = event.endAt ? new Date(event.endAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''
  return end ? `${start}-${end}` : start
}

function stableEventId(sourceId: string, uid: string, start: string, title: string): string {
  return createHash('sha256').update(`${sourceId}|${uid}|${start}|${title}`).digest('hex').slice(0, 16)
}

function normalizeCalendarId(id: string): string {
  return String(id || '').trim().split(':').pop() || ''
}

function decodeIcsText(value: string): string {
  return value.replace(/\\n/gi, '\n').replace(/\\,/g, ',').replace(/\\;/g, ';').replace(/\\\\/g, '\\').trim()
}

function emailish(value: string): string {
  return decodeIcsText(value).replace(/^mailto:/i, '').trim()
}

function clean(value: string): string {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function snippet(value: string, max: number): string {
  const cleanValue = clean(value)
  return cleanValue.length > max ? `${cleanValue.slice(0, max - 3)}...` : cleanValue
}

function formatMutationWhen(start: number, end: number | null, allDay: boolean): string {
  if (allDay) return new Date(start).toLocaleDateString()
  const startLabel = new Date(start).toLocaleString()
  const endLabel = end ? new Date(end).toLocaleString() : ''
  return endLabel ? `${startLabel} - ${endLabel}` : startLabel
}
