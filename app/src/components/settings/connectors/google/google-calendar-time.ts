export function macTimeZone(): string { return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC' }
export function safeCalendarTimeZone(zone?: string): string {
  try { if (zone) { new Intl.DateTimeFormat('en', { timeZone: zone }).format(0); return zone } } catch { /* Show usable controls with the Mac zone when remote metadata is invalid. */ }
  return macTimeZone()
}
function partsAt(date: Date, timeZone: string): Record<string, string> {
  return Object.fromEntries(new Intl.DateTimeFormat('en-GB', { timeZone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }).formatToParts(date).map((part) => [part.type, part.value]))
}
export function timestampInZone(date: Date, timeZone: string): string {
  const parts = partsAt(date, timeZone)
  const localAsUtc = Date.UTC(Number(parts.year), Number(parts.month) - 1, Number(parts.day), Number(parts.hour), Number(parts.minute), Number(parts.second))
  const offset = Math.round((localAsUtc - Math.floor(date.getTime() / 1000) * 1000) / 60_000)
  const sign = offset < 0 ? '-' : '+'
  const hours = String(Math.floor(Math.abs(offset) / 60)).padStart(2, '0')
  const minutes = String(Math.abs(offset) % 60).padStart(2, '0')
  return `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}${sign}${hours}:${minutes}`
}
export function timedEventError(start: string, end: string, timeZone: string): string | null {
  try { new Intl.DateTimeFormat('en', { timeZone }).format(0) } catch { return 'Choose an IANA time zone, such as Europe/London or America/New_York.' }
  for (const value of [start, end]) {
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/.test(value) || !Number.isFinite(Date.parse(value))) return 'Use a full date and time with seconds and an explicit offset, such as 2026-10-16T10:00:00+01:00.'
    const formatted = timestampInZone(new Date(value), timeZone)
    if (formatted.slice(0, 19) !== value.slice(0, 19)) return 'The date or UTC offset does not match this time zone; check the daylight-saving transition and choose an explicit valid offset.'
  }
  const duration = Date.parse(end) - Date.parse(start)
  if (duration <= 0 || duration > 366 * 86_400_000) return 'End must be after start and the range must be at most one year.'
  return null
}
export function calendarTimeLabel(value: string, zone = macTimeZone()): string {
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return `${value} · All day`
  if (!Number.isFinite(Date.parse(value))) return value || 'Time unavailable'
  try { return `${new Intl.DateTimeFormat('en-GB', { timeZone: zone, dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))} · ${zone} · ${timestampInZone(new Date(value), zone).slice(19)}` }
  catch { return `${value} · Time zone unavailable` }
}

export function wallTimeCandidates(local: string, timeZone: string): string[] {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$/.test(local)) return []
  const wall = local.length === 16 ? `${local}:00` : local
  const guess = Date.parse(`${wall}Z`)
  if (!Number.isFinite(guess)) return []
  try {
    const offsets = new Set([-36, 0, 36].map((hours) => {
      const probe = new Date(guess + hours * 3_600_000)
      const stamp = timestampInZone(probe, timeZone)
      return Date.parse(`${stamp.slice(0, 19)}Z`) - Math.floor(probe.getTime() / 1000) * 1000
    }))
    return [...offsets].map((offset) => timestampInZone(new Date(guess - offset), timeZone)).filter((candidate) => candidate.slice(0, 19) === wall).sort()
  } catch { return [] }
}

export function allDayEventError(start: string, end: string, timeZone: string): string | null {
  try { new Intl.DateTimeFormat('en', { timeZone }).format(0) } catch { return 'Choose a valid IANA time zone.' }
  for (const value of [start, end]) if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString().slice(0, 10) !== value) return 'Choose valid all-day calendar dates.'
  const duration = Date.parse(end) - Date.parse(start)
  return duration <= 0 || duration > 366 * 86_400_000 ? 'The exclusive end date must be after start and at most one year later.' : null
}
