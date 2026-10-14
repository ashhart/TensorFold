// Wall-clock scheduling shared by the editor and host, with no Electron or Node imports.
// Pinned schedules skip nonexistent local times and use only the first occurrence of an
// overlapping time. Missing timeZone is reserved for existing host-local schedules.
const DAY_MS = 86_400_000
const HOUR_MS = 3_600_000
const formatters = new Map<string, Intl.DateTimeFormat>()

export function normalizeAutomationTimeZone(value: unknown): string {
  if (typeof value !== 'string' || value.length > 100) throw new Error('Choose a valid IANA timezone, such as Europe/London or UTC.')
  const zone = value.trim()
  if (!/^(?:UTC|[A-Za-z][A-Za-z0-9._+-]*(?:\/[A-Za-z0-9._+-]+)+)$/i.test(zone)) throw new Error('Choose a valid IANA timezone, such as Europe/London or UTC.')
  try {
    return new Intl.DateTimeFormat('en-US', { timeZone: zone }).resolvedOptions().timeZone
  } catch {
    throw new Error('Choose a valid IANA timezone, such as Europe/London or UTC.')
  }
}

export function deviceAutomationTimeZone(): string | undefined {
  try { return normalizeAutomationTimeZone(new Intl.DateTimeFormat().resolvedOptions().timeZone) }
  catch { return undefined }
}

export function automationTimeZones(): string[] {
  const supported = (Intl as typeof Intl & { supportedValuesOf?: (key: string) => string[] }).supportedValuesOf
  let zones: string[] = []
  try { zones = supported?.('timeZone') ?? [] } catch { /* typing a valid zone remains available */ }
  const device = deviceAutomationTimeZone()
  return [...new Set(['UTC', ...(device ? [device] : []), ...zones])].sort()
}

function formatter(zone: string): Intl.DateTimeFormat {
  let value = formatters.get(zone)
  if (!value) {
    value = new Intl.DateTimeFormat('en-US-u-ca-gregory-nu-latn', {
      timeZone: zone, year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
    })
    if (formatters.size >= 64) formatters.clear()
    formatters.set(zone, value)
  }
  return value
}

function parts(format: Intl.DateTimeFormat, instant: number) {
  const values: Record<string, number> = {}
  for (const part of format.formatToParts(instant)) if (part.type !== 'literal') values[part.type] = Number(part.value)
  return { year: values.year, month: values.month, day: values.day, hour: values.hour, minute: values.minute, second: values.second }
}

function wallEpoch(format: Intl.DateTimeFormat, instant: number): number {
  const p = parts(format, instant)
  return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second)
}

export function nextDailyAutomationRun(hour: number, minute: number, from: number, timeZone?: string): number {
  if (!Number.isInteger(hour) || hour < 0 || hour > 23 || !Number.isInteger(minute) || minute < 0 || minute > 59 || !Number.isFinite(from) || !Number.isFinite(new Date(from).getTime())) {
    throw new Error('Choose a valid daily schedule time.')
  }
  if (timeZone === undefined) {
    // Legacy records keep the host's local-time interpretation, including Date's gap
    // normalization. Calendar addition fixes the previous fixed-24-hour DST drift.
    const date = new Date(from)
    date.setHours(hour, minute, 0, 0)
    if (date.getTime() <= from) {
      date.setDate(date.getDate() + 1)
      date.setHours(hour, minute, 0, 0)
    }
    return date.getTime()
  }
  const zone = normalizeAutomationTimeZone(timeZone)
  const format = formatter(zone)
  const current = parts(format, from)
  const startDay = Date.UTC(current.year, current.month - 1, current.day)
  // Sampling offsets on both sides of the target date handles DST, fractional-hour
  // transitions and international-date-line changes. Every candidate is checked against
  // its actual calendar fields; a missing wall time is never shifted into another hour.
  for (let day = 0; day < 8; day++) {
    const nominal = startDay + day * DAY_MS + hour * HOUR_MS + minute * 60_000
    const offsets = new Set<number>()
    for (let distance = -48; distance <= 48; distance += 6) {
      const sample = nominal + distance * HOUR_MS
      offsets.add(wallEpoch(format, sample) - sample)
    }
    const candidates = [...offsets].map((offset) => nominal - offset)
      .filter((instant) => wallEpoch(format, instant) === nominal)
      .sort((a, b) => a - b)
    // Deliberately choose the first occurrence before checking `from`, so the second
    // occurrence of a repeated clock time cannot fire again after the first one passed.
    if (candidates.length && candidates[0] > from) return candidates[0]
  }
  throw new Error('The next daily time could not be resolved in this timezone.')
}

export function dailyAutomationPreview(hour: number, minute: number, from: number, timeZone: string): string {
  const zone = normalizeAutomationTimeZone(timeZone)
  const instant = nextDailyAutomationRun(hour, minute, from, zone)
  return new Intl.DateTimeFormat(undefined, {
    timeZone: zone, year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short',
  }).format(instant)
}
