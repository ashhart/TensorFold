export function boundedString(value: unknown, label: string, max: number, empty = false): string {
  if (typeof value !== 'string' || value.length > max || /\0/.test(value) || !empty && !value.trim()) throw new Error(`Invalid ${label}.`)
  return value
}
export function googleResourceId(value: unknown, label = 'resource ID'): string {
  const id = boundedString(value, label, 512)
  if (/[\r\n\s/\\?#%]/.test(id) || id === '.' || id === '..') throw new Error(`Invalid ${label}.`)
  return encodeURIComponent(id)
}
export function googlePage(limit = 20, pageToken?: string): { maxResults: string; pageToken?: string } {
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 50) throw new Error('Choose between 1 and 50 results.')
  if (pageToken !== undefined) boundedString(pageToken, 'page token', 2048)
  return { maxResults: String(limit), ...(pageToken ? { pageToken } : {}) }
}
export function validEmail(value: unknown): string {
  const email = boundedString(value, 'email address', 254).trim()
  if (!/^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$/.test(email)) throw new Error('Use a plain email address without display names.')
  return email
}
export function emailList(value: unknown, required = false): string[] {
  if (value === undefined && !required) return []
  if (!Array.isArray(value) || value.length > 50 || required && !value.length) throw new Error('Provide between 1 and 50 email recipients.')
  return [...new Set(value.map(validEmail))]
}
export function rfc3339(value: unknown): string {
  const text = boundedString(value, 'date and time', 64)
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,3})?(?:Z|[+-](\d{2}):(\d{2}))$/.exec(text)
  if (!match || !Number.isFinite(Date.parse(text))) throw new Error('Use an ISO date and time with an explicit UTC offset.')
  const [year, month, day, hour, minute, second, offsetHour = 0, offsetMinute = 0] = match.slice(1).map(value => value === undefined ? 0 : Number(value))
  const days = new Date(Date.UTC(year, month, 0)).getUTCDate()
  if (year < 1000 || month < 1 || month > 12 || day < 1 || day > days || hour > 23 || minute > 59 || second > 59 || offsetHour > 23 || offsetMinute > 59) throw new Error('Choose a valid calendar date and time.')
  return text
}
export function timeWindow(timeMin: unknown, timeMax: unknown): { timeMin: string; timeMax: string } {
  const start = rfc3339(timeMin); const end = rfc3339(timeMax)
  if (Date.parse(end) <= Date.parse(start) || Date.parse(end) - Date.parse(start) > 366 * 24 * 60 * 60 * 1000) throw new Error('Choose a positive time range of at most one year.')
  return { timeMin: start, timeMax: end }
}
export function calendarDate(value: unknown): string {
  const text = boundedString(value, 'all-day date', 10)
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text)
  if (!match) throw new Error('Use YYYY-MM-DD for both dates of an all-day event.')
  const [year, month, day] = match.slice(1).map(Number)
  if (year < 1000 || month < 1 || month > 12 || day < 1 || day > new Date(Date.UTC(year, month, 0)).getUTCDate()) throw new Error('Choose a valid all-day calendar date.')
  return text
}
export function eventWindow(start: unknown, end: unknown): { start: string; end: string } {
  const allDayStart = typeof start === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(start)
  const allDayEnd = typeof end === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(end)
  if (allDayStart !== allDayEnd) throw new Error('Both event boundaries must be all-day dates or timed dates, never a mixture.')
  if (allDayStart) {
    const first = calendarDate(start); const last = calendarDate(end)
    if (last <= first || Date.parse(last) - Date.parse(first) > 366 * 24 * 60 * 60 * 1000) throw new Error('The exclusive all-day end must follow the start by at most one year.')
    return { start: first, end: last }
  }
  const range = timeWindow(start, end)
  return { start: range.timeMin, end: range.timeMax }
}
export function validTimeZone(value: unknown): string {
  const zone = boundedString(value, 'time zone', 100)
  try { new Intl.DateTimeFormat('en', { timeZone: zone }).format(0) } catch { throw new Error('Choose a valid IANA time zone.') }
  return zone
}
export function responseText(value: unknown, max = 16_384): string { return typeof value === 'string' ? value.replace(/\0/g, '').slice(0, max) : '' }
export function responseArray(value: unknown, max = 50): any[] { return Array.isArray(value) ? value.slice(0, max) : [] }
export function returnedPageToken(value: unknown): string | undefined { return typeof value === 'string' && value.length <= 2048 && !/[\r\n\0]/.test(value) ? value : undefined }
