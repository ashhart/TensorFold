// NL automation-intent builder (VST-038): parse a plain-English description into a draft
// automation config, so "daily at 9am, summarize my inbox and email it to me"
// becomes a scheduled automation the user can review + enable. Pure and deterministic — no
// model call — so it's testable and instant; a model-assisted refinement can layer on top.
//
// The parser is intentionally conservative: it extracts what it can confidently infer
// (cadence, delivery target, mode) and leaves the rest at safe defaults. Every field it
// sets is reported in `matched` so the UI can show what it understood and let the user
// correct it — never a silent guess presented as fact.

import { normalizeAutomationTimeZone } from '../../shared/automation-time'

export type IntentScheduleKind = 'manual' | 'interval' | 'daily'
export type IntentMode = 'chat' | 'agent' | 'digest'
export type IntentDeliver = 'inbox' | 'email' | 'telegram'

export interface AutomationIntent {
  name: string
  prompt: string
  scheduleKind: IntentScheduleKind
  everyMinutes: number
  atHour: number
  atMinute: number
  timeZone?: string
  mode: IntentMode
  deliverTo: IntentDeliver
  // What the parser confidently extracted (for a "here's what I understood" UI); anything
  // not listed fell back to a default the user should confirm.
  matched: string[]
}

const UNIT_MINUTES: Record<string, number> = {
  minute: 1, minutes: 1, min: 1, mins: 1,
  hour: 60, hours: 60, hr: 60, hrs: 60,
  day: 1440, days: 1440,
}

// "every 5 minutes" / "every hour" / "each 30 mins" → interval minutes, or null.
function parseInterval(text: string): number | null {
  const m = /\b(?:every|each)\s+(?:(\d+)\s+)?(minutes?|mins?|min|hours?|hrs?|hr|days?)\b/i.exec(text)
  if (!m) return null
  const count = m[1] ? parseInt(m[1], 10) : 1
  const unit = UNIT_MINUTES[m[2].toLowerCase()]
  if (!unit || count <= 0) return null
  return Math.min(count * unit, 7 * 1440) // cap at a week to keep the interval sane
}

// "at 9am" / "at 14:30" / "at 9:05 pm" → {hour, minute}, or null.
function parseTimeOfDay(text: string): { hour: number; minute: number } | null {
  const m = /\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/i.exec(text)
  if (!m) return null
  let hour = parseInt(m[1], 10)
  const minute = m[2] ? parseInt(m[2], 10) : 0
  const mer = m[3]?.toLowerCase()
  if (mer === 'pm' && hour < 12) hour += 12
  if (mer === 'am' && hour === 12) hour = 0
  if (hour > 23 || minute > 59) return null
  return { hour, minute }
}

// Strip the schedule/delivery phrasing so the leftover reads as the actual instruction.
function extractPrompt(text: string): string {
  return text
    .replace(/\b(?:every|each)\s+(?:\d+\s+)?(?:minutes?|mins?|min|hours?|hrs?|hr|days?)\b/gi, '')
    .replace(/\b(?:(?:every|each)\s+days?|everyday)\b/gi, '')
    .replace(/\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b/gi, '')
    .replace(/\b(?:daily|each morning|every morning|nightly)\b/gi, '')
    .replace(/\b(?:and\s+)?(?:email|e-mail|send|deliver|message|text|dm)\s+(?:it\s+|the (?:result|summary|output)\s+)?(?:to\s+)?(?:me|my inbox|telegram|slack)?\b/gi, '')
    .replace(/^[\s,.:;-]+|[\s,.:;-]+$/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function titleFromPrompt(prompt: string, fallback: string): string {
  const firstClause = prompt.split(/[.,\n]/)[0].trim()
  const base = firstClause || fallback
  const title = base.charAt(0).toUpperCase() + base.slice(1)
  return title.length > 60 ? `${title.slice(0, 57)}…` : title
}

export function parseAutomationIntent(input: string, defaultTimeZone?: string): AutomationIntent {
  const text = (input ?? '').trim()
  const lower = text.toLowerCase()
  const matched: string[] = []

  // These calendars have no scheduler representation. Refuse an explicit restriction
  // before inferring a daily clock time, rather than widening it to every day.
  const calendar = '(?:weekdays?|weekends?|weeks?|months?|years?|mon(?:day)?s?|tue(?:s(?:day)?)?s?|wed(?:nesday)?s?|thu(?:rs(?:day)?)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?)'
  const unsupportedCalendar = new RegExp(`\\b(?:every|each|on|next|this)\\s+(?:(?:other|\\d+)\\s+)?${calendar}\\b`, 'i').test(text)
    || /^(?:(?:run|repeat|schedule)\s+)?(?:weekly|monthly|yearly|annually|biweekly|fortnightly|weekdays|weekends)\b/i.test(text)
    || new RegExp(`^(?:run|repeat|schedule)\\s+${calendar}\\b`, 'i').test(text)
    || new RegExp(`^${calendar}\\s+at\\b`, 'i').test(text)
    || /\b(?:run|repeat|schedule(?:d)?)\s+(?:weekly|monthly|yearly|annually|biweekly|fortnightly)\b/i.test(text)
    || /\b(?:weekdays?|weekends?)\s+only\b/i.test(text)
    || new RegExp(`\\bat\\s+\\d{1,2}(?::\\d{2})?\\s*(?:am|pm)?\\s+${calendar}\\b`, 'i').test(text)
  if (unsupportedCalendar) throw new Error('This calendar restriction is not supported yet. Choose a daily time with an IANA timezone or an elapsed-time interval; weekday, weekly and monthly calendars cannot be scheduled.')

  let scheduleKind: IntentScheduleKind = 'manual'
  let everyMinutes = 60
  let atHour = 9
  let atMinute = 0

  // A prefix, suffix or explicit "run hourly" is cadence; an ordinary phrase such
  // as "calculate the hourly rate" stays a manual instruction with its text intact.
  const hourly = /^(?:run\s+)?hourly\b|\b(?:run|repeat|schedule)\s+hourly\b|\bhourly\s*[,.!?]?$/i.exec(text)
  const interval = parseInterval(lower) ?? (hourly ? 60 : null)
  const timeOfDay = parseTimeOfDay(lower)
  const mentionsDaily = /\b(daily|every ?day|each day|every morning|each morning|nightly)\b/.test(lower)
  if (interval != null && timeOfDay && !(interval === 1440 && mentionsDaily && !hourly)) {
    throw new Error('Elapsed-time intervals with a start time are not supported yet; choose an interval or a daily time.')
  }

  if (interval != null && !timeOfDay) {
    scheduleKind = 'interval'
    everyMinutes = interval
    matched.push(`runs every ${everyMinutes} min`)
  } else if (timeOfDay || mentionsDaily) {
    scheduleKind = 'daily'
    if (timeOfDay) { atHour = timeOfDay.hour; atMinute = timeOfDay.minute; matched.push(`daily at ${String(atHour).padStart(2, '0')}:${String(atMinute).padStart(2, '0')}`) }
    else { matched.push('runs daily') }
  } else if (interval != null) {
    scheduleKind = 'interval'
    everyMinutes = interval
    matched.push(`runs every ${everyMinutes} min`)
  }

  // Delivery target.
  let deliverTo: IntentDeliver = 'inbox'
  if (/\b(telegram|dm me|text me)\b/.test(lower)) { deliverTo = 'telegram'; matched.push('delivers to Telegram') }
  else if (/\b(e-?mail|send.*(?:to )?me|to my inbox by email)\b/.test(lower)) { deliverTo = 'email'; matched.push('delivers by email') }

  // Mode: a digest phrasing needs no model; "agent"/"run"/"fix"/"edit"/"open a PR" implies
  // the tool loop; otherwise a single chat completion.
  let mode: IntentMode = 'chat'
  if (/\b(digest|briefing|summary of (?:my )?(?:activity|jobs|day)|status report)\b/.test(lower)) { mode = 'digest'; matched.push('digest mode (no model)') }
  else if (/\b(agent|run tests|fix|edit|refactor|open a pr|open a pull request|commit|build|deploy|check out|clone)\b/.test(lower)) { mode = 'agent'; matched.push('agent mode (tool loop)') }

  // Accept an explicit IANA identifier, not ambiguous abbreviations such as CST.
  // An invalid identifier is an error rather than an invitation to use the host zone.
  const zoneMatch = /\b(?:time\s*zone|tz)\b\s*[:=]?\s*([^\s,;]+)/i.exec(text)
    // A slash identifier immediately after the clock time expresses timezone intent,
    // even if its region is unknown; a task path later in the instruction does not.
    ?? /\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+(?:in\s+)?([A-Za-z][A-Za-z0-9_.+-]*(?:\/[A-Za-z0-9_.+-]+)+)\b/i.exec(text)
    ?? /\b(?:in\s+)?(?:Africa|America|Antarctica|Arctic|Asia|Atlantic|Australia|Europe|Indian|Pacific|Etc|US|Canada|Brazil|Mexico|Chile)\/[A-Za-z0-9_+./-]+\b/i.exec(text)
    ?? /\bin\s+(UTC)\b/i.exec(text)
    ?? /\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+(UTC)\b/i.exec(text)
  let timeZone: string | undefined
  if (scheduleKind === 'daily' && !zoneMatch && /\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+(?:in\s+)?(?:GMT|BST|EST|EDT|CST|CDT|MST|MDT|PST|PDT|IST|CET|CEST)\b/i.test(text)) {
    throw new Error('Use an explicit IANA timezone, such as America/Los_Angeles or UTC, instead of a timezone abbreviation.')
  }
  if (zoneMatch && scheduleKind === 'daily') {
    const zone = zoneMatch[1] ?? zoneMatch[0].replace(/^in\s+/i, '')
    timeZone = normalizeAutomationTimeZone(zone)
    matched.push(`timezone ${timeZone}`)
  } else if (scheduleKind === 'daily' && defaultTimeZone !== undefined) {
    timeZone = normalizeAutomationTimeZone(defaultTimeZone)
    matched.push(`timezone ${timeZone} (review before enabling)`)
  }
  const withoutZone = timeZone && zoneMatch ? text.replace(zoneMatch[0], '') : text
  const prompt = extractPrompt(hourly && scheduleKind === 'interval' ? withoutZone.replace(hourly[0], '') : withoutZone) || text
  return {
    name: titleFromPrompt(prompt, 'Automation'),
    prompt,
    scheduleKind,
    everyMinutes,
    atHour,
    atMinute,
    ...(timeZone ? { timeZone } : {}),
    mode,
    deliverTo,
    matched,
  }
}
