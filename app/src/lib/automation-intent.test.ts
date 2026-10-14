// Run: jiti src/lib/automation-intent.test.ts
//
// NL automation-intent builder (VST-038): deterministic parse of plain-English descriptions
// into draft automation configs — cadence, time-of-day, mode, delivery, cleaned prompt.
import assert from 'node:assert/strict'
import { parseAutomationIntent } from './automation-intent'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('automation-intent tests:')

{
  const intent = parseAutomationIntent('every 5 minutes, check the CI dashboard for failures')
  assert.equal(intent.scheduleKind, 'interval')
  assert.equal(intent.everyMinutes, 5)
  assert.match(intent.prompt, /check the CI dashboard/i)
  assert.ok(!/every 5 minutes/i.test(intent.prompt), 'the cadence phrasing is stripped from the prompt')
  assert.ok(intent.matched.some((m) => /every 5 min/.test(m)))
  ok('interval: "every 5 minutes" → interval cadence, cleaned prompt')
}

{
  assert.equal(parseAutomationIntent('every hour summarize new emails').everyMinutes, 60)
  assert.equal(parseAutomationIntent('each 2 hours run the smoke tests').everyMinutes, 120)
  assert.equal(parseAutomationIntent('every 3 days prune old branches').everyMinutes, 3 * 1440)
  ok('interval units: minutes/hours/days with and without a count')
}

{
  const nine = parseAutomationIntent('daily at 9am, email me a summary of my calendar')
  assert.equal(nine.scheduleKind, 'daily')
  assert.equal(nine.atHour, 9)
  assert.equal(nine.atMinute, 0)
  assert.equal(nine.deliverTo, 'email')

  const evening = parseAutomationIntent('at 6:30 pm send me the day\'s standup notes')
  assert.equal(evening.scheduleKind, 'daily')
  assert.equal(evening.atHour, 18)
  assert.equal(evening.atMinute, 30)

  const midnight = parseAutomationIntent('at 12am archive yesterday\'s logs')
  assert.equal(midnight.atHour, 0)

  const noon = parseAutomationIntent('at 12pm ping the team')
  assert.equal(noon.atHour, 12)
  ok('daily: "at 9am"/"6:30 pm"/"12am"/"12pm" → correct 24h time')
}

{
  for (const prompt of ['hourly summarize alerts', 'summarize alerts hourly', 'run hourly, summarize alerts']) {
    const result = parseAutomationIntent(prompt, 'UTC')
    assert.equal(result.scheduleKind, 'interval', 'an intended hourly cadence must never become daily')
    assert.equal(result.everyMinutes, 60)
    assert.equal(result.timeZone, undefined, 'hourly is an elapsed-time interval')
    assert.match(result.prompt, /summarize alerts/i)
  }
  const ordinary = parseAutomationIntent('calculate the hourly rate for this project')
  assert.equal(ordinary.scheduleKind, 'manual')
  assert.match(ordinary.prompt, /hourly rate/, 'an ordinary task modifier is not a schedule and remains in the instruction')
  ok('hourly prefix, suffix and run phrasing use an interval without changing ordinary task text')
}

{
  for (const prompt of [
    'every weekday, post the metrics to telegram',
    'every weekday at 9am summarize alerts',
    'every Monday at 9am summarize alerts',
    'at 9am on Fridays summarize alerts',
    'weekly at 9am summarize alerts',
    'run weekly, summarize alerts',
    'each 2 weeks summarize alerts',
    'every month at 9am summarize alerts',
    'monthly summarize alerts',
    'daily at 9am on weekdays summarize alerts',
    'at 9am next Monday summarize alerts',
  ]) assert.throws(() => parseAutomationIntent(prompt, 'UTC'), /calendar.*not supported/i, 'unsupported restrictions must not become daily schedules')
  assert.equal(parseAutomationIntent('every 7 days summarize alerts').scheduleKind, 'interval', 'an explicitly supported elapsed-day interval still works')
  assert.equal(parseAutomationIntent('Sun activity summary').scheduleKind, 'manual', 'a weekday abbreviation used as an ordinary topic is not a calendar')
  assert.equal(parseAutomationIntent('daily at 9am summarize the monthly report', 'UTC').scheduleKind, 'daily', 'a report description is not a monthly calendar restriction')
  for (const prompt of ['hourly at 9am summarize alerts', 'every 2 hours at 9am summarize alerts', 'every 3 days at 9am summarize alerts']) {
    assert.throws(() => parseAutomationIntent(prompt, 'UTC'), /interval.*start time/i, 'an unsupported interval start-time restriction must be reviewed rather than silently dropped')
  }
  assert.equal(parseAutomationIntent('every day at 9am summarize alerts', 'UTC').scheduleKind, 'daily', 'the supported daily clock-time phrase remains a daily schedule')
  ok('unsupported weekday, weekly and monthly calendars are rejected while supported daily and elapsed-day schedules remain intact')
}

{
  assert.equal(parseAutomationIntent('give me a digest of my activity each morning').mode, 'digest')
  assert.equal(parseAutomationIntent('every day at 8am, run tests and fix any failures').mode, 'agent')
  assert.equal(parseAutomationIntent('summarize the latest AI news').mode, 'chat')
  ok('mode: digest / agent / chat inferred from phrasing')
}

{
  const manual = parseAutomationIntent('draft a weekly investor update from our metrics')
  assert.equal(manual.scheduleKind, 'manual')
  assert.match(manual.prompt, /investor update/i)
  assert.ok(manual.name.length > 0 && manual.name.length <= 60)
  ok('no schedule → manual, prompt preserved, a title is derived')
}

{
  const empty = parseAutomationIntent('   ')
  assert.equal(empty.scheduleKind, 'manual')
  assert.equal(empty.mode, 'chat')
  ok('empty input: safe defaults, no throw')
}

{
  const explicit = parseAutomationIntent('daily at 9am in Europe/London, summarize my tasks', 'America/New_York')
  assert.equal(explicit.timeZone, 'Europe/London', 'an explicit zone wins over the device zone')
  assert.ok(!/Europe\/London|\bin\s*,/.test(explicit.prompt), 'recognized timing context is removed from the task instruction')
  assert.equal(parseAutomationIntent('daily at 9am UTC, summarize my tasks').timeZone, 'UTC')
  assert.equal(parseAutomationIntent('daily at 9am timezone: Asia/Tokyo, summarize my tasks').timeZone, 'Asia/Tokyo')
  assert.equal(parseAutomationIntent('daily at 9am, summarize my tasks', 'America/New_York').timeZone, 'America/New_York')
  assert.equal(parseAutomationIntent('daily at 9am, summarize my tasks').timeZone, undefined, 'the parser cannot invent a timezone when none is supplied')
  assert.throws(() => parseAutomationIntent('daily at 9am in Europe/Unknown, summarize my tasks', 'Europe/London'), /IANA timezone/, 'an intended invalid zone cannot fall back to the device')
  assert.throws(() => parseAutomationIntent('daily at 9am in Mars/Olympus, summarize tasks', 'Europe/London'), /IANA timezone/, 'an unknown timezone region cannot fall back to the device')
  assert.throws(() => parseAutomationIntent('daily at 9am mars/olympus, summarize tasks', 'Europe/London'), /IANA timezone/, 'a bare slash identifier beside the time is also validated')
  assert.equal(parseAutomationIntent('daily at 9am run tests in src/lib', 'Europe/London').timeZone, 'Europe/London', 'an ordinary task directory is not confused with a timezone suffix')
  assert.throws(() => parseAutomationIntent('daily at 9am timezone CST, summarize my tasks', 'Europe/London'), /IANA timezone/, 'ambiguous explicit abbreviations are refused')
  assert.throws(() => parseAutomationIntent('daily at 9am PST, summarize my tasks', 'Europe/London'), /IANA timezone/, 'a timezone suffix cannot silently become device-local time')
  assert.equal(parseAutomationIntent('each 2 hours summarize my tasks', 'Europe/London').timeZone, undefined, 'elapsed-time intervals do not depend on a wall-clock timezone')
  ok('timezone: explicit IANA zones, reviewed device default, invalid zones and elapsed-time intervals')
}

console.log(`  ${passed} passed`)
