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
  const weekday = parseAutomationIntent('every weekday, post the metrics to telegram')
  assert.equal(weekday.scheduleKind, 'daily')
  assert.equal(weekday.deliverTo, 'telegram')
  ok('daily keyword without a time defaults to daily + delivery target')
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

console.log(`  ${passed} passed`)
