// Run: jiti electron/agent/calendar-connector.fix.test.ts
// The bug: parseProperty split params on ';' and name/value on ':' without honoring RFC 5545
// double-quoted parameter values, mis-parsing properties with a quoted ';' or ':' in a param.
import assert from 'node:assert/strict'
import { parseProperty } from './calendar-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('calendar-connector fix tests:')

{
  // A quoted ';' inside a param value must NOT be read as a param boundary.
  const p = parseProperty('ATTENDEE;CN="Doe; John";RSVP=TRUE:mailto:j@example.test')
  assert.ok(p)
  assert.equal(p!.name, 'ATTENDEE')
  assert.equal(p!.value, 'mailto:j@example.test')
  assert.equal(p!.params, 'CN="Doe; John";RSVP=TRUE')
  ok('quoted ";" in a param value is not split as a param boundary')
}

{
  // A quoted ':' inside a param value must NOT be read as the value boundary.
  const p = parseProperty('ATTENDEE;CN="Smith:John":mailto:s@example.test')
  assert.ok(p)
  assert.equal(p!.name, 'ATTENDEE')
  assert.equal(p!.value, 'mailto:s@example.test')
  assert.equal(p!.params, 'CN="Smith:John"')
  ok('quoted ":" in a param value does not truncate the property')
}

{
  // Plain unquoted properties still parse exactly as before.
  const p = parseProperty('SUMMARY:Team sync')
  assert.equal(p!.name, 'SUMMARY')
  assert.equal(p!.value, 'Team sync')
  assert.equal(p!.params, '')
  ok('plain property without params unchanged')
}

{
  const p = parseProperty('DTSTART;TZID=America/New_York:20240101T120000')
  assert.equal(p!.name, 'DTSTART')
  assert.equal(p!.params, 'TZID=America/New_York')
  assert.equal(p!.value, '20240101T120000')
  ok('unquoted param + value unchanged')
}

{
  // A line with no colon is not a property.
  assert.equal(parseProperty('BEGIN-no-colon'), null)
  ok('a line without a colon returns null')
}

console.log(`\n${passed} checks passed.`)
