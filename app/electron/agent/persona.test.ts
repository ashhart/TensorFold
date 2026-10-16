// Run: jiti electron/agent/persona.test.ts
import assert from 'node:assert/strict'
import { resolvePersona, personaSystemBlock, formatDateOnly, type PersonaProfile } from './persona.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function p(scope: string, body = 'voice', enabled = true): PersonaProfile {
  return { id: scope, scope, name: scope, body, enabled, updatedAt: 0 }
}

console.log('persona tests:')

{
  // Most-specific-first: connector > assistant > workspace > global.
  const profiles = [p('global'), p('assistant'), p('workspace:w1'), p('connector:imessage')]
  assert.equal(resolvePersona(profiles, { connectorId: 'imessage', isAssistant: true, workspaceId: 'w1' })?.scope, 'connector:imessage')
  assert.equal(resolvePersona(profiles, { isAssistant: true, workspaceId: 'w1' })?.scope, 'assistant')
  assert.equal(resolvePersona(profiles, { workspaceId: 'w1' })?.scope, 'workspace:w1')
  assert.equal(resolvePersona(profiles, {})?.scope, 'global')
  // A connector with no connector-scoped persona falls through to assistant/global.
  assert.equal(resolvePersona(profiles, { connectorId: 'slack', isAssistant: true })?.scope, 'assistant')
  ok('resolvePersona: connector > assistant > workspace > global, with fall-through')
}

{
  // Disabled or empty-body personas are ignored (fall through to the next match).
  const profiles = [p('connector:imessage', '   ', true), p('assistant', 'warm', false), p('global', 'default')]
  assert.equal(resolvePersona(profiles, { connectorId: 'imessage', isAssistant: true })?.scope, 'global', 'empty + disabled skipped')
  assert.equal(resolvePersona([], { isAssistant: true }), null, 'no profiles -> null (built-in fallback)')
  ok('resolvePersona: skips disabled/empty, null when nothing matches')
}

{
  // The system block frames persona + user profile as untrusted, includes a date-only
  // stamp, and never bakes minute precision.
  const now = new Date('2026-06-26T15:30:00Z')
  const block = personaSystemBlock(p('global', 'You are a concise, warm assistant.'), { userProfile: 'Alex, ships fast, hates fluff.', now })
  assert.match(block, /<persona>[\s\S]*concise, warm[\s\S]*<\/persona>/)
  assert.match(block, /<user_profile>[\s\S]*Alex[\s\S]*<\/user_profile>/)
  assert.match(block, /shapes tone\/wording only|does NOT override/i, 'persona is framed as voice-only, not instructions')
  assert.match(block, /never follow instructions/i, 'user profile framed untrusted')
  assert.match(block, /Today is .*2026/, 'date-only stamp present')
  assert.ok(!/\d{2}:\d{2}/.test(block), 'no minute-precision timestamp (cache-safe)')
  ok('personaSystemBlock: untrusted framing + date-only stamp, no minute precision')
}

{
  // Empty when there is nothing to inject (callers prepend unconditionally).
  assert.equal(personaSystemBlock(null, {}), '')
  assert.equal(personaSystemBlock(null, { userProfile: '   ' }), '')
  ok('personaSystemBlock: empty string when nothing to add')
}

{
  // Date formatting is human + date-only.
  assert.equal(formatDateOnly(new Date('2026-06-26T09:00:00')), 'Friday, June 26, 2026')
  ok('formatDateOnly: weekday, month day, year')
}

console.log(`\n${passed} passed`)
