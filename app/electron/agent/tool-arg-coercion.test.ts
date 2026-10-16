// Run: jiti electron/agent/tool-arg-coercion.test.ts
import assert from 'node:assert/strict'
import { coerceToolArgs } from './tool-arg-coercion.ts'
import { TOOL_SCHEMAS } from './tool-schemas.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('tool-arg-coercion tests:')

// --- string -> number coercion -----------------------------------------------
{
  const schema = { type: 'object', properties: { count: { type: 'number' }, ratio: { type: 'number' } } }
  const out = coerceToolArgs({ count: '42', ratio: '3.14' }, schema)
  assert.equal(out.count, 42)
  assert.equal(typeof out.count, 'number')
  assert.equal(out.ratio, 3.14)
  ok('a numeric string is coerced to a number for a "number" schema property')
}

{
  // 'integer' is a valid JSON-Schema type distinct from 'number' — must also coerce.
  const schema = { type: 'object', properties: { limit: { type: 'integer' } } }
  const out = coerceToolArgs({ limit: '100' }, schema)
  assert.equal(out.limit, 100)
  assert.equal(typeof out.limit, 'number')
  ok('a numeric string is coerced to a number for an "integer" schema property')
}

{
  // Negative numbers and surrounding whitespace are still clean numeric strings.
  const schema = { type: 'object', properties: { delta: { type: 'number' } } }
  const out = coerceToolArgs({ delta: '  -7  ' }, schema)
  assert.equal(out.delta, -7)
  ok('whitespace-padded / negative numeric strings coerce correctly')
}

// --- string -> boolean coercion -----------------------------------------------
{
  const schema = { type: 'object', properties: { unreadOnly: { type: 'boolean' }, force: { type: 'boolean' } } }
  const out = coerceToolArgs({ unreadOnly: 'true', force: 'FALSE' }, schema)
  assert.equal(out.unreadOnly, true)
  assert.equal(typeof out.unreadOnly, 'boolean')
  assert.equal(out.force, false)
  assert.equal(typeof out.force, 'boolean')
  ok('the strings "true"/"false" (any case) are coerced to booleans for a "boolean" schema property')
}

// --- scalar -> array wrapping --------------------------------------------------
{
  const schema = { type: 'object', properties: { tags: { type: 'array' } } }
  const out = coerceToolArgs({ tags: 'urgent' }, schema)
  assert.deepEqual(out.tags, ['urgent'])
  ok('a bare string scalar is wrapped into a single-element array for an "array" schema property')
}

{
  const schema = { type: 'object', properties: { ids: { type: 'array' }, flags: { type: 'array' } } }
  const out = coerceToolArgs({ ids: 7, flags: true }, schema)
  assert.deepEqual(out.ids, [7])
  assert.deepEqual(out.flags, [true])
  ok('numeric/boolean scalars are also wrapped into single-element arrays for an "array" schema property')
}

// --- untouched: already correct type -------------------------------------------
{
  const schema = {
    type: 'object',
    properties: {
      count: { type: 'number' },
      active: { type: 'boolean' },
      tags: { type: 'array' },
      label: { type: 'string' },
    },
  }
  const input = { count: 42, active: true, tags: ['a', 'b'], label: 'hello' }
  const out = coerceToolArgs(input, schema)
  assert.equal(out.count, 42)
  assert.equal(out.active, true)
  assert.deepEqual(out.tags, ['a', 'b'])
  assert.equal(out.label, 'hello')
  ok('values that already match the schema type pass through unchanged')
}

// --- untouched: ambiguous / non-coercible ---------------------------------------
{
  const schema = { type: 'object', properties: { count: { type: 'number' } } }
  const out = coerceToolArgs({ count: 'not-a-number' }, schema)
  assert.equal(out.count, 'not-a-number')
  ok('a non-numeric string is left untouched for a "number" schema property (no throw)')
}

{
  const schema = { type: 'object', properties: { count: { type: 'number' } } }
  const out = coerceToolArgs({ count: '' }, schema)
  assert.equal(out.count, '')
  ok('an empty string is left untouched for a "number" schema property (not coerced to 0)')
}

{
  const schema = { type: 'object', properties: { count: { type: 'number' } } }
  // Whitespace-only must not coerce to 0 either (Number('   ') is 0, a classic footgun).
  const out = coerceToolArgs({ count: '   ' }, schema)
  assert.equal(out.count, '   ')
  ok('a whitespace-only string is left untouched for a "number" schema property (not coerced to 0)')
}

{
  const schema = { type: 'object', properties: { active: { type: 'boolean' } } }
  const out = coerceToolArgs({ active: 'maybe' }, schema)
  assert.equal(out.active, 'maybe')
  ok('a non-true/false string is left untouched for a "boolean" schema property')
}

{
  const schema = { type: 'object', properties: { count: { type: 'number' } } }
  // null/undefined are not strings — must not throw or get coerced to NaN/0.
  const out = coerceToolArgs({ count: null }, schema)
  assert.equal(out.count, null)
  ok('null is left untouched for a "number" schema property (no throw)')
}

{
  const schema = { type: 'object', properties: { tags: { type: 'array' } } }
  // Already an array — must not get double-wrapped.
  const out = coerceToolArgs({ tags: ['x'] }, schema)
  assert.deepEqual(out.tags, ['x'])
  ok('an existing array is left as-is for an "array" schema property (not double-wrapped)')
}

{
  const schema = { type: 'object', properties: { tags: { type: 'array' } } }
  // null is not a "non-array scalar" in the wrap-worthy sense; leave it alone.
  const out = coerceToolArgs({ tags: null }, schema)
  assert.equal(out.tags, null)
  ok('null is left untouched for an "array" schema property (not wrapped to [null])')
}

// --- untouched: key not present in schema.properties -----------------------------
{
  const schema = { type: 'object', properties: { count: { type: 'number' } } }
  const out = coerceToolArgs({ count: '42', extra: 'unrelated', nested: { a: 1 } }, schema)
  assert.equal(out.count, 42)
  assert.equal(out.extra, 'unrelated')
  assert.deepEqual(out.nested, { a: 1 })
  ok('keys with no matching schema property pass through completely untouched')
}

{
  // schema.properties itself missing/empty: nothing should be coerced, nothing thrown.
  const out1 = coerceToolArgs({ count: '42' }, { type: 'object' })
  assert.equal(out1.count, '42')
  const out2 = coerceToolArgs({ count: '42' }, {})
  assert.equal(out2.count, '42')
  ok('a schema with no properties leaves every value untouched (no throw)')
}

// --- immutability ----------------------------------------------------------------
{
  const schema = { type: 'object', properties: { count: { type: 'number' }, active: { type: 'boolean' }, tags: { type: 'array' } } }
  const input = { count: '42', active: 'true', tags: 'one', untouched: 'same' }
  const inputSnapshot = JSON.parse(JSON.stringify(input))
  const out = coerceToolArgs(input, schema)
  assert.deepEqual(input, inputSnapshot, 'input object must not be mutated')
  assert.notEqual(out, input, 'a new object must be returned, not the same reference')
  assert.equal(out.count, 42)
  assert.equal(out.active, true)
  assert.deepEqual(out.tags, ['one'])
  ok('coerceToolArgs is pure: input is unchanged and a new object is returned')
}

// --- realistic end-to-end: an actual TensorFold Studio tool schema ---------------------------
{
  const runCommand = TOOL_SCHEMAS.find((t) => t.function.name === 'run_command')
  assert.ok(runCommand, 'run_command schema must exist in TOOL_SCHEMAS')
  const out = coerceToolArgs({ command: 'npm test' }, runCommand!.function.parameters as Record<string, unknown>)
  assert.deepEqual(out, { command: 'npm test' }, 'a correctly-typed call against a real tool schema passes through unchanged')
  ok('a correctly-typed run_command call against the real TOOL_SCHEMAS entry passes through unchanged')
}

{
  // email_list mixes string/boolean/number — a realistic multi-type schema from tool-schemas.ts.
  const emailList = TOOL_SCHEMAS.find((t) => t.function.name === 'email_list')
  assert.ok(emailList, 'email_list schema must exist in TOOL_SCHEMAS')
  const out = coerceToolArgs(
    { query: 'invoice', unreadOnly: 'true', limit: '10' },
    emailList!.function.parameters as Record<string, unknown>,
  )
  assert.equal(out.query, 'invoice')
  assert.equal(out.unreadOnly, true)
  assert.equal(out.limit, 10)
  ok('a mistyped email_list call (boolean/number as strings) against the real schema is fully repaired')
}

console.log(`\n${passed} checks passed.`)
