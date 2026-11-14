import assert from 'node:assert/strict'
import { test } from 'node:test'
import { check, checkJson } from '../src/schema.ts'

test('const, pattern, lengths and own properties', () => {
  assert.equal(checkJson('8', { type: 'integer', const: 7 }).ok, false)
  assert.equal(checkJson('"ab"', { type: 'string', pattern: '^[a-z]{3}$' }).ok, false)
  assert.equal(checkJson('[1]', { type: 'array', minItems: 2 }).ok, false)
  assert.deepEqual(check({}, { type: 'object', required: ['toString'] }), ['$.toString: missing'])
  assert.match(checkJson('{"a": 1}', { type: 'object', oneOf: [{}] }).detail, /oneOf not checked/)
  assert.equal(checkJson('{"city": "Rome"}', { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] }).detail, 'valid JSON, matches the schema')
})

test('const and enum compare objects whatever their key order; boolean schemas mean all or nothing', () => {
  assert.deepEqual(check({ b: 2, a: 1 }, { const: { a: 1, b: 2 } }), [])
  assert.equal(check({ a: 1 }, { const: { a: 1, b: 2 } }).length, 1)
  assert.deepEqual(check({ y: [1, { k: 'v', j: 0 }] }, { enum: [{ y: [1, { j: 0, k: 'v' }] }] }), [])
  assert.equal(check([1], { type: 'array', items: false }).length, 1)
  assert.deepEqual(check([], { type: 'array', items: false }), [])
  assert.deepEqual(check({ x: 3 }, { type: 'object', properties: { x: true } }), [])
  assert.equal(check({ x: 3 }, { type: 'object', properties: { x: false } }).length, 1)
  assert.equal(check(5, false).length, 1)
  assert.deepEqual(check(5, true), [])
})
