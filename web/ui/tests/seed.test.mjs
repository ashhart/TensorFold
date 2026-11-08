import assert from 'node:assert/strict'
import { test } from 'node:test'
import { drawSeed } from '../src/seed.ts'

test('no seed in the preset: one is drawn per reply, within what a request accepts', () => {
  assert.equal(drawSeed({ temperature: null, seed: null }, () => 0.5), 2 ** 30)
  assert.equal(drawSeed({ temperature: 0.7 }, () => 1 - Number.EPSILON), 2 ** 31 - 1)
  assert.equal(drawSeed({}, () => 0), 0)
})

test("the preset's own seed, or a greedy reply, draws none", () => {
  assert.equal(drawSeed({ temperature: 1, seed: 3 }), null)
  assert.equal(drawSeed({ temperature: 0, seed: null }), null)
})
