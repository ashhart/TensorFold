import assert from 'node:assert/strict'
import { test } from 'node:test'
import { bytes, clock, count, secs } from '../src/units.ts'

const GiB = 1024 ** 3

test('durations roll over', () => {
  assert.equal(secs(0.07), '70 ms')
  assert.equal(secs(3.14), '3.1 s')
  assert.equal(secs(59.99), '1 min 00 s')
  assert.equal(secs(125), '2 min 05 s')
  assert.equal(secs(5400), '1 h 30 min')
  assert.equal(secs(200000), '2 d 7 h')
  assert.equal(secs(null), '—')
  assert.equal(clock(45), '45 s')
})

test('sizes roll over', () => {
  assert.equal(bytes(0.4 * GiB), '410 MB')
  assert.equal(bytes(17.99 * GiB, 1), '18.0 GB')
  assert.equal(bytes(122 * GiB), '122 GB')
  assert.equal(bytes(2048 * GiB), '2.0 TB')
  assert.equal(bytes(0), '0 GB')
  assert.equal(bytes(undefined), '—')
})

test('counts', () => {
  assert.deepEqual([950, 1200, 12345, 3.4e6, 1.1e9].map(count), ['950', '1.2k', '12k', '3.4M', '1.1B'])
})
