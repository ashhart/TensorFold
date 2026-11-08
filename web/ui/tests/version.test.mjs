import assert from 'node:assert/strict'
import { test } from 'node:test'
import { before, engineSkew, displayEngineVersion } from '../src/version.ts'

test('releases compare as numbers, not as text', () => {
  assert.equal(before('1.0.9', [1, 0, 11]), true)          // as text, "1.0.9" would come after "1.0.11"
  assert.equal(before('1.0.11', [1, 0, 11]), false)
  assert.equal(before('1.0.12', [1, 0, 11]), false)
  assert.equal(before('1.1.0', [1, 0, 11]), false)
  assert.equal(before('0.6.6 (Python)', [1, 0, 11]), true)
  assert.equal(before(null, [1, 0, 11]), false)             // nothing to compare: no warning
  assert.equal(before('', [1, 0, 11]), false)
})

test('two native releases differ; an unknown one or the Python line does not count', () => {
  assert.equal(engineSkew('1.0.11', '1.0.9'), true)
  assert.equal(engineSkew('1.0.11', '1.0.11'), false)
  assert.equal(engineSkew('1.0.11', null), false)
  assert.equal(engineSkew(undefined, '1.0.9'), false)
  assert.equal(engineSkew('1.0.11', '0.6.6 (Python)'), false)
})

test('machine and flags labels use the same authoritative engine version', () => {
  assert.equal(displayEngineVersion('1.2.4', '1.1.6'), '1.2.4')
  assert.equal(displayEngineVersion('0.6.6 (Python)', null), '0.6.6 (Python)')
  assert.equal(displayEngineVersion(null, '1.1.6'), '1.1.6')
  assert.equal(displayEngineVersion(undefined, undefined), 'not installed')
})
