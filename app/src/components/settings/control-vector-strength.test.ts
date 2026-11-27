import assert from 'node:assert/strict'
import { test } from 'node:test'
import { strengthOnRefusal } from './control-vector-strength'

test('a refused strength change shows the snapshot, not the attempt', () => {
  // the engine keeps alpha 2 and refuses 1.5; the field must read 2, as the preferences and arguments do
  assert.deepEqual(strengthOnRefusal({ alpha: 2 }, 'the engine refused the strength'), { shown: '2', error: 'the engine refused the strength' })
})

test('a refused change with no snapshot clears the field', () => {
  assert.deepEqual(strengthOnRefusal(null, 'no vector'), { shown: '', error: 'no vector' })
})
