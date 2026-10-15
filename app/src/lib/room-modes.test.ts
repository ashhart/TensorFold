import assert from 'node:assert/strict'
import { ROOM_MODE_OPTIONS, roomModeDefaultEffort } from './room-modes.ts'

const expected = {
  plan: 'normal',
  implement: 'deep',
  review: 'normal',
  debug: 'deep',
  incident: 'max',
  research: 'normal',
} as const

assert.deepEqual(
  Object.fromEntries(ROOM_MODE_OPTIONS.map((option) => [option.id, option.budget])),
  expected,
  'renderer mode budgets stay aligned with the enforced main-process outcome profiles',
)
for (const [mode, budget] of Object.entries(expected)) {
  assert.equal(roomModeDefaultEffort(mode as keyof typeof expected), budget)
}

console.log('room mode defaults: every mode selects its real deliberation budget')
