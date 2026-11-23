import assert from 'node:assert/strict'
import { pickRoomConfigPatch } from './room-ticket-patch-policy'

const seat = { id: 'reviewer', name: 'Reviewer', role: 'Review supplied input', providerId: 'fixture', model: 'fixture' }
for (const effort of ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultrathink', 'ultracode', 'dynamic', '512']) {
  const seats = [{ ...seat, reasoningEffort: effort }]
  const patch = pickRoomConfigPatch({ seats })
  assert.deepEqual(patch.seats, seats, 'a valid per-agent effort must survive the real room patch boundary')
  assert.deepEqual(Object.keys(patch), ['seats'])
  assert.equal(Object.getPrototypeOf(patch), null, 'the existing patch isolation boundary remains intact')
}
assert.deepEqual(pickRoomConfigPatch({ seats: [seat] }).seats, [seat], 'legacy agents keep their inherited app setting')
for (const effort of [null, '', ' ', 'HIGH', 'ultrathinks', 4, true]) {
  assert.throws(() => pickRoomConfigPatch({ seats: [{ ...seat, reasoningEffort: effort }] }), /reasoningEffort/)
}
console.log('Room agent effort policy checks passed: supported saved values, legacy absence and rejected malformed values.')
