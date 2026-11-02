import assert from 'node:assert/strict'
import { DesktopPowerBlocker } from './power-blocker'
const live = new Set<number>(), starts: string[] = [], stops: number[] = []
let next = 0
const blocker = new DesktopPowerBlocker({ start: type => { starts.push(type); live.add(++next); return next },
  isStarted: id => live.has(id), stop: id => { stops.push(id); live.delete(id) } })
blocker.start(); blocker.start()
assert.equal(starts.length, 1, 'reapplying a preference cannot leak an inhibitor')
live.clear(); blocker.start(); assert.equal(starts.length, 2, 'an inhibitor lost after session changes is reacquired')
blocker.stop(); blocker.stop(); assert.deepEqual(stops, [2])
const refused = new DesktopPowerBlocker({ start: () => 1, isStarted: () => false, stop: () => assert.fail('No active inhibitor') })
assert.throws(() => refused.start(), /refused/); refused.stop()
console.log('Desktop power blocker start, idempotence, session recovery, refusal and cleanup passed.')
