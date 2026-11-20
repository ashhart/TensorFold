import assert from 'node:assert/strict'
import { PullInactivity } from './pull-inactivity'
import { PullClock } from './test-fixtures/pull-clock'

let expired = 0
const clock = new PullClock()
const guard = new PullInactivity(() => { expired++ }, clock.schedule)
guard.start(4096)
clock.advance(59_999)
assert.equal(expired, 0, 'a resumed cache baseline does not expire before 60 seconds')
guard.sample(4096)
guard.sample(0)
clock.advance(1)
assert.equal(expired, 1, 'unchanged or falling byte totals do not extend the deadline')
guard.start(4096)
clock.advance(59_999)
guard.sample(8192)
clock.advance(1)
assert.equal(expired, 1, 'new bytes grant a fresh 60 seconds')
clock.advance(59_999)
assert.equal(expired, 2)
guard.start(8192)
guard.stop()
clock.advance(60_000)
assert.equal(expired, 2, 'terminal cleanup cancels the deadline')
assert.equal(clock.pending, 0)
console.log('4/4 pull inactivity checks passed: exact boundary, resumed baseline, growth reset and cleanup')
