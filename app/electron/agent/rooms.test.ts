// Run: ./node_modules/.bin/jiti electron/agent/rooms.test.ts
// BUG #53: per-seat heartbeat + stuck-planner stall detection for rooms.
import assert from 'node:assert/strict'
import {
  evaluateSeatHeartbeats,
  initSeatHeartbeat,
  rearmSeatHeartbeats,
  recordSeatBeat,
  seatHeartbeatIntervalSeconds,
  stalledSeatNames,
  type SeatRuntimeLike,
} from './rooms'
import type { Seat } from './types'

const seat = (over: Partial<Seat> = {}): Seat => ({
  id: 'seat-1',
  name: 'Planner',
  role: 'Plan',
  providerId: 'provider-1',
  model: 'model-1',
  ...over,
})

let passed = 0
const ok = (msg: string) => { passed++; console.log(`ok - ${msg}`) }

// 1: interval derived from deliberation effort (recommended defaults)
{
  assert.equal(seatHeartbeatIntervalSeconds('quick'), 30)
  assert.equal(seatHeartbeatIntervalSeconds('normal'), 60)
  assert.equal(seatHeartbeatIntervalSeconds('deep'), 90)
  assert.equal(seatHeartbeatIntervalSeconds('max'), 120)
  // unknown/undefined falls back to the normal cadence
  assert.equal(seatHeartbeatIntervalSeconds(undefined), 60)
  assert.equal(seatHeartbeatIntervalSeconds(null), 60)
  ok('seatHeartbeatIntervalSeconds maps deliberation effort to a cadence')
}

// 2: heartbeat initialization
{
  const now = 1_000_000
  const hb = initSeatHeartbeat(seat(), 60, now)
  assert.equal(hb.seatId, 'seat-1')
  assert.equal(hb.seatName, 'Planner')
  assert.equal(hb.intervalSeconds, 60)
  assert.equal(hb.stallGraceSeconds, 30)
  assert.equal(hb.required, true)
  assert.equal(hb.count, 0)
  assert.equal(hb.lastBeatAt, now)
  assert.equal(hb.nextDueAt, now + 60_000)
  assert.equal(hb.missedSince, null)
  ok('initSeatHeartbeat seeds count/lastBeatAt/nextDueAt and clears missedSince')
}

// 3: heartbeat increments and clears missedSince on a turn
{
  const start = 2_000_000
  const hb = initSeatHeartbeat(seat(), 60, start)
  hb.missedSince = start - 5_000 // pretend it had been stalled
  const beatAt = start + 70_000
  recordSeatBeat(hb, beatAt)
  assert.equal(hb.count, 1)
  assert.equal(hb.lastBeatAt, beatAt)
  assert.equal(hb.nextDueAt, beatAt + 60_000)
  assert.equal(hb.missedSince, null)

  const beat2 = beatAt + 40_000
  recordSeatBeat(hb, beat2)
  assert.equal(hb.count, 2)
  assert.ok(hb.lastBeatAt! > beatAt) // monotonic
  ok('recordSeatBeat increments count, advances lastBeatAt, clears stall flag')
}

// 4: evaluateSeatHeartbeats detects a stall after the grace period
{
  const now = 3_000_000
  const hb = initSeatHeartbeat(seat({ name: 'Builder' }), 60, now - 120_000)
  hb.count = 1
  hb.lastBeatAt = now - 120_000 // 2 min ago
  hb.nextDueAt = now - 60_000 // due 1 min ago (interval elapsed)
  hb.missedSince = null
  const runtimes: SeatRuntimeLike[] = [{ seat: seat({ name: 'Builder' }), heartbeat: hb }]
  const stalled = evaluateSeatHeartbeats(runtimes, now)
  assert.equal(stalled.length, 1)
  assert.ok(stalled[0].includes('Builder'))
  assert.equal(hb.missedSince, hb.nextDueAt) // recorded at the due time
  ok('evaluateSeatHeartbeats marks a seat stalled after grace expires')
}

// 5: within the grace window no stall is reported
{
  const now = 4_000_000
  const hb = initSeatHeartbeat(seat(), 60, now - 70_000)
  hb.count = 1
  hb.lastBeatAt = now - 70_000
  hb.nextDueAt = now - 10_000 // 10s overdue, grace is 30s
  hb.missedSince = null
  const stalled = evaluateSeatHeartbeats([{ seat: seat(), heartbeat: hb }], now)
  assert.equal(stalled.length, 0)
  assert.equal(hb.missedSince, null)
  ok('evaluateSeatHeartbeats tolerates delays inside the grace window')
}

// 6: an already-stalled seat is not reported again (no duplicate notices)
{
  const now = 5_000_000
  const hb = initSeatHeartbeat(seat(), 60, now - 200_000)
  hb.count = 1
  hb.lastBeatAt = now - 200_000
  hb.nextDueAt = now - 140_000
  hb.missedSince = now - 110_000 // already flagged
  const stalled = evaluateSeatHeartbeats([{ seat: seat(), heartbeat: hb }], now)
  assert.equal(stalled.length, 0)
  ok('evaluateSeatHeartbeats does not re-report a seat already flagged stalled')
}

// 7: non-required seats are exempt from stall detection
{
  const now = 6_000_000
  const hb = initSeatHeartbeat(seat(), 60, now - 200_000)
  hb.required = false
  hb.lastBeatAt = now - 200_000
  hb.nextDueAt = now - 140_000
  hb.missedSince = null
  const stalled = evaluateSeatHeartbeats([{ seat: seat(), heartbeat: hb }], now)
  assert.equal(stalled.length, 0)
  assert.equal(hb.missedSince, null)
  ok('evaluateSeatHeartbeats skips non-required seats')
}

// 8: recovery — a stalled seat taking a turn clears the flag
{
  const now = 7_000_000
  const hb = initSeatHeartbeat(seat(), 60, now - 200_000)
  hb.missedSince = now - 100_000 // stalled
  recordSeatBeat(hb, now)
  assert.equal(hb.missedSince, null)
  assert.equal(hb.lastBeatAt, now)
  assert.ok(hb.count >= 1)
  ok('recordSeatBeat clears stall when a stalled seat recovers')
}

// 9: stalledSeatNames feeds the FINAL gate — a stalled teammate blocks finalize
{
  const now = 8_000_000
  const builderHb = initSeatHeartbeat(seat({ id: 'seat-b', name: 'Builder' }), 60, now - 200_000)
  builderHb.missedSince = now - 100_000 // stalled
  const plannerHb = initSeatHeartbeat(seat({ id: 'seat-p', name: 'Planner' }), 60, now)
  const runtimes: SeatRuntimeLike[] = [
    { seat: seat({ name: 'Builder' }), heartbeat: builderHb },
    { seat: seat({ name: 'Planner' }), heartbeat: plannerHb },
  ]
  // Planner is the one trying to FINAL, so it is excluded; Builder still blocks.
  const blocking = stalledSeatNames(runtimes, 'Planner')
  assert.deepEqual(blocking, ['Builder'])
  // When no one is stalled, finalize is unobstructed.
  builderHb.missedSince = null
  assert.deepEqual(stalledSeatNames(runtimes, 'Planner'), [])
  ok('stalledSeatNames reports stalled teammates and excludes the finalizing seat')
}

// 10: pass-start re-arm keeps a healthy waiting seat from false-stalling.
// Mirrors runRoom's top-of-pass refresh: a seat that beat last pass but whose
// nextDueAt is now stale (a slow full pass elapsed) must not be flagged when its
// nextDueAt is re-armed forward. Regression for the FINAL-blocking false positive.
{
  const now = 9_000_000
  // Builder beat a while ago; one slow pass pushed wall clock past nextDueAt+grace.
  const builderHb = initSeatHeartbeat(seat({ id: 'seat-b', name: 'Builder' }), 60, now - 200_000)
  builderHb.count = 1
  builderHb.lastBeatAt = now - 200_000
  builderHb.nextDueAt = now - 140_000 // already past nextDueAt + grace
  builderHb.missedSince = null
  const runtimes: SeatRuntimeLike[] = [{ seat: seat({ name: 'Builder' }), heartbeat: builderHb }]

  // Without re-arming it would be flagged stalled (sanity check of the hazard):
  const wouldStall = builderHb.nextDueAt + builderHb.stallGraceSeconds * 1000 <= now
  assert.equal(wouldStall, true)

  // runRoom's top-of-pass re-arm for healthy required seats:
  const passStart = now
  for (const rt of runtimes) {
    const hb = rt.heartbeat
    if (hb.required && hb.missedSince == null) hb.nextDueAt = passStart + hb.intervalSeconds * 1000
  }
  const stalled = evaluateSeatHeartbeats(runtimes, now)
  assert.equal(stalled.length, 0)
  assert.equal(builderHb.missedSince, null)
  // And the healthy seat does not block a teammate's FINAL.
  assert.deepEqual(stalledSeatNames(runtimes, 'Planner'), [])
  ok('top-of-pass re-arm prevents false stall on a healthy waiting seat')
}

// 11: an already-stalled seat is NOT re-armed (genuine stall survives the pass).
{
  const now = 10_000_000
  const hb = initSeatHeartbeat(seat({ name: 'Hung' }), 60, now - 300_000)
  hb.count = 1
  hb.nextDueAt = now - 200_000
  hb.missedSince = now - 170_000 // genuinely flagged stalled earlier
  const runtimes: SeatRuntimeLike[] = [{ seat: seat({ name: 'Hung' }), heartbeat: hb }]
  const before = hb.nextDueAt
  const passStart = now
  for (const rt of runtimes) {
    const h = rt.heartbeat
    if (h.required && h.missedSince == null) h.nextDueAt = passStart + h.intervalSeconds * 1000
  }
  assert.equal(hb.nextDueAt, before) // untouched — still stalled
  assert.deepEqual(stalledSeatNames(runtimes, 'Other'), ['Hung'])
  ok('top-of-pass re-arm leaves a genuinely stalled seat flagged (still blocks FINAL)')
}

// 12: a no-provider seat marked not-required is excluded from stall + FINAL gate.
// Mirrors runRoom's skip path: required=false, nextDueAt=null, missedSince=null.
{
  const now = 11_000_000
  const hb = initSeatHeartbeat(seat({ id: 'seat-np', name: 'NoProvider' }), 60, now - 300_000)
  // Apply the no-provider excuse runRoom performs on first skip:
  hb.required = false
  hb.nextDueAt = null
  hb.missedSince = null
  const runtimes: SeatRuntimeLike[] = [{ seat: seat({ name: 'NoProvider' }), heartbeat: hb }]
  // Even far past any window it is never flagged...
  assert.deepEqual(evaluateSeatHeartbeats(runtimes, now), [])
  assert.equal(hb.missedSince, null)
  // ...and never blocks another seat's FINAL.
  assert.deepEqual(stalledSeatNames(runtimes, 'Planner'), [])
  ok('no-provider seat excused from heartbeat tracking does not block FINAL')
}

// 13: rearmSeatHeartbeats only refreshes seats that actually ran last pass.
// Regression for the dead-code stall detection: the old loop re-armed EVERY
// non-stalled seat at pass start, so a hung seat (never beat) was perpetually
// pushed forward and evaluateSeatHeartbeats could never flag it.
{
  const now = 12_000_000
  // Healthy seat that beat last pass but whose nextDueAt is now stale.
  const healthy = initSeatHeartbeat(seat({ id: 'seat-h', name: 'Healthy' }), 60, now - 200_000)
  healthy.count = 1
  healthy.lastBeatAt = now - 200_000
  healthy.nextDueAt = now - 140_000 // already past nextDueAt + grace
  healthy.missedSince = null
  // Hung seat that has NOT run any pass: window must keep counting.
  const hung = initSeatHeartbeat(seat({ id: 'seat-x', name: 'Hung' }), 60, now - 200_000)
  hung.count = 1
  hung.lastBeatAt = now - 200_000
  hung.nextDueAt = now - 140_000 // already past nextDueAt + grace
  hung.missedSince = null

  const runtimes: SeatRuntimeLike[] = [
    { seat: seat({ id: 'seat-h', name: 'Healthy' }), heartbeat: healthy },
    { seat: seat({ id: 'seat-x', name: 'Hung' }), heartbeat: hung },
  ]
  // Only 'Healthy' ran last pass.
  rearmSeatHeartbeats(runtimes, new Set(['seat-h']), now)

  // Healthy is pushed forward (not stalled); Hung is untouched and still overdue.
  assert.equal(healthy.nextDueAt, now + 60 * 1000)
  assert.equal(hung.nextDueAt, now - 140_000)

  const stalled = evaluateSeatHeartbeats(runtimes, now)
  // The hung seat is flagged; the healthy one is not.
  assert.equal(stalled.length, 1)
  assert.equal(healthy.missedSince, null)
  assert.notEqual(hung.missedSince, null)
  assert.deepEqual(stalledSeatNames(runtimes, 'Healthy'), ['Hung'])
  ok('rearmSeatHeartbeats refreshes only seats that ran last pass; a hung seat stays flaggable')
}

// 14: on the first pass (empty ranLastPass) no seat is re-armed, and a seat that is
// not yet overdue is not falsely flagged — re-arm is a no-op, evaluate is the gate.
{
  const now = 13_000_000
  const hb = initSeatHeartbeat(seat({ id: 'seat-f', name: 'First' }), 60, now)
  const runtimes: SeatRuntimeLike[] = [{ seat: seat({ id: 'seat-f', name: 'First' }), heartbeat: hb }]
  const before = hb.nextDueAt
  rearmSeatHeartbeats(runtimes, new Set<string>(), now)
  assert.equal(hb.nextDueAt, before) // untouched on the first pass
  assert.deepEqual(evaluateSeatHeartbeats(runtimes, now), []) // not overdue yet
  ok('rearmSeatHeartbeats is a no-op on the first pass (empty ranLastPass)')
}

console.log(`\n${passed} room heartbeat assertions passed.`)
