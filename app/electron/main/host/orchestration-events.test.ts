// Tests for the additive orchestration event broker (orchestration-events.ts): the tiny
// in-process pub/sub the supervisors push lifecycle events through, plus the pure event
// shapers. Covers subscribe/emit/unsubscribe with no listener leak, faulty-subscriber
// isolation, the shapers dropping undefined optionals, and the window bridge forwarding.
//
// Pure node + assert (matches loop-host.test.ts); no Electron, no disk.

import assert from 'node:assert/strict'
import {
  OrchestrationEventBroker,
  orchestrationEvents,
  missionLifecycleEvent,
  roomLifecycleEvent,
  jobLifecycleEvent,
  emitMissionLifecycle,
  emitJobLifecycle,
  wireOrchestrationEventsToWindow,
  type OrchestrationEvent,
} from './orchestration-events'

async function run() {
  // 1) subscribe -> emit delivers to the listener; unsubscribe stops further delivery
  //    AND removes the listener (no leak: size returns to 0).
  {
    const broker = new OrchestrationEventBroker()
    const got: OrchestrationEvent[] = []
    const unsub = broker.subscribe((e) => got.push(e))
    assert.equal(broker.size, 1, 'one subscriber after subscribe')

    broker.emit(missionLifecycleEvent('started', 'm1', { runIndex: 1 }))
    assert.equal(got.length, 1, 'subscriber receives the emitted event')
    assert.equal(got[0].scope, 'mission')

    unsub()
    assert.equal(broker.size, 0, 'unsubscribe removes the listener (no leak)')
    broker.emit(missionLifecycleEvent('finalized', 'm1'))
    assert.equal(got.length, 1, 'no delivery after unsubscribe')
  }

  // 2) unsubscribe is idempotent: calling it twice does not throw and does not corrupt
  //    the set (a second unsub of the same listener must not drop a different one).
  {
    const broker = new OrchestrationEventBroker()
    const a: OrchestrationEvent[] = []
    const b: OrchestrationEvent[] = []
    const unsubA = broker.subscribe((e) => a.push(e))
    broker.subscribe((e) => b.push(e))
    unsubA()
    unsubA() // second call is a no-op
    broker.emit(jobLifecycleEvent('j1', 'running', 'mission'))
    assert.equal(a.length, 0, 'unsubscribed listener gets nothing')
    assert.equal(b.length, 1, 'the still-subscribed listener is unaffected by the double unsub')
    assert.equal(broker.size, 1, 'exactly one listener remains')
  }

  // 3) Multiple subscribers all receive every event; clear() drops them all.
  {
    const broker = new OrchestrationEventBroker()
    let c1 = 0
    let c2 = 0
    broker.subscribe(() => { c1++ })
    broker.subscribe(() => { c2++ })
    broker.emit(roomLifecycleEvent('turn', 'r1', { seat: 'Ada' }))
    assert.equal(c1, 1)
    assert.equal(c2, 1)
    broker.clear()
    assert.equal(broker.size, 0, 'clear() removes every subscriber')
    broker.emit(roomLifecycleEvent('finalized', 'r1'))
    assert.equal(c1, 1, 'no delivery after clear')
  }

  // 4) A faulty subscriber is isolated: its throw is swallowed and does NOT stop the
  //    remaining subscribers from receiving the event (a bad listener can't break a run).
  {
    const broker = new OrchestrationEventBroker()
    const delivered: string[] = []
    broker.subscribe(() => { throw new Error('boom') })
    broker.subscribe((e) => { delivered.push(e.scope) })
    assert.doesNotThrow(() => broker.emit(missionLifecycleEvent('created', 'm1', { status: 'queued' })), 'emit must not propagate a subscriber error')
    assert.deepEqual(delivered, ['mission'], 'the healthy subscriber still received the event')
  }

  // 5) A subscriber that unsubscribes mid-dispatch does not perturb the current emit
  //    (we iterate a snapshot of the listener set).
  {
    const broker = new OrchestrationEventBroker()
    let other = 0
    const unsubSelf = broker.subscribe(() => { unsubSelf() })
    broker.subscribe(() => { other++ })
    assert.doesNotThrow(() => broker.emit(jobLifecycleEvent('j9', 'completed', 'thread')))
    assert.equal(other, 1, 'the other subscriber still ran during this dispatch')
    assert.equal(broker.size, 1, 'the self-removing subscriber is gone after dispatch')
  }

  // 6) emit on an empty broker is a cheap no-op (no throw).
  {
    const broker = new OrchestrationEventBroker()
    assert.doesNotThrow(() => broker.emit(missionLifecycleEvent('started', 'm1')))
  }

  // --- Pure event shapers ---------------------------------------------------

  // 7) missionLifecycleEvent: fixed scope/phase/id + ts; drops undefined optionals and
  //    floors/guards runIndex; truncates a long reason; drops an empty-string reason.
  {
    const full = missionLifecycleEvent('checkpoint', 'm2', {
      status: 'running',
      title: 'Ship it',
      repeatMode: 'until_condition',
      runIndex: 3.9,
      reason: 'pass 3 verified',
      ts: 123,
    })
    assert.deepEqual(full, {
      scope: 'mission',
      phase: 'checkpoint',
      missionId: 'm2',
      status: 'running',
      title: 'Ship it',
      repeatMode: 'until_condition',
      runIndex: 3, // floored
      reason: 'pass 3 verified',
      ts: 123,
    })

    const sparse = missionLifecycleEvent('started', 'm3', { reason: '', runIndex: Number.NaN, ts: 1 })
    assert.deepEqual(sparse, { scope: 'mission', phase: 'started', missionId: 'm3', ts: 1 }, 'undefined/empty/NaN optionals are omitted')
    assert.ok(!('reason' in sparse) && !('runIndex' in sparse) && !('status' in sparse))

    const long = missionLifecycleEvent('finalized', 'm4', { reason: 'x'.repeat(900), ts: 1 })
    assert.equal(long.reason?.length, 400, 'reason is truncated to 400 chars')

    const noTs = missionLifecycleEvent('created', 'm5')
    assert.equal(typeof noTs.ts, 'number', 'ts defaults to a timestamp when omitted')
  }

  // 8) roomLifecycleEvent: drops empty seat; keeps state.
  {
    assert.deepEqual(roomLifecycleEvent('turn', 'r2', { state: 'running', seat: 'Ada', ts: 5 }), {
      scope: 'room', phase: 'turn', roomId: 'r2', state: 'running', seat: 'Ada', ts: 5,
    })
    assert.deepEqual(roomLifecycleEvent('finalized', 'r3', { state: 'complete', seat: '', ts: 5 }), {
      scope: 'room', phase: 'finalized', roomId: 'r3', state: 'complete', ts: 5,
    }, 'empty seat omitted')
  }

  // 9) jobLifecycleEvent: required status/kind always present; null owner ids kept
  //    (null is meaningful — "not owned"), undefined owner ids omitted.
  {
    assert.deepEqual(jobLifecycleEvent('j2', 'waiting_approval', 'mission', { missionId: 'm6', roomId: null, ts: 7 }), {
      scope: 'job', phase: 'status_changed', jobId: 'j2', status: 'waiting_approval', kind: 'mission', missionId: 'm6', roomId: null, ts: 7,
    })
    const minimal = jobLifecycleEvent('j3', 'queued', 'thread', { ts: 7 })
    assert.deepEqual(minimal, { scope: 'job', phase: 'status_changed', jobId: 'j3', status: 'queued', kind: 'thread', ts: 7 }, 'undefined owner ids omitted')
  }

  // 10) Convenience emitters guard empty ids (no event for ''), and push through the
  //     singleton broker.
  {
    const seen: OrchestrationEvent[] = []
    const unsub = orchestrationEvents.subscribe((e) => seen.push(e))
    try {
      emitMissionLifecycle('started', '', { runIndex: 1 }) // empty id -> guarded, no emit
      assert.equal(seen.length, 0, 'empty mission id emits nothing')
      emitMissionLifecycle('started', 'm7', { runIndex: 2 })
      emitJobLifecycle('j4', 'running', 'room', { roomId: 'r4' })
      assert.equal(seen.length, 2, 'non-empty ids emit through the singleton')
      assert.equal(seen[0].scope, 'mission')
      assert.equal(seen[1].scope, 'job')
    } finally {
      unsub()
    }
    assert.equal(orchestrationEvents.size, 0, 'singleton has no leaked subscribers after the test')
  }

  // 11) wireOrchestrationEventsToWindow forwards each broker event to the send thunk and
  //     returns a working unsubscribe; a throwing send is swallowed (a destroyed
  //     webContents must not break a live run).
  {
    const broker = new OrchestrationEventBroker()
    const sent: OrchestrationEvent[] = []
    const unwire = wireOrchestrationEventsToWindow((e) => sent.push(e), broker)
    broker.emit(missionLifecycleEvent('created', 'm8', { status: 'queued' }))
    assert.equal(sent.length, 1, 'bridge forwards the event to the window send')

    const broker2 = new OrchestrationEventBroker()
    const unwire2 = wireOrchestrationEventsToWindow(() => { throw new Error('window gone') }, broker2)
    assert.doesNotThrow(() => broker2.emit(roomLifecycleEvent('started', 'r5')), 'a throwing send is isolated')
    unwire2()

    unwire()
    broker.emit(missionLifecycleEvent('finalized', 'm8'))
    assert.equal(sent.length, 1, 'no forwarding after the bridge is unwired')
    assert.equal(broker.size, 0, 'unwire removes the bridge subscriber (no leak)')
  }

  console.log('PASS orchestration-events: subscribe/emit/unsubscribe (no leak), faulty-subscriber isolation, pure shapers, window bridge')
}

run().catch((e) => { console.error(e); process.exit(1) })
