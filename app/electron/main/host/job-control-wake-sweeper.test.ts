// Tests for the live-session room wake sweeper (sweepDueRoomWakes): a parked room
// whose nextWakeAt has passed is woken exactly once, a running room is skipped, and
// future wakes are left alone.
import assert from 'node:assert/strict'
import { store } from '../state'
import { createJobControlRuntime } from './job-control-runtime'

const saved = {
  jobs: store.jobs,
  rooms: store.rooms,
  tickets: store.tickets,
  projects: store.projects,
}

const roomJob = (id: string, roomId: string, status: string, nextWakeAt: number | null): any => ({
  id, kind: 'room', owner: roomId, roomId, status, nextWakeAt, pendingApprovals: [],
})

function makeRuntime(starts: string[]) {
  return createJobControlRuntime({
    handle: () => {},
    patchJob: (job: any, patch: any) => {
      const next = { ...job, ...patch }
      store.jobs = store.jobs.map((j) => (j.id === job.id ? next : j))
      return next
    },
    persistJobs: () => {},
    emitJob: () => {},
    startRoomRun: async (roomId: string) => { starts.push(roomId); return { ok: true } },
  } as any)
}

try {
  store.rooms = [{ id: 'room-1', projectId: 'p1', problem: 'goal' } as any]
  store.projects = [{ id: 'p1' } as any]
  store.tickets = []

  // 1) PARKED + PAST DUE — woken, wake marker cleared, and never double-woken.
  {
    const starts: string[] = []
    const runtime = makeRuntime(starts)
    store.jobs = [roomJob('job-1', 'room-1', 'paused', Date.now() - 1_000)]

    await runtime.sweepDueRoomWakes()
    assert.deepEqual(starts, ['room-1'], 'past-due parked room was woken')
    assert.equal(store.jobs[0].nextWakeAt, null, 'wake marker cleared on wake')

    await runtime.sweepDueRoomWakes()
    assert.deepEqual(starts, ['room-1'], 'a cleared wake does not fire again')
    console.log('ok - a parked room with a past nextWakeAt is woken exactly once')
  }

  // 2) RUNNING ROOM — skipped (its job is not paused).
  {
    const starts: string[] = []
    const runtime = makeRuntime(starts)
    store.jobs = [roomJob('job-2', 'room-1', 'running', Date.now() - 1_000)]

    await runtime.sweepDueRoomWakes()
    assert.deepEqual(starts, [], 'a running room is never re-woken')
    console.log('ok - a running room is skipped')
  }

  // 3) NOT YET DUE / NON-ROOM / UNRESOLVED APPROVAL — all left alone.
  {
    const starts: string[] = []
    const runtime = makeRuntime(starts)
    store.jobs = [
      roomJob('job-3', 'room-1', 'paused', Date.now() + 60_000),
      { id: 'job-4', kind: 'thread', owner: 't1', threadId: 't1', status: 'paused', nextWakeAt: Date.now() - 1_000, pendingApprovals: [] } as any,
      { ...roomJob('job-5', 'room-1', 'paused', Date.now() - 1_000), pendingApprovals: [{ callId: 'c1', resolvedAt: null }] },
    ]

    await runtime.sweepDueRoomWakes()
    assert.deepEqual(starts, [], 'future wakes, non-room jobs, and approval-blocked jobs are skipped')
    console.log('ok - non-due, non-room, and approval-blocked jobs are skipped')
  }

  // 4) QUEUED WITH DUE WAKE — jobs:retry leaves a room job 'queued' with a nextWakeAt;
  //    the sweeper must fire those too (previously they never fired in-session).
  {
    const starts: string[] = []
    const runtime = makeRuntime(starts)
    store.jobs = [roomJob('job-6', 'room-1', 'queued', Date.now() - 1_000)]

    await runtime.sweepDueRoomWakes()
    assert.deepEqual(starts, ['room-1'], 'a queued room job with a due wake fires')
    assert.equal(store.jobs[0].nextWakeAt, null, 'wake marker cleared on wake')
    console.log('ok - a queued room job with a due nextWakeAt is woken')
  }

  // 5) RE-ENTRY GUARD — a second startRoomWakeSweeper call must not stack intervals.
  {
    const runtime = makeRuntime([])
    let intervalCount = 0
    const realSetInterval = globalThis.setInterval
    globalThis.setInterval = ((handler: any, timeout?: any) => { intervalCount++; return realSetInterval(handler, timeout) }) as any
    try {
      runtime.startRoomWakeSweeper()
      runtime.startRoomWakeSweeper()
    } finally {
      globalThis.setInterval = realSetInterval
    }
    assert.equal(intervalCount, 1, 'the sweeper interval is only installed once')
    console.log('ok - startRoomWakeSweeper is re-entry safe')
  }
} finally {
  store.jobs = saved.jobs
  store.rooms = saved.rooms
  store.tickets = saved.tickets
  store.projects = saved.projects
}

console.log('job control wake sweeper tests passed')
