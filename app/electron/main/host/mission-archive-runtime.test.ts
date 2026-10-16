import assert from 'node:assert/strict'
import { abortMissionsForArchive, missionIdsForArchive } from './mission-runtime'
import { createJobControlRuntime } from './job-control-runtime'
import { store } from '../state'
import { createMission } from '../../agent/missions'

const state = {
  missions: [
    { id: 'direct-project', projectId: 'project-a', threadId: null },
    { id: 'derived-project', projectId: null, threadId: 'thread-a' },
    { id: 'direct-thread', projectId: 'project-b', threadId: 'thread-target' },
    { id: 'both', projectId: 'project-a', threadId: 'thread-a' },
    { id: 'other', projectId: 'project-b', threadId: 'thread-b' },
  ],
  threads: [
    { id: 'thread-a', projectId: 'project-a' },
    { id: 'thread-target', projectId: 'project-b' },
    { id: 'thread-b', projectId: 'project-b' },
  ],
}

assert.deepEqual(
  missionIdsForArchive(state, { projectId: 'project-a' }).sort(),
  ['both', 'derived-project', 'direct-project'],
  'project archive includes missions whose ownership is derived from their source thread',
)
assert.deepEqual(
  missionIdsForArchive(state, { threadId: 'thread-target' }),
  ['direct-thread'],
  'thread archive identifies its directly linked mission',
)
assert.deepEqual(missionIdsForArchive(state, {}), [], 'an empty archive target cannot match all missions')

const controllers = new Map<string, AbortController>()
for (const mission of state.missions) controllers.set(mission.id, new AbortController())
controllers.get('derived-project')!.abort()
controllers.delete('direct-project') // persisted but not currently active

assert.deepEqual(
  abortMissionsForArchive(state, controllers, { projectId: 'project-a' }),
  ['both'],
  'only a live, not-already-aborted controller is newly aborted',
)
assert.equal(controllers.get('both')!.signal.aborted, true)
assert.equal(controllers.get('other')!.signal.aborted, false, 'unrelated project mission remains active')

const savedProjects = store.projects
const savedThreads = store.threads
const previousRole = process.env.VONTRA_SECURITY_ROLE
try {
  process.env.VONTRA_SECURITY_ROLE = 'operator'
  store.projects = [{ id: 'archived-project', archivedAt: Date.now() } as any]
  store.threads = []
  const mission = { id: 'blocked-start', projectId: 'archived-project', threadId: null, status: 'paused' } as any
  let replacements = 0
  let launches = 0
  const runtime = createJobControlRuntime({
    handle: () => {},
    getMission: () => mission,
    replaceMission: (next: any) => { replacements++; Object.assign(mission, next); return mission },
    runMission: async () => { launches++ },
  } as any)
  await assert.rejects(() => runtime.setMissionStatusLocal(mission.id, 'running'), /Restore the archived project/)
  assert.equal(mission.status, 'paused', 'failed preflight leaves the durable mission non-running')
  assert.equal(replacements, 0, 'failed preflight occurs before startMission persistence')
  assert.equal(launches, 0, 'failed preflight launches no background promise')

  store.projects = [{ id: 'local-project' } as any]
  let launchFailureMission = createMission({ title: 'Launch', objective: 'Test', projectId: 'local-project' })
  let finished = 0
  const failingRuntime = createJobControlRuntime({
    handle: () => {},
    getMission: () => launchFailureMission,
    replaceMission: (next: any) => { launchFailureMission = next; return next },
    runMission: async () => { throw new Error('injected launcher failure') },
    activeJob: () => null,
    finishJob: () => { finished++ },
  } as any)
  await failingRuntime.setMissionStatusLocal(launchFailureMission.id, 'running')
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(launchFailureMission.status, 'failed', 'unexpected background rejection settles the mission instead of leaving it running')
  assert.match(launchFailureMission.journal.at(-1)?.message ?? '', /injected launcher failure/)
  assert.equal(finished, 1, 'background launch rejection settles the linked job path')
} finally {
  store.projects = savedProjects
  store.threads = savedThreads
  if (previousRole == null) delete process.env.VONTRA_SECURITY_ROLE
  else process.env.VONTRA_SECURITY_ROLE = previousRole
}

console.log('mission-archive-runtime.test: archive ownership projection and active abort passed')
