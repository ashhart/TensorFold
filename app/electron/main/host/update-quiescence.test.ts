import assert from 'node:assert/strict'
import { assertUpdateQuiescent, type UpdateQuiescence } from './update-quiescence'

const idle: UpdateQuiescence = { ready: true, threadControllers: 0, roomControllers: 0, missionControllers: 0, runningMissions: 0, pendingApprovals: 0, runningThreads: 0, activeJobs: 0, runningAutomations: 0, liveTerminals: 0, mobileOperations: 0, engineRequests: 0, modelLoading: false, modelDownloads: 0 }
assert.doesNotThrow(() => assertUpdateQuiescent(idle))
for (const key of ['threadControllers', 'roomControllers', 'missionControllers', 'runningMissions', 'pendingApprovals', 'runningThreads', 'activeJobs', 'runningAutomations', 'liveTerminals', 'mobileOperations', 'engineRequests', 'modelDownloads'] as const) {
  assert.throws(() => assertUpdateQuiescent({ ...idle, [key]: 1 }), /Finish or stop/, key)
  assert.throws(() => assertUpdateQuiescent({ ...idle, [key]: NaN }), /Finish or stop/, `${key} unknown cannot prove idle`)
}
assert.throws(() => assertUpdateQuiescent({ ...idle, ready: false }), /finish starting/)
assert.throws(() => assertUpdateQuiescent({ ...idle, modelLoading: true }), /Finish or stop/)
console.log('Update restart requires every local work, approval, download and terminal signal to be idle')
