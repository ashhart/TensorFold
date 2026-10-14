import assert from 'node:assert/strict'
import {
  addMissionCheckpoint,
  addMissionScratchMemory,
  buildMissionInspectionSnapshot,
  completeMission,
  createMission,
  forgetMissionScratchMemory,
  formatMissionScratchMemory,
  inspectMissionById,
  normalizeMission,
  pauseMission,
  requestMissionApproval,
  resolveMissionApproval,
  startMission,
} from './missions.ts'
import type { Mission } from './types'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

function baseMission(now = 1_000): Mission {
  return createMission({
    title: 'Research sources',
    objective: 'Collect source-backed findings.',
    stopCondition: 'Three cited findings are ready.',
    repeatMode: 'until_condition',
    contextMode: 'build_on_previous',
    runBudgetClass: 'mission',
  }, now)
}

console.log('missions tests:')

{
  const mission = baseMission()
  assert.deepEqual(mission.scratchMemory, [])
  ok('creates missions with a separate empty scratch-memory lane')
}

{
  const mission = addMissionScratchMemory(baseMission(), {
    kind: 'finding',
    text: '  Candidate source list belongs to this mission only.  ',
    source: 'agent',
    runIndex: 2,
    checkpointId: 'checkpoint-2',
  }, 2_000)
  assert.equal(mission.scratchMemory.length, 1)
  assert.equal(mission.scratchMemory[0].kind, 'finding')
  assert.equal(mission.scratchMemory[0].text, 'Candidate source list belongs to this mission only.')
  assert.equal(mission.scratchMemory[0].runIndex, 2)
  assert.equal(mission.scratchMemory[0].checkpointId, 'checkpoint-2')
  assert.equal(mission.projectId, null)
  assert.ok(mission.journal.some((event) => event.data?.scratchMemoryId === mission.scratchMemory[0].id))
  ok('adds scratch memory to the mission record without using project memory')
}

{
  const mission = addMissionScratchMemory(baseMission(), { text: 'Temporary verifier note.' }, 2_000)
  const id = mission.scratchMemory[0].id
  const forgotten = forgetMissionScratchMemory(mission, id, 2_500)
  assert.deepEqual(forgotten.scratchMemory, [])
  assert.ok(forgotten.journal.some((event) => event.data?.scratchMemoryId === id))
  ok('forgets mission scratch memory independently of project memory')
}

{
  const mission = addMissionScratchMemory(baseMission(), { kind: 'decision', text: 'Use the archived brief as evidence.', source: 'user', runIndex: 1 }, 2_000)
  const formatted = formatMissionScratchMemory(mission)
  assert.match(formatted, /\[decision; user run 1\] Use the archived brief as evidence\./)
  ok('formats mission scratch memory for later mission passes')
}

{
  const normalized = normalizeMission({
    ...baseMission(),
    scratchMemory: [
      { id: '', kind: 'finding', text: '  Keep this note.  ', source: 'agent', runIndex: 1.8, checkpointId: undefined as any, createdAt: 0, updatedAt: 0 },
      { id: 'blank', kind: 'note', text: '   ', source: 'agent', runIndex: null, checkpointId: null, createdAt: 0, updatedAt: 0 },
    ],
  } as Mission, 3_000)
  assert.equal(normalized.scratchMemory.length, 1)
  assert.equal(normalized.scratchMemory[0].text, 'Keep this note.')
  assert.equal(normalized.scratchMemory[0].runIndex, 1)
  assert.equal(normalized.scratchMemory[0].checkpointId, null)
  assert.ok(normalized.scratchMemory[0].id)
  ok('normalizes legacy or partial mission scratch memory safely')
}

{
  let mission = baseMission()
  mission = addMissionCheckpoint(mission, {
    label: 'Evidence collected',
    summary: 'Found the archived verifier output.',
    status: 'running',
  }, 2_000)
  mission = addMissionScratchMemory(mission, { kind: 'decision', text: 'Use the archived verifier output.', runIndex: 1 }, 2_100)
  mission = completeMission(mission, 'Mission finished with required evidence.', 3_000)
  const snapshot = buildMissionInspectionSnapshot(mission, 4_000)
  assert.equal(snapshot.generatedAt, 4_000)
  assert.equal(snapshot.mission.status, 'completed')
  assert.equal(snapshot.terminal, true)
  assert.equal(snapshot.active, false)
  assert.equal(snapshot.rerunRequired, false)
  assert.equal(snapshot.checkpointCount, 1)
  assert.ok(snapshot.journalCount >= 4)
  assert.equal(snapshot.scratchMemoryCount, 1)
  assert.equal(snapshot.pendingApprovalCount, 0)
  assert.equal(snapshot.latestCheckpointId, mission.checkpoints[0].id)
  assert.equal(snapshot.latestJournalId, mission.journal.at(-1)?.id)
  assert.equal(snapshot.mission.finishedAt, 3_000)
  ok('builds a read-only inspection snapshot for completed missions without rerunning')
}

{
  const completed = completeMission(baseMission(), 'Done.', 2_000)
  const snapshot = inspectMissionById([baseMission(), completed], completed.id, 3_000)
  assert.equal(snapshot.mission.id, completed.id)
  assert.equal(snapshot.mission.status, 'completed')
  assert.equal(snapshot.rerunRequired, false)
  assert.throws(() => inspectMissionById([], completed.id), /Mission not found/)
  ok('opens a completed mission from stored mission records by id')
}

{
  // A mission stalled (by the heartbeat check) WHILE waiting for approval must resume when the
  // approval is resolved — not stay permanently stalled.
  const waiting = requestMissionApproval(startMission(baseMission(1_000), 2_000), { callId: 'c1', title: 'Approve deploy', summary: 'Deploy to prod?' }, 3_000)
  assert.equal(waiting.status, 'waiting_approval')
  const stalled: Mission = { ...waiting, status: 'stalled' } // simulate evaluateMissionHeartbeat flipping it
  const resolved = resolveMissionApproval(stalled, waiting.pendingApprovals[0].id, 'allow', 4_000)
  assert.equal(resolved.status, 'running', 'resolving the last approval resumes a stalled mission')
  assert.equal(resolved.pendingApprovals.length, 0)
  ok('resolving an approval resumes a mission that stalled while waiting')
}

{
  // A user-paused mission must not be silently re-activated into waiting_approval by
  // an in-flight or stale approval request — paused -> waiting_approval is illegal.
  const paused = pauseMission(startMission(baseMission(1_000), 2_000), 'Paused by user.', 3_000)
  assert.equal(paused.status, 'paused')
  const after = requestMissionApproval(paused, { callId: 'c1', title: 'Approve deploy', summary: 'Deploy?' }, 4_000)
  assert.equal(after.status, 'paused', 'requestMissionApproval is a no-op on a paused mission')
  assert.equal(after.pendingApprovals.length, 0, 'no approval is parked on a paused mission')
  assert.equal(after, paused, 'the paused mission is returned unchanged (no journal append)')
  ok('requestMissionApproval refuses the illegal paused -> waiting_approval transition')
}

{
  // The legal running -> waiting_approval path is unaffected.
  const running = startMission(baseMission(1_000), 2_000)
  assert.equal(running.status, 'running')
  const waiting = requestMissionApproval(running, { callId: 'c2', title: 'Approve', summary: 'OK?' }, 3_000)
  assert.equal(waiting.status, 'waiting_approval')
  assert.equal(waiting.pendingApprovals.length, 1)
  ok('requestMissionApproval still allows the legal running -> waiting_approval transition')
}

console.log(`\n${passed} checks passed.`)
