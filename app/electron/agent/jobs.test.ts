// Run: ./node_modules/.bin/jiti electron/agent/jobs.test.ts
import assert from 'node:assert/strict'
import {
  cancelJob,
  buildRoomSupervisorJobMetadata,
  claimJobLease,
  completeJob,
  deadLetterJob,
  detectStuckJobs,
  enqueueJob,
  failJob,
  heartbeatJobLease,
  markRemoteJobLive,
  markRemoteJobsUnverifiable,
  normalizeJobs,
  recordJobApproval,
  reconcileJobsAfterRestart,
  resolveJobApproval,
  restartPolicyAutoResumes,
  restartResumeApprovalMode,
  retryJob,
  resumeJob,
  updateJob,
} from './jobs'

{
  const first = enqueueJob([], { kind: 'mission', owner: 'mission-1', missionId: 'mission-1', projectId: 'project-1' }, 100)
  assert.equal(first.created, true)
  assert.equal(first.job.status, 'queued')
  assert.equal(first.job.missionId, 'mission-1')

  const second = enqueueJob(first.jobs, { kind: 'mission', owner: 'mission-1', missionId: 'mission-1', projectId: 'project-1' }, 120)
  assert.equal(second.created, false)
  assert.equal(second.jobs.length, 1)
}

{
  let { jobs, job } = enqueueJob([], {
    kind: 'thread', owner: 'remote-thread', threadId: 'remote-thread', executionHostId: 'ssh-host',
  }, 925)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 950, 1_000).jobs
  const live = markRemoteJobLive(jobs, job.id, 'ssh-host', 'Remote run started.', 975)
  jobs = live.jobs
  assert.equal(live.job?.executionVerification, 'live')

  const dropped = markRemoteJobsUnverifiable(jobs, 'ssh-host', 'Orchestrator connection dropped.', 1_000)
  assert.equal(dropped.changed, true)
  assert.equal(dropped.affected.length, 1)
  assert.equal(dropped.jobs[0].status, 'running', 'transport loss does not invent process failure')
  assert.equal(dropped.jobs[0].executionVerification, 'unverifiable')
  assert.equal(dropped.jobs[0].finishedAt, null)

  const duplicate = markRemoteJobsUnverifiable(dropped.jobs, 'ssh-host', 'Orchestrator connection dropped.', 1_050)
  assert.equal(duplicate.changed, false)
  assert.equal(duplicate.jobs[0].updatedAt, 1_000)

  const heartbeat = heartbeatJobLease(dropped.jobs, job.id, 'worker-a', 1_100, 1_000)
  assert.equal(heartbeat.job?.executionVerification, 'live', 'a fresh remote heartbeat restores verifiability')
  assert.equal(heartbeat.job?.executionVerifiedAt, 1_100)

  const failed = failJob(heartbeat.jobs, job.id, 'Remote process reported exit 1.', 1_200)
  assert.equal(failed.job?.status, 'failed')
  assert.equal(failed.job?.executionVerification, 'exited', 'an observed terminal transition is distinct from transport loss')
  assert.equal(failed.job?.executionVerifiedAt, 1_200)

  const local = enqueueJob([], { kind: 'thread', owner: 'local-thread', threadId: 'local-thread' }, 1_300)
  const untouched = markRemoteJobsUnverifiable(local.jobs, 'ssh-host', 'drop', 1_400)
  assert.equal(untouched.changed, false)
  assert.equal(untouched.jobs[0].executionVerification, null)
}

{
  let { jobs, job } = enqueueJob([], { kind: 'ticket', owner: 'ticket-1', ticketId: 'ticket-1', threadId: 'thread-1' }, 200)
  const claimed = claimJobLease(jobs, job.id, 'worker-a', 250, 1_000)
  jobs = claimed.jobs
  job = claimed.job!
  assert.equal(job.status, 'running')
  assert.equal(job.attempt, 1)
  assert.equal(job.leaseOwner, 'worker-a')
  assert.equal(job.leaseExpiresAt, 1_250)

  const blocked = claimJobLease(jobs, job.id, 'worker-b', 300, 1_000)
  assert.equal(blocked.job, null)

  const beat = heartbeatJobLease(jobs, job.id, 'worker-a', 500, 1_000)
  jobs = beat.jobs
  job = beat.job!
  assert.equal(job.leaseExpiresAt, 1_500)
  assert.equal(job.lastUsefulProgressAt, 500)

  const done = completeJob(jobs, job.id, 900)
  assert.equal(done.job?.status, 'completed')
  assert.equal(done.job?.leaseOwner, null)
  assert.equal(done.job?.finishedAt, 900)
}

{
  let { jobs, job } = enqueueJob([], { kind: 'automation', owner: 'auto-1', automationId: 'auto-1', resumeAfterRestart: 'ask' }, 1_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 1_100).jobs
  const reconciled = reconcileJobsAfterRestart(jobs, 2_000)
  assert.equal(reconciled.changed, true)
  assert.equal(reconciled.jobs[0].status, 'paused')
  assert.equal(reconciled.jobs[0].leaseOwner, null)
  assert.match(reconciled.jobs[0].blockedReason ?? '', /confirmation/)
}

{
  const normalized = normalizeJobs([{ id: 'j1', kind: 'room', status: 'running' } as any], 3_000)
  assert.equal(normalized[0].owner, 'room:j1')
  assert.equal(normalized[0].attempt, 0)
  assert.equal(normalized[0].lastUsefulProgressAt, 3_000)
  assert.equal(normalized[0].resumeAfterRestart, 'ask')
  assert.deepEqual(normalized[0].pendingApprovals, [])

  const cancelled = cancelJob(normalized, 'j1', 'stop', 3_500)
  assert.equal(cancelled.job?.status, 'cancelled')
  assert.equal(cancelled.job?.blockedReason, 'stop')
}

{
  const normalized = normalizeJobs([
    { id: 'legacy-manual', kind: 'thread', status: 'running', resumeAfterRestart: 'manual' },
    { id: 'legacy-auto', kind: 'thread', status: 'running', resumeAfterRestart: 'auto' },
  ] as any, 3_750)
  assert.equal(normalized.find((job) => job.id === 'legacy-manual')?.resumeAfterRestart, 'ask')
  assert.equal(normalized.find((job) => job.id === 'legacy-auto')?.resumeAfterRestart, 'full_policy')
}

{
  let { jobs, job } = enqueueJob([], { kind: 'mission', owner: 'mission-2', missionId: 'mission-2' }, 4_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 4_100).jobs
  const paused = resumeJob(jobs, job.id, 'Continue from checkpoint 2.', 4_500)
  assert.equal(paused.job?.status, 'queued')
  assert.equal(paused.job?.leaseOwner, null)
  assert.equal(paused.job?.leaseExpiresAt, null)
  assert.equal(paused.job?.blockedReason, null)
  assert.equal(paused.job?.resumePlan, 'Continue from checkpoint 2.')
}

{
  let { jobs, job } = enqueueJob([], { kind: 'automation', owner: 'auto-2', automationId: 'auto-2' }, 5_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 5_100).jobs
  const failed = deadLetterJob(jobs, job.id, 'Repeated provider limit.', 5_500)
  assert.equal(failed.job?.status, 'dead_letter')
  assert.equal(failed.job?.blockedReason, 'Repeated provider limit.')
  assert.equal(failed.job?.finishedAt, 5_500)

  const retried = retryJob(failed.jobs, job.id, { reason: 'Manual retry after reset.', nextWakeAt: 6_000 }, 5_600)
  assert.equal(retried.job?.status, 'queued')
  assert.equal(retried.job?.nextWakeAt, 6_000)
  assert.equal(retried.job?.blockedReason, 'Manual retry after reset.')
  assert.equal(retried.job?.leaseOwner, null)
  assert.equal(retried.job?.currentRunId, null)
}

{
  let { jobs, job } = enqueueJob([], { kind: 'ticket', owner: 'ticket-2', ticketId: 'ticket-2' }, 7_000)
  const completed = completeJob(jobs, job.id, 7_100)
  assert.equal(retryJob(completed.jobs, job.id, { reason: 'nope' }, 7_200).job, null)
  assert.equal(resumeJob(completed.jobs, job.id, 'nope', 7_200).job, null)
  assert.equal(deadLetterJob(completed.jobs, job.id, 'nope', 7_200).job, null)
}

{
  let { jobs, job } = enqueueJob([], { kind: 'room', owner: 'room-1', roomId: 'room-1' }, 8_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 8_100, 1_000).jobs
  const detected = detectStuckJobs(jobs, 9_250, { leaseGraceMs: 100 })
  assert.equal(detected.changed, true)
  assert.equal(detected.stalled.length, 1)
  assert.equal(detected.jobs[0].status, 'stalled')
  assert.equal(detected.jobs[0].leaseOwner, null)
  assert.match(detected.jobs[0].blockedReason ?? '', /lease expired/i)

  const redetected = detectStuckJobs(detected.jobs, 10_250, { leaseGraceMs: 100 })
  assert.equal(redetected.changed, false, 'already stalled jobs are not re-stalled every reconciliation pass')
  assert.equal(redetected.stalled.length, 0)
  assert.equal(redetected.jobs[0].updatedAt, detected.jobs[0].updatedAt, 'stalled job timestamp is preserved until explicit resume/retry')
}

{
  const room = {
    id: 'room-1',
    projectId: 'project-1',
    name: 'Execution room',
    seats: [{ id: 'seat-1', name: 'Builder', role: 'Build', lane: 'development', providerId: 'provider-1', model: 'model-1' }],
    problem: 'Ship the backend slice',
    state: 'running',
    events: [{ id: 'event-1', tick: 1, kind: 'tool', seat: 'Builder', target: 'all', text: 'start_ticket_work', createdAt: 8_050 }],
    finalText: '',
    createdAt: 8_000,
    updatedAt: 8_050,
  } as any
  const tickets = [
    { id: 'ticket-1', projectId: 'project-1', roomId: 'room-1', title: 'Build hook', status: 'in_progress', runState: 'running' },
    { id: 'ticket-2', projectId: 'project-1', roomId: 'room-1', title: 'Review hook', status: 'review', runState: null },
  ] as any
  const metadata = buildRoomSupervisorJobMetadata({
    room,
    project: { id: 'project-1', name: 'TensorFold Studio' },
    tickets,
    patch: {
      missionId: 'mission-room-1',
      currentTurnSeat: 'Builder',
      workerBudget: { maxActive: 2, maxStarts: 4 },
      workerStarts: 1,
      phase: { phase: 'build', status: 'running', ticketId: 'ticket-1', ticketTitle: 'Build hook', seat: 'Builder', threadId: 'thread-1', jobId: 'job-ticket-1', paths: ['electron/main.ts'] },
    },
  }, 8_100)
  assert.equal(metadata.schema, 'vontra.room-supervisor/1')
  assert.equal(metadata.missionId, 'mission-room-1')
  assert.equal(metadata.roomState, 'running')
  assert.equal(metadata.workerBudget.active, 1)
  assert.equal(metadata.workerBudget.started, 1)
  assert.equal(metadata.ticketCounts.in_progress, 1)
  assert.equal(metadata.ticketCounts.review, 1)
  assert.equal(metadata.phases.build.status, 'running')
  assert.equal(metadata.phases.build.threadId, 'thread-1')
  assert.equal(metadata.phases.review.status, 'idle')

  const completed = buildRoomSupervisorJobMetadata({
    room,
    tickets,
    previous: metadata,
    patch: { phase: { phase: 'build', status: 'completed', ticketId: 'ticket-1', summary: 'Worker finished.' } },
  }, 8_500)
  assert.equal(completed.missionId, 'mission-room-1')
  assert.equal(completed.phases.build.status, 'completed')
  assert.equal(completed.phases.build.threadId, 'thread-1')
  assert.equal(completed.phases.build.summary, 'Worker finished.')

  const reviewed = buildRoomSupervisorJobMetadata({
    room,
    tickets,
    previous: completed,
    patch: { phase: { phase: 'review', status: 'completed', ticketId: 'ticket-1', ticketTitle: 'Build hook', seat: 'Reviewer', threadId: 'thread-2', jobId: 'job-review-1', paths: ['electron/main.ts'], summary: 'Reviewer passed.' } },
  }, 8_900)
  assert.equal(reviewed.phases.build.status, 'completed')
  assert.equal(reviewed.phases.review.status, 'completed')
  assert.equal(reviewed.phases.review.seat, 'Reviewer')
  assert.equal(reviewed.phases.review.threadId, 'thread-2')
  assert.equal(reviewed.phases.review.summary, 'Reviewer passed.')
}

// BUG #53: seat heartbeats flow from the live room into supervisor metadata, and
// persist across patches that do not refresh the room state.
{
  const stalledHeartbeats = {
    'seat-1': {
      seatId: 'seat-1', seatName: 'Builder', intervalSeconds: 60, stallGraceSeconds: 30,
      required: true, count: 2, lastBeatAt: 9_000, nextDueAt: 9_060, missedSince: 9_060,
    },
  }
  const room = {
    id: 'room-2', projectId: 'project-1', name: 'Stall room',
    seats: [{ id: 'seat-1', name: 'Builder', role: 'Build', providerId: 'provider-1', model: 'model-1' }],
    seatHeartbeats: stalledHeartbeats,
    problem: 'Investigate stall', state: 'running', events: [], finalText: '',
    createdAt: 9_000, updatedAt: 9_100,
  } as any
  const metadata = buildRoomSupervisorJobMetadata({ room, tickets: [] }, 9_200)
  assert.ok(metadata.seatHeartbeats)
  assert.equal(metadata.seatHeartbeats!['seat-1'].missedSince, 9_060)
  assert.equal(metadata.seatHeartbeats!['seat-1'].seatName, 'Builder')

  // A later patch with the room's heartbeats cleared falls back to the prior snapshot.
  const roomNoHb = { ...room, seatHeartbeats: undefined } as any
  const carried = buildRoomSupervisorJobMetadata({ room: roomNoHb, tickets: [], previous: metadata }, 9_300)
  assert.ok(carried.seatHeartbeats)
  assert.equal(carried.seatHeartbeats!['seat-1'].missedSince, 9_060)
}

{
  let { jobs, job } = enqueueJob([], { kind: 'mission', owner: 'mission-3', missionId: 'mission-3' }, 10_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 10_100, 20_000).jobs
  const detected = detectStuckJobs(jobs, 11_500, { progressTimeoutMs: 1_000 })
  assert.equal(detected.jobs[0].status, 'stalled')
  assert.match(detected.jobs[0].blockedReason ?? '', /No useful progress/)
}

{
  let { jobs, job } = enqueueJob([], { kind: 'ticket', owner: 'ticket-3', ticketId: 'ticket-3', metadata: { sameOutputCount: 3 } }, 12_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 12_100, 20_000).jobs
  const detected = detectStuckJobs(jobs, 12_200, { repeatedNoProgressThreshold: 3 })
  assert.equal(detected.jobs[0].status, 'stalled')
  assert.match(detected.jobs[0].blockedReason ?? '', /same-output/)
  assert.equal(detected.jobs[0].metadata?.stuckDetectedAt, 12_200)

  const noProgress = enqueueJob([], { kind: 'thread', owner: 'repeat-tool-call', threadId: 'repeat-tool-call', metadata: { noProgressCount: 3 } }, 12_300)
  const noProgressRunning = claimJobLease(noProgress.jobs, noProgress.job.id, 'worker-a', 12_350, 20_000).jobs
  const noProgressDetected = detectStuckJobs(noProgressRunning, 12_400, { repeatedNoProgressThreshold: 3 })
  assert.equal(noProgressDetected.jobs[0].status, 'stalled')
  assert.match(noProgressDetected.jobs[0].blockedReason ?? '', /same-output\/no-progress/)

  const repeatedNoProgress = enqueueJob([], { kind: 'thread', owner: 'repeat-failing-tool', threadId: 'repeat-failing-tool', metadata: { repeatedNoProgressCount: 4 } }, 12_500)
  const repeatedRunning = claimJobLease(repeatedNoProgress.jobs, repeatedNoProgress.job.id, 'worker-a', 12_550, 20_000).jobs
  const repeatedDetected = detectStuckJobs(repeatedRunning, 12_600, { repeatedNoProgressThreshold: 4 })
  assert.equal(repeatedDetected.jobs[0].status, 'stalled')
  assert.match(repeatedDetected.jobs[0].blockedReason ?? '', /same-output\/no-progress/)
}

{
  const waiting = normalizeJobs([{ id: 'approval-1', kind: 'thread', owner: 'thread-1', status: 'waiting_approval', leaseExpiresAt: 100, lastUsefulProgressAt: 100, updatedAt: 100, createdAt: 100 } as any], 13_000)
  const detected = detectStuckJobs(waiting, 30_000, { progressTimeoutMs: 1_000, leaseGraceMs: 1_000 })
  assert.equal(detected.changed, false)
  assert.equal(detected.jobs[0].status, 'waiting_approval')
}

{
  let { jobs, job } = enqueueJob([], { kind: 'thread', owner: 'thread-approval', threadId: 'thread-approval', resumeAfterRestart: 'ask' }, 14_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 14_050, 30_000).jobs
  const recorded = recordJobApproval(jobs, job.id, {
    callId: 'call-1',
    threadId: 'thread-approval',
    itemId: 'item-1',
    tool: 'run_command',
    policy: { scope: 'tool', key: 'tool:run_command:test', label: 'Run command', explanation: 'Command needs approval.' },
  }, 14_100)
  jobs = recorded.jobs
  job = recorded.job!
  assert.equal(job.status, 'waiting_approval')
  assert.equal(job.resumeAfterRestart, 'pending_approval')
  assert.equal(job.pendingApprovals.length, 1)
  assert.equal(job.pendingApprovals[0].resolvedAt, null)

  const reconciled = reconcileJobsAfterRestart(jobs, 14_500)
  assert.equal(reconciled.changed, true)
  assert.equal(reconciled.jobs[0].status, 'waiting_approval')
  assert.equal(reconciled.jobs[0].leaseOwner, null)
  assert.equal(reconciled.jobs[0].pendingApprovals[0].callId, 'call-1')

  const resolved = resolveJobApproval(reconciled.jobs, 'call-1', 'allow', 14_700)
  assert.equal(resolved.job?.status, 'queued')
  assert.equal(resolved.job?.pendingApprovals[0].decision, 'allow')
  assert.equal(resolved.job?.pendingApprovals[0].resolvedAt, 14_700)
  assert.equal(resolved.job?.resumePlan, 'Continue after the restored approval was resolved.')
}

{
  let { jobs, job } = enqueueJob([], { kind: 'thread', owner: 'thread-two-approvals', threadId: 'thread-two-approvals', resumeAfterRestart: 'full_policy' }, 15_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 15_050, 30_000).jobs
  let recorded = recordJobApproval(jobs, job.id, {
    callId: 'call-a',
    threadId: 'thread-two-approvals',
    itemId: 'item-a',
    tool: 'run_command',
    policy: { scope: 'tool', key: 'tool:run_command:test-a', label: 'Run command A', explanation: 'First approval.' },
  }, 15_100)
  jobs = recorded.jobs
  job = recorded.job!
  recorded = recordJobApproval(jobs, job.id, {
    callId: 'call-b',
    threadId: 'thread-two-approvals',
    itemId: 'item-b',
    tool: 'write_file',
    policy: { scope: 'tool', key: 'tool:write_file:path-b', label: 'Write file B', explanation: 'Second approval.' },
  }, 15_120)
  jobs = recorded.jobs

  const reconciled = reconcileJobsAfterRestart(jobs, 15_500)
  assert.equal(reconciled.jobs[0].status, 'waiting_approval')
  assert.equal(reconciled.jobs[0].resumeAfterRestart, 'pending_approval')
  assert.equal(reconciled.jobs[0].pendingApprovals.filter((approval) => !approval.resolvedAt).length, 2)

  const firstResolved = resolveJobApproval(reconciled.jobs, 'call-a', 'allow', 15_700)
  assert.equal(firstResolved.job?.status, 'waiting_approval')
  assert.equal(firstResolved.job?.resumeAfterRestart, 'pending_approval')
  assert.equal(firstResolved.job?.pendingApprovals.filter((approval) => !approval.resolvedAt).length, 1)

  const secondResolved = resolveJobApproval(firstResolved.jobs, 'call-b', 'deny', 15_900)
  assert.equal(secondResolved.job?.status, 'queued')
  assert.equal(secondResolved.job?.resumeAfterRestart, 'full_policy')
  assert.equal(secondResolved.job?.pendingApprovals.filter((approval) => !approval.resolvedAt).length, 0)
  assert.equal(secondResolved.job?.resumePlan, 'Continue after the restored approval was resolved.')
}

{
  let { jobs, job } = enqueueJob([], { kind: 'automation', owner: 'auto-resume', automationId: 'auto-resume', resumeAfterRestart: 'full_policy' }, 16_000)
  jobs = claimJobLease(jobs, job.id, 'worker-a', 16_100, 30_000).jobs
  const reconciled = reconcileJobsAfterRestart(jobs, 16_500)
  assert.equal(reconciled.changed, true)
  assert.equal(reconciled.jobs[0].status, 'queued')
  assert.equal(reconciled.jobs[0].leaseOwner, null)
  assert.equal(reconciled.jobs[0].currentRunId, null)
  assert.equal(reconciled.jobs[0].resumePlan, 'Automatically requeued after restart with the saved approval policy.')
}

{
  let { jobs, job } = enqueueJob([], { kind: 'automation', owner: 'auto-webhook', automationId: 'auto-webhook', metadata: { name: 'Webhook brief' } }, 18_000)
  const refreshed = enqueueJob(jobs, {
    kind: 'automation',
    owner: 'auto-webhook',
    automationId: 'auto-webhook',
    connectorEventId: 'event-webhook-1',
    metadata: { trigger: 'webhook', connectorId: 'webhook' },
  }, 18_100)
  jobs = refreshed.jobs
  job = refreshed.job
  assert.equal(refreshed.created, false)
  assert.equal(job.connectorEventId, 'event-webhook-1')
  assert.equal(job.metadata?.trigger, 'webhook')
  assert.equal(job.metadata?.connectorId, 'webhook')

  jobs = claimJobLease(jobs, job.id, 'worker-a', 18_150, 30_000).jobs
  const running = updateJob(jobs, job.id, { currentRunId: 'automation:auto-webhook:run-1', lastUsefulProgressAt: 18_200 }, 18_200)
  assert.equal(running.job?.currentRunId, 'automation:auto-webhook:run-1')
  assert.equal(running.job?.lastUsefulProgressAt, 18_200)
}

{
  const policies = [
    ['never', 'paused', /disabled/],
    ['ask', 'paused', /confirmation/],
    ['read_only', 'queued', /read-only mode/],
    ['low_risk', 'queued', /low-risk tool policy/],
    ['full_policy', 'queued', /saved approval policy/],
  ] as const
  for (const [policy, status, planPattern] of policies) {
    let { jobs, job } = enqueueJob([], { kind: 'thread', owner: `restart-${policy}`, threadId: `restart-${policy}`, resumeAfterRestart: policy }, 17_000)
    jobs = claimJobLease(jobs, job.id, 'worker-a', 17_100, 30_000).jobs
    const reconciled = reconcileJobsAfterRestart(jobs, 17_500)
    assert.equal(reconciled.changed, true)
    assert.equal(reconciled.jobs[0].status, status)
    assert.equal(reconciled.jobs[0].leaseOwner, null)
    assert.equal(reconciled.jobs[0].leaseExpiresAt, null)
    assert.equal(reconciled.jobs[0].resumeAfterRestart, policy)
    assert.match(`${reconciled.jobs[0].blockedReason ?? ''} ${reconciled.jobs[0].resumePlan ?? ''}`, planPattern)
  }
  assert.equal(restartPolicyAutoResumes('never'), false)
  assert.equal(restartPolicyAutoResumes('ask'), false)
  assert.equal(restartPolicyAutoResumes('read_only'), true)
  assert.equal(restartPolicyAutoResumes('low_risk'), true)
  assert.equal(restartPolicyAutoResumes('full_policy'), true)
  assert.equal(restartResumeApprovalMode('read_only', 'auto'), 'readonly')
  assert.equal(restartResumeApprovalMode('low_risk', 'auto'), 'suggest', 'low-risk resume never inherits Full Auto or its sandbox bypass')
  assert.equal(restartResumeApprovalMode('full_policy', 'suggest'), 'suggest')
}

{
  let { jobs, job } = enqueueJob([], {
    kind: 'room',
    owner: 'room-active-worker',
    roomId: 'room-active-worker',
    resumeAfterRestart: 'ask',
    metadata: {
      schema: 'vontra.room-supervisor/1',
      workerBudget: { maxActive: 2, maxStarts: 4, active: 1, started: 1 },
      phases: {
        build: {
          phase: 'build',
          status: 'running',
          ticketId: 'ticket-build',
          ticketTitle: 'Build hook',
          threadId: 'thread-worker',
          jobId: 'job-ticket-build',
          updatedAt: 18_900,
        },
      },
    },
  }, 18_800)
  jobs = claimJobLease(jobs, job.id, 'room-worker', 18_850, 30_000).jobs
  jobs = updateJob(jobs, job.id, { currentRunId: 'room:room-active-worker:turn-3' }, 18_900).jobs
  const reconciled = reconcileJobsAfterRestart(jobs, 19_000)
  assert.equal(reconciled.changed, true)
  assert.equal(reconciled.jobs[0].kind, 'room')
  assert.equal(reconciled.jobs[0].status, 'paused')
  assert.equal(reconciled.jobs[0].currentRunId, null)
  assert.equal(reconciled.jobs[0].leaseOwner, null)
  assert.equal(reconciled.jobs[0].leaseExpiresAt, null)
  assert.equal((reconciled.jobs[0].metadata?.workerBudget as any)?.active, 1)
  assert.match(reconciled.jobs[0].resumePlan ?? '', /room worker was active/i)
  assert.match(reconciled.jobs[0].resumePlan ?? '', /worker thread\/evidence/i)
}

{
  const policies = [
    ['ask', 'paused', /confirmation/],
    ['read_only', 'queued', /read-only mode/],
    ['low_risk', 'queued', /low-risk tool policy/],
    ['full_policy', 'queued', /saved approval policy/],
  ] as const
  for (const [policy, status, planPattern] of policies) {
    let { jobs, job } = enqueueJob([], { kind: 'mission', owner: `mission-restart-${policy}`, missionId: `mission-restart-${policy}`, resumeAfterRestart: policy }, 19_000)
    jobs = claimJobLease(jobs, job.id, 'mission-worker', 19_100, 30_000).jobs
    const reconciled = reconcileJobsAfterRestart(jobs, 19_500)
    assert.equal(reconciled.jobs[0].kind, 'mission')
    assert.equal(reconciled.jobs[0].missionId, `mission-restart-${policy}`)
    assert.equal(reconciled.jobs[0].status, status)
    assert.equal(reconciled.jobs[0].leaseOwner, null)
    assert.equal(reconciled.jobs[0].leaseExpiresAt, null)
    assert.match(`${reconciled.jobs[0].blockedReason ?? ''} ${reconciled.jobs[0].resumePlan ?? ''}`, planPattern)
  }
}

console.log('jobs tests passed')
