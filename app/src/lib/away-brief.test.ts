import assert from 'node:assert/strict'
import { buildAwayBriefItems } from './away-brief'
import type { Automation, ConnectorEvent, Job, Mission, Room } from './types'

const now = 1_725_000_000_000
const since = now - 24 * 60 * 60_000

const connectorEvent = eventFixture({
  id: 'evt-source',
  connectorId: 'docs',
  title: 'Roadmap source changed',
  summary: 'The roadmap picked up a new acceptance check.',
  sources: [{ id: 'docs:roadmap.md', connectorId: 'docs', title: 'roadmap.md', path: '/repo/roadmap.md' }],
  createdAt: now - 10_000,
})
const room = roomFixture({ id: 'room-launch', name: 'Launch Room', state: 'waiting_external', updatedAt: now - 20_000 })
const mission = missionFixture({ id: 'mission-review', title: 'Review source drift', status: 'running', updatedAt: now - 30_000 })
const automation = automationFixture({
  id: 'auto-brief',
  name: 'Morning Brief',
  runs: [{ id: 'run-1', status: 'error', error: 'Provider quota exhausted', output: '', startedAt: now - 45_000, finishedAt: now - 40_000 }],
})
const prJob = jobFixture({
  id: 'job-pr',
  kind: 'pr_loop',
  status: 'waiting_approval',
  metadata: { title: 'Connector feed PR', localPrId: 'lpr_123', branch: 'codex/connector-feed' },
  updatedAt: now - 50_000,
})

{
  const rows = buildAwayBriefItems({
    now,
    since,
    connectorEvents: [connectorEvent],
    rooms: [room],
    roomStatus: {},
    missions: [mission],
    automations: [automation],
    jobs: [prJob],
  })
  assert.deepEqual(rows.map((row) => row.kind).sort(), ['automation', 'connector', 'mission', 'pr_loop', 'room'])
  assert.deepEqual(rows.find((row) => row.kind === 'connector')?.sourceIds, ['docs:roadmap.md'])
  assert.match(rows.find((row) => row.kind === 'pr_loop')?.title ?? '', /Connector feed PR/)
  assert.match(rows.find((row) => row.kind === 'automation')?.detail ?? '', /Provider quota/)
}

{
  const staleCompletedPr = jobFixture({
    id: 'job-old-pr',
    kind: 'pr_loop',
    status: 'completed',
    updatedAt: since - 1,
    finishedAt: since - 1,
  })
  const rows = buildAwayBriefItems({
    now,
    since,
    connectorEvents: [eventFixture({ id: 'evt-old', createdAt: since - 1, requiresApproval: true })],
    rooms: [roomFixture({ id: 'room-old', state: 'ready', updatedAt: since - 1 })],
    missions: [missionFixture({ id: 'mission-old', status: 'completed', updatedAt: since - 1 })],
    automations: [automationFixture({ id: 'auto-running', runs: [{ id: 'run-running', status: 'running', output: '', startedAt: since - 1, finishedAt: null }] })],
    jobs: [staleCompletedPr],
  })
  assert.ok(rows.some((row) => row.kind === 'connector' && row.subjectId === 'evt-old'), 'pending connector approvals remain visible')
  assert.ok(rows.some((row) => row.kind === 'automation' && row.subjectId === 'auto-running'), 'running automations remain visible')
  assert.ok(!rows.some((row) => row.kind === 'room'), 'quiet stale rooms are filtered')
  assert.ok(!rows.some((row) => row.kind === 'mission'), 'completed stale missions are filtered')
  assert.ok(!rows.some((row) => row.kind === 'pr_loop'), 'completed stale PR loops are filtered')
}

{
  const rows = buildAwayBriefItems({
    now,
    since,
    limit: 2,
    connectorEvents: [connectorEvent],
    rooms: [room],
    missions: [mission],
    automations: [automation],
    jobs: [prJob],
  })
  assert.equal(rows.length, 2)
  assert.ok(rows[0].createdAt >= rows[1].createdAt)
}

function eventFixture(overrides: Partial<ConnectorEvent> = {}): ConnectorEvent {
  return {
    id: 'evt',
    connectorId: 'email',
    pluginName: 'Email',
    action: 'message_received',
    direction: 'inbound',
    classification: 'communication',
    priority: 'normal',
    trustLevel: 'notify',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: null,
    resolvedBy: null,
    title: 'Connector event',
    summary: 'Something changed.',
    actor: 'Connector',
    threadKey: 'connector:thread',
    payloadKeys: [],
    createdAt: now,
    ...overrides,
  }
}

function roomFixture(overrides: Partial<Room> = {}): Room {
  return {
    id: 'room',
    projectId: 'project',
    name: 'Room',
    seats: [],
    problem: 'Room problem.',
    state: 'ready',
    events: [],
    finalText: '',
    createdAt: now - 60_000,
    updatedAt: now - 60_000,
    ...overrides,
  }
}

function missionFixture(overrides: Partial<Mission> = {}): Mission {
  return {
    id: 'mission',
    title: 'Mission',
    objective: 'Mission objective.',
    stopCondition: '',
    repeatMode: 'until_condition',
    contextMode: 'build_on_previous',
    roleProfileId: 'operator',
    fanOutBudget: 'single',
    maxRuns: 15,
    costCapUsd: null,
    expiresAt: null,
    destructiveApprovalRequired: true,
    status: 'queued',
    runBudgetClass: 'mission',
    budget: { executionClass: 'mission', label: 'Mission', maxWallSeconds: 3600, heartbeatIntervalSeconds: 30, stallGraceSeconds: 120, requiresHeartbeat: true, startedAt: null, deadlineAt: null },
    heartbeat: { intervalSeconds: 30, stallGraceSeconds: 120, required: true, count: 0, lastBeatAt: null, nextDueAt: null, missedSince: null },
    checkpoints: [],
    journal: [],
    scratchMemory: [],
    pendingApprovals: [],
    threadId: null,
    projectId: null,
    cwd: null,
    providerId: null,
    model: null,
    createdAt: now - 60_000,
    updatedAt: now - 60_000,
    startedAt: null,
    finishedAt: null,
    ...overrides,
  }
}

function automationFixture(overrides: Partial<Automation> = {}): Automation {
  return {
    id: 'automation',
    name: 'Automation',
    prompt: 'Summarize changes.',
    scheduleKind: 'daily',
    everyMinutes: 0,
    atHour: 9,
    atMinute: 0,
    providerId: null,
    model: null,
    mode: 'agent',
    cwd: null,
    deliverTo: 'inbox',
    enabled: true,
    lastRunAt: null,
    nextRunAt: null,
    lastStatus: null,
    runs: [],
    createdAt: now - 60_000,
    updatedAt: now - 60_000,
    ...overrides,
  }
}

function jobFixture(overrides: Partial<Job> = {}): Job {
  return {
    id: 'job',
    kind: 'pr_loop',
    status: 'queued',
    attempt: 1,
    owner: 'TensorFold Studio',
    projectId: null,
    missionId: null,
    roomId: null,
    threadId: null,
    currentRunId: null,
    currentThreadId: null,
    nextWakeAt: null,
    lastUsefulProgressAt: null,
    blockedReason: null,
    resumePlan: null,
    resumeAfterRestart: 'ask',
    pendingApprovals: [],
    leaseOwner: null,
    leaseExpiresAt: null,
    createdAt: now - 60_000,
    updatedAt: now - 60_000,
    ...overrides,
  }
}

console.log('away brief tests passed')
