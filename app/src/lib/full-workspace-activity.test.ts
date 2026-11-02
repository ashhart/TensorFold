import assert from 'node:assert/strict'
import { fullWorkspaceActivity } from './full-workspace-activity'

const thread: any = { id: 'thread', title: 'Synthetic conversation', status: 'running', updatedAt: 2, timeline: [] }
const state: any = { threads: [thread], status: {}, approvals: {}, jobs: [], rooms: [], roomStatus: {}, automations: [] }
assert.equal(fullWorkspaceActivity(state).length, 1)
state.jobs = [{ id: 'job', kind: 'thread', threadId: 'thread', status: 'running', updatedAt: 3, pendingApprovals: [] }]
assert.equal(fullWorkspaceActivity(state).length, 1, 'a thread and its durable job are one activity')
state.status.thread = 'awaiting';assert.equal(fullWorkspaceActivity(state)[0].needsYou, true)
state.status.thread = 'done';state.jobs = [];assert.equal(fullWorkspaceActivity(state).length, 0, 'live completion overrides stale recorded running status')
thread.timeline = [{ id: 'tool', kind: 'tool', tool: 'write_file', state: 'awaiting' }];state.approvals.thread = [{ callId: 'call', itemId: 'tool' }]
assert.equal(fullWorkspaceActivity(state)[0].state, 'Needs you', 'an actual inline approval stays discoverable even between status events')
state.approvals = {};thread.timeline = [];state.threads = []
state.rooms = [{ id: 'room', name: 'Synthetic room', state: 'waiting_user', updatedAt: 4 }];state.jobs = [{ id: 'room-job', kind: 'room', roomId: 'room', status: 'waiting_approval', updatedAt: 5, pendingApprovals: [] }]
assert.equal(fullWorkspaceActivity(state).length, 1);assert.equal(fullWorkspaceActivity(state)[0].needsYou, true)
state.automations = [{ id: 'routine', name: 'Synthetic routine', updatedAt: 6, runs: [{ status: 'running' }] }];state.jobs.push({ id: 'routine-job', kind: 'automation', automationId: 'routine', status: 'running', updatedAt: 7, pendingApprovals: [] })
assert.equal(fullWorkspaceActivity(state).length, 2)
state.jobs.push({ id: 'other', kind: 'connector', status: 'stalled', updatedAt: 8, pendingApprovals: [] })
const rows = fullWorkspaceActivity(state);assert.equal(rows.length, 3);assert.equal(rows[0].state, 'Needs attention')
state.jobs.push({ id: 'finished', kind: 'thread', status: 'completed', updatedAt: 10, pendingApprovals: [] });assert.equal(fullWorkspaceActivity(state).length, 3)
assert.equal(state.rooms[0].state, 'waiting_user');assert.equal(state.jobs[0].status, 'waiting_approval', 'activity presentation never changes ownership or state')

const task: any = { id: 'task', title: 'Investigate parser', kind: 'task', projectId: 'project', status: 'idle', updatedAt: 10, timeline: [{ id: 'question', question: { state: 'pending', resolvedAt: null } }] }
const attention: any = { threads: [task], status: { task: 'done' }, approvals: {}, jobs: [], rooms: [], roomStatus: {}, automations: [], projects: [{ id: 'project', name: 'Compiler' }] }
let item = fullWorkspaceActivity(attention)[0]
assert.equal(item.reason, 'question', 'a durable unanswered question stays visible after a live status transition')
assert.equal(item.projectName, 'Compiler')
assert.equal(item.kind, 'task')
assert.deepEqual(item.view, { name: 'thread', id: 'task' })
task.timeline[0].question.state = 'answered'
assert.equal(fullWorkspaceActivity(attention).length, 0, 'answered questions leave the attention list')
task.timeline[0].question = { state: 'pending', resolvedAt: 9 }
assert.equal(fullWorkspaceActivity(attention).length, 0, 'a resolved question cannot appear pending from a stale state label')
task.timeline = []
task.runRecovery = { runId: 'run', retry: 'requires_review', canStartNewRun: true }
assert.equal(fullWorkspaceActivity(attention)[0].reason, 'recovery', 'uncertain outcomes point to the existing run review flow')
task.runRecoveryReviewedId = 'run'
assert.equal(fullWorkspaceActivity(attention).length, 0, 'an acknowledged outcome is no longer a pending review')
task.runRecoveryReviewedId = undefined
task.runRecovery.canStartNewRun = false
assert.equal(fullWorkspaceActivity(attention).length, 0, 'draining operations do not advertise a review action before it is available')
task.runRecovery = undefined
attention.status.task = 'error'
assert.equal(fullWorkspaceActivity(attention)[0].reason, 'error')
attention.status.task = 'streaming'
task.timeline = [{ id: 'approve', kind: 'tool', state: 'awaiting' }]
attention.approvals.task = [{ itemId: 'approve', callId: 'call' }]
attention.jobs = [{ id: 'durable', kind: 'thread', threadId: 'task', status: 'stalled', updatedAt: 11, pendingApprovals: [] }]
item = fullWorkspaceActivity(attention)[0]
assert.equal(fullWorkspaceActivity(attention).length, 1)
assert.equal(item.reason, 'approval', 'an actionable approval is not hidden by a duplicate stalled job')
assert.equal(item.updatedAt, 11)
assert.equal(item.title, 'Investigate parser')
assert.equal(item.projectName, 'Compiler')

attention.approvals = {}
task.timeline = []
attention.status.task = 'done'
attention.jobs = [{ id: 'handoff', kind: 'thread', threadId: 'old', currentThreadId: 'task', status: 'waiting_approval', updatedAt: 12, pendingApprovals: [] }]
item = fullWorkspaceActivity(attention)[0]
assert.deepEqual(item.view, { name: 'thread', id: 'task' }, 'a handed-off job opens its current thread')
attention.jobs[0].archivedAt = 13
assert.equal(fullWorkspaceActivity(attention).length, 0, 'reconciled-away durable jobs do not revive old work')
task.archivedAt = 14
attention.status.task = 'error'
assert.equal(fullWorkspaceActivity(attention).length, 0, 'archived threads remain hidden')

const sourceState: any = { threads: [{ id: 'worker', kind: 'task', title: 'Worker', status: 'idle', updatedAt: 1, timeline: [] }], status: {}, approvals: {}, rooms: [{ id: 'team', name: 'Team', state: 'running', updatedAt: 1 }], roomStatus: {}, automations: [], jobs: [{ id: 'team-job', kind: 'room', roomId: 'team', currentThreadId: 'worker', status: 'stalled', updatedAt: 20, pendingApprovals: [] }] }
item = fullWorkspaceActivity(sourceState)[0]
assert.equal(item.reason, 'stalled')
assert.equal(item.title, 'Team')
assert.deepEqual(item.view, { name: 'room', id: 'team' }, 'room supervision issues open the room that owns the work')
sourceState.jobs[0].status = 'waiting_approval'
assert.deepEqual(fullWorkspaceActivity(sourceState)[0].view, { name: 'thread', id: 'worker' }, 'a concrete worker approval opens the thread where the user can act')
console.log('Work activity grouping, approval/question/recovery provenance, deduplication, handoffs and source navigation passed')
