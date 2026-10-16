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
console.log('Full chat work activity deduplication, waiting approvals, stale status and room/routine/background visibility passed')
