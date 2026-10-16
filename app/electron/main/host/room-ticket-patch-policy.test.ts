import assert from 'node:assert/strict'
import {
  assertRoomConfigUpdateAllowed,
  assertRoomSeatRoutesConfigured,
  applyRoomManualMemoryCommand,
  assertTicketUserUpdateAllowed,
  appendRoomSteeringEvent,
  pickRoomConfigPatch,
  pickRoomManualMemoryCommand,
  pickTicketUserPatch,
  normalizeRemoteTicketHost,
  ticketRunEvidence,
  pickTicketRemoteLifecycleCommand,
  pickTicketCreateInput,
} from './room-ticket-patch-policy'

assert.deepEqual({ ...pickRoomConfigPatch({ seats: [], mode: 'review' }) }, { seats: [], mode: 'review' })
assert.throws(() => pickRoomConfigPatch({ state: 'complete' }), /state cannot be changed/)
assert.throws(() => pickRoomConfigPatch({ events: [] }), /events cannot be changed/)
assert.throws(() => pickRoomConfigPatch({ memory: [] }), /memory cannot be changed/)
assert.throws(() => pickRoomConfigPatch(JSON.parse('{"__proto__":{"state":"complete"}}')), /__proto__ cannot be changed/)
assert.throws(() => pickRoomConfigPatch(Object.create({ events: [] })), /must not inherit fields/)
assert.throws(() => pickRoomConfigPatch({ mode: 'bogus' }), /mode has an invalid value/)
assert.throws(() => pickRoomConfigPatch({ seats: {} }), /seats must be an array/)
assert.throws(() => pickRoomConfigPatch({ workerBudget: { maxActive: 0, maxStarts: 4 } }), /maxActive/)
const provider = { id: 'provider-1', name: 'Provider', kind: 'local' as const, baseUrl: '', apiKey: '', models: ['model-1'] }
const seat = { id: 'seat-1', name: 'Builder', role: 'Build', providerId: provider.id, model: 'model-1' }
assert.doesNotThrow(() => assertRoomSeatRoutesConfigured([seat], [provider], { localOnly: true }))
assert.throws(() => assertRoomSeatRoutesConfigured([{ ...seat, model: 'missing' }], [provider], null), /model that is not offered/)
assert.throws(() => assertRoomSeatRoutesConfigured([seat], [{ ...provider, kind: 'cloud' }], { localOnly: true }), /must use a local provider/)
assert.deepEqual(pickRoomManualMemoryCommand({ action: 'add', kind: 'finding', text: '  durable fact  ' }), { action: 'add', kind: 'finding', text: 'durable fact' })
assert.deepEqual(pickRoomManualMemoryCommand({ action: 'remove', entryId: 'memory-1' }), { action: 'remove', entryId: 'memory-1' })
assert.throws(() => pickRoomManualMemoryCommand({ action: 'add', kind: 'forged', text: 'x' }), /kind has an invalid value/)
const runtimeMemory = { id: 'runtime-1', kind: 'finding' as const, text: 'Runtime fact', source: 'room' as const, createdAt: 1, updatedAt: 1 }
const manualMemory = { id: 'manual-1', kind: 'decision' as const, text: 'Manual fact', source: 'manual' as const, createdAt: 1, updatedAt: 1 }
assert.deepEqual(applyRoomManualMemoryCommand({ memory: [runtimeMemory, manualMemory] }, { action: 'remove', entryId: manualMemory.id }, () => 'unused'), [runtimeMemory])
assert.throws(() => applyRoomManualMemoryCommand({ memory: [runtimeMemory] }, { action: 'remove', entryId: runtimeMemory.id }, () => 'unused'), /Only manually added/)
assert.deepEqual(applyRoomManualMemoryCommand({ memory: [runtimeMemory] }, { action: 'add', kind: 'finding', text: 'New' }, () => 'manual-2', 9)[0], { id: 'manual-2', kind: 'finding', text: 'New', source: 'manual', createdAt: 9, updatedAt: 9 })
assert.throws(() => assertRoomConfigUpdateAllowed({ state: 'running' }, { seats: [] }), /Stop the active room run/)
assert.throws(() => assertRoomConfigUpdateAllowed({ state: 'ready' }, { seats: [] }, true), /Stop the active room run/)
assert.doesNotThrow(() => assertRoomConfigUpdateAllowed({ state: 'ready' }, { seats: [] }))

const steerRoom = {
  state: 'running' as const,
  seats: [{ id: 'review', name: 'Reviewer', role: 'Review', providerId: 'p', model: 'm', roleProfileId: 'reviewer' as const }],
  events: [{ id: 'old', tick: 4, kind: 'notice', seat: 'system', target: 'all', text: 'old', createdAt: 1 }],
}
const steer = appendRoomSteeringEvent(steerRoom, '  @reviewer inspect the diff  ', () => 'steer-1', 9)
assert.deepEqual(steer, { id: 'steer-1', tick: 5, kind: 'user', seat: 'you', target: 'Reviewer', text: '@reviewer inspect the diff', createdAt: 9 })
assert.equal(steerRoom.events.at(-1), steer)
assert.throws(() => appendRoomSteeringEvent({ ...steerRoom, state: 'ready' }, 'wait', () => 'x'), /Start the room/)

assert.deepEqual({ ...pickTicketUserPatch({ status: 'ready', roomId: 'room-1' }) }, { status: 'ready', roomId: 'room-1' })
for (const field of ['threadId', 'runState', 'startedAt', 'execution', 'evidence', 'assignee', 'orchestratorHostId']) {
  assert.throws(() => pickTicketUserPatch({ [field]: null }), new RegExp(`${field} cannot be changed`))
}
assert.throws(() => pickTicketUserPatch(JSON.parse('{"__proto__":{"runState":"done"}}')), /__proto__ cannot be changed/)
assert.throws(() => pickTicketUserPatch(Object.create({ runState: 'done' })), /must not inherit fields/)
assert.throws(() => pickTicketUserPatch({ status: 'bogus' }), /status has an invalid value/)
assert.throws(() => pickTicketUserPatch({ allowedCapabilities: ['root'] }), /allowedCapabilities/)
assert.throws(() => pickTicketUserPatch({ allowedCapabilities: ['read', 'read'] }), /unique supported values/)
assert.deepEqual({ ...pickTicketCreateInput({ projectId: 'project-1', title: 'Ship it', status: 'todo' }) }, { projectId: 'project-1', title: 'Ship it', status: 'todo' })
for (const field of ['threadId', 'runState', 'startedAt', 'execution', 'evidence', 'assignee']) {
  assert.throws(() => pickTicketCreateInput({ projectId: 'project-1', title: 'Forged', [field]: null }), new RegExp(`${field} cannot be changed`))
}
assert.throws(() => assertTicketUserUpdateAllowed({ runState: 'queued' }, { status: 'completed' }), /Stop the active ticket run/)
assert.throws(() => assertTicketUserUpdateAllowed({ runState: 'running' }, { roomId: null }), /Stop the active ticket run/)
assert.throws(() => assertTicketUserUpdateAllowed({ runState: 'running' }, { allowedCapabilities: ['network'] }), /Stop the active ticket run/)
assert.throws(() => assertTicketUserUpdateAllowed({ runState: null }, { executionMode: 'room' }, true), /Stop the active ticket run/)

assert.deepEqual(pickTicketRemoteLifecycleCommand({ action: 'begin', threadId: 'thread-1', hostId: 'host-1' }), { action: 'begin', threadId: 'thread-1', hostId: 'host-1' })
assert.throws(() => pickTicketRemoteLifecycleCommand({ action: 'sync', runState: 'done' }), /runState is not supported/)
assert.throws(() => pickTicketRemoteLifecycleCommand(JSON.parse('{"action":"cancel","__proto__":{"action":"sync"}}')), /__proto__ is not supported/)
const correlatedTimeline = [
  { id: 'old', kind: 'assistant' as const, text: 'old output', trace: { traceId: 'trace-old', runId: 'old-run', surface: 'remote' as const } },
  { id: 'wrong', kind: 'assistant' as const, text: 'wrong run', trace: { traceId: 'trace-wrong', runId: 'other-run', surface: 'remote' as const } },
  { id: 'current', kind: 'assistant' as const, text: 'current output', trace: { traceId: 'trace-current', runId: 'run-2', surface: 'remote' as const } },
]
assert.deepEqual(ticketRunEvidence(correlatedTimeline, { timelineBaseline: 1, runId: 'run-2' }), { correlated: true, ran: true, failed: false })
assert.deepEqual(ticketRunEvidence(correlatedTimeline.slice(0, 1), { timelineBaseline: 1, runId: 'run-2' }), { correlated: true, ran: false, failed: false })
assert.deepEqual(ticketRunEvidence(correlatedTimeline, { timelineBaseline: undefined, runId: null }), { correlated: false, ran: false, failed: false })
const normalizedTicket = normalizeRemoteTicketHost({ execution: { target: 'remote', hostId: 'other-controller' } } as any, 'local-profile')
assert.equal(normalizedTicket.orchestratorHostId, 'local-profile')
assert.equal(normalizedTicket.execution?.hostId, 'local-profile')

console.log('room-ticket-patch-policy.test.ts passed')
