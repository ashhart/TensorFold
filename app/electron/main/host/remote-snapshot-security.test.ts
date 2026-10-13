import assert from 'node:assert/strict'
import { store } from '../state'
import { pairedRemoteArchiveSnapshot, pairedRemoteControlSnapshot, redactRemoteControlEventForPaired, remoteControlSnapshot } from './remote-snapshot'

const secret = 'snapshot-secret-material'
store.settings = {
  remoteControl: { serverName: 'Host' },
  providers: [{ id: 'p', name: 'Provider', kind: 'cloud', apiKey: secret, models: ['m'] }],
} as any
store.projects = [{ id: 'project', name: 'Project', cwd: `/private/${secret}`, workspace: { kind: 'ssh', hostId: secret, cwd: '/srv' }, memory: secret, team: secret, defaultProviderAccountId: secret, createdAt: 1, updatedAt: 1 }] as any
store.threads = [{ id: 'thread', title: 'Thread', kind: 'task', cwd: `/private/${secret}`, workspace: null, providerId: 'p', providerAccountId: secret, model: 'm', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: secret }], timeline: [{ id: 'item', kind: 'tool', output: secret, args: { token: secret }, diff: [{ type: 'add', text: secret }], state: 'ok' }], goal: secret, notepad: secret, annotations: secret }] as any
store.rooms = [{ id: 'room', projectId: 'project', name: 'Room', seats: [], rootCwd: `/private/${secret}`, problem: secret, state: 'idle', events: [{ id: 'event', tick: 1, kind: 'notice', seat: 'system', target: '', text: secret, args: { secret }, path: secret, createdAt: 1 }], memory: [{ id: 'memory', kind: 'finding', text: secret, source: 'room', createdAt: 1, updatedAt: 1 }], memorySummary: secret, finalText: secret, createdAt: 1, updatedAt: 1 }] as any
store.tickets = [{ id: 'ticket', projectId: 'project', title: 'Ticket', body: secret, type: 'task', status: 'todo', priority: 'low', effort: 'small', tags: [], acceptanceCriteria: [secret], evidence: [{ id: 'e', kind: 'note', summary: secret, createdAt: 1 }], parentId: null, assignee: null, threadId: null, runState: null, order: 0, createdAt: 1, updatedAt: 1 }] as any
store.automations = [{ id: 'automation', prompt: secret }] as any
store.missions = [{ id: 'mission', objective: secret, pendingApprovals: [] }] as any
store.jobs = [{ id: 'job', kind: 'thread', owner: 'thread', status: 'queued', metadata: { secret }, pendingApprovals: [{ id: 'a', tool: secret }], createdAt: 1, updatedAt: 1 }] as any
store.connectorEvents = [{ id: 'connector', payload: { secret }, createdAt: 1 }] as any
store.cliSessions = []
store.projects.push({ id: 'archived-project', archivedAt: 2, updatedAt: 2 } as any)
store.threads.push({ id: 'archived-thread', archivedAt: 2, updatedAt: 2 } as any)
store.rooms.push({ id: 'archived-room', archivedAt: 2, updatedAt: 2 } as any)
store.tickets.push({ id: 'archived-ticket', archivedAt: 2, updatedAt: 2 } as any)

const runtime = { appVersion: '1', headless: false, activeThreadIds: [], activeRoomIds: [], activeMissionIds: [] }
const owner = remoteControlSnapshot(runtime)
assert.match(JSON.stringify(owner), new RegExp(secret), 'owner snapshot retains local data')
assert.deepEqual(owner.projects.map((item) => item.id), ['project'])
assert.deepEqual(owner.threads.map((item) => item.id), ['thread'])
assert.deepEqual(owner.rooms.map((item) => item.id), ['room'])
assert.deepEqual(owner.tickets.map((item) => item.id), ['ticket'], 'archived ticket tombstones never reappear in reconnect snapshots')
const paired = pairedRemoteControlSnapshot(runtime)
assert.doesNotMatch(JSON.stringify(paired), new RegExp(secret), 'paired projection omits prompts, paths, memory, room text, job metadata, and credentials')
assert.equal(paired.threads[0].apiMessages.length, 0)
assert.equal(paired.rooms[0].events[0].text, '')
assert.equal(paired.automations.length, 0)
assert.equal(paired.missions.length, 0)
assert.equal(paired.connectorEvents.length, 0)

const pairedArchives = pairedRemoteArchiveSnapshot({
  projects: [{ ...store.projects[0], archivedAt: 2 }],
  threads: [{ ...store.threads[0], archivedAt: 2 }],
  rooms: [{ ...store.rooms[0], archivedAt: 2 }],
  tickets: [{ ...store.tickets[0], archivedAt: 2 }],
} as any)
assert.doesNotMatch(JSON.stringify(pairedArchives), new RegExp(secret), 'paired archive listing is minimized like the active snapshot')

const event = redactRemoteControlEventForPaired({ kind: 'agent', data: { type: 'item_delta', runId: 'r', threadId: 't', id: 'i', field: 'text', chunk: secret } })
assert.doesNotMatch(JSON.stringify(event), new RegExp(secret), 'paired live events cannot reintroduce redacted transcript chunks')

console.log('remote paired snapshot minimization tests passed')
