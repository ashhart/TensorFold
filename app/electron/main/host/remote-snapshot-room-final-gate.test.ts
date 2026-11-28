import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-room-final-block-'))
const { store } = await import('../state')
const { pairedRemoteArchiveSnapshot, pairedRemoteControlSnapshot, redactRemoteControlEventForPaired } = await import('./remote-snapshot')

const secret = 'PAIRED_FINAL_BLOCK_SECRET'
const blocker = { id: 'block', tick: 1, kind: 'check', seat: 'moderator', target: 'builder', text: `Final blocked: ${secret}`, createdAt: 1 }
const builderCopy = { id: 'builder-copy', tick: 2, kind: 'check', seat: 'builder', target: 'all', text: `Final blocked: ${secret}`, createdAt: 2 }
const ordinaryCheck = { id: 'ordinary', tick: 3, kind: 'check', seat: 'moderator', target: 'all', text: 'All checks passed.', createdAt: 3 }
const final = { id: 'final', tick: 4, kind: 'final', seat: 'moderator', target: 'all', text: 'Room finalized.', createdAt: 4 }

store.settings = { remoteControl: { serverName: 'Host' }, providers: [] } as any
store.projects = [{ id: 'project', name: 'Project', cwd: '/private/project', archivedAt: null, createdAt: 1, updatedAt: 1 }] as any
store.rooms = [{
  id: 'room', projectId: 'project', name: 'Room', seats: [], problem: secret, state: 'ready',
  events: [blocker, builderCopy, ordinaryCheck, final], memory: [], memorySummary: '', finalText: secret,
  createdAt: 1, updatedAt: 1,
}] as any
store.threads = []
store.tickets = []
store.automations = []
store.missions = []
store.jobs = []
store.connectorEvents = []
const runtime = { appVersion: '1', headless: false, activeThreadIds: [], activeRoomIds: [], activeMissionIds: [] }

function assertProjection(room: any, label: string) {
  const events = room.events
  assert.equal(events[0].text, '', `${label}: the moderator message stays redacted`)
  assert.equal(events[0].finalBlock, true, `${label}: the moderator verdict survives as a safe bit`)
  assert.equal(events[0].args, undefined)
  assert.equal(events[0].path, undefined)
  assert.equal(events[1].finalBlock, undefined, `${label}: an identical check from another seat is not the moderator verdict`)
  assert.equal(events[2].finalBlock, undefined, `${label}: an ordinary moderator check is not a block`)
  assert.equal(events[3].kind, 'final', `${label}: a later final event still clears the verdict`)
  assert.equal(events[3].finalBlock, undefined)
  assert.doesNotMatch(JSON.stringify(room), new RegExp(secret), `${label}: the blocker message never leaves the owner`)
}

assertProjection(pairedRemoteControlSnapshot(runtime).rooms[0], 'paired snapshot')
assertProjection(pairedRemoteArchiveSnapshot({
  projects: [], threads: [], tickets: [],
  rooms: [{ ...store.rooms[0], archivedAt: 2 }],
} as any).rooms[0], 'paired archive')

const liveBlock = redactRemoteControlEventForPaired({ kind: 'room', data: { roomId: 'room', type: 'event', item: blocker } as any })
assert.equal((liveBlock.data as any).item.text, '', 'paired live event: the moderator message stays redacted')
assert.equal((liveBlock.data as any).item.finalBlock, true, 'paired live event: the moderator verdict survives as a safe bit')
assert.doesNotMatch(JSON.stringify(liveBlock), new RegExp(secret), 'paired live event: the blocker message never leaves the owner')

const liveBuilderCopy = redactRemoteControlEventForPaired({ kind: 'room', data: { roomId: 'room', type: 'event', item: builderCopy } as any })
assert.equal((liveBuilderCopy.data as any).item.finalBlock, undefined, 'paired live event: another seat cannot set the moderator verdict')

const liveFinal = redactRemoteControlEventForPaired({ kind: 'room', data: { roomId: 'room', type: 'final', text: secret } as any })
assert.equal((liveFinal.data as any).text, '', 'paired live final: the final text stays redacted')
assert.equal((liveFinal.data as any).finalBlock, undefined, 'paired live final: no block verdict on a final event')

console.log('paired room snapshots, archives and live events keep the moderator final-block verdict while the blocker message stays redacted')
