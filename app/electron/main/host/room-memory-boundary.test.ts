import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const workspace = readFileSync(new URL('./workspace-runtime-ipc.ts', import.meta.url), 'utf8')
const main = readFileSync(new URL('./remote-host-api.ts', import.meta.url), 'utf8')
const remote = readFileSync(new URL('../../agent/remote-control-mutations.ts', import.meta.url), 'utf8')
const store = readFileSync(new URL('../../../src/lib/store/slices/roomsSlice.ts', import.meta.url), 'utf8')
const view = readFileSync(new URL('../../../src/components/rooms/RoomView.tsx', import.meta.url), 'utf8')

const local = workspace.slice(workspace.indexOf("handle('room:manualMemory'"), workspace.indexOf('// ---- Kanban tickets ----'))
assert.match(local, /pickRoomManualMemoryCommand/)
assert.match(local, /applyRoomManualMemoryCommand/)
assert.match(local, /randomUUID/, 'local host stamps memory ids and timestamps')
const localRoomUpdate = workspace.slice(workspace.indexOf("handle('rooms:update'"), workspace.indexOf("handle('rooms:delete'"))
assert.match(localRoomUpdate, /assertRoomSeatRoutesConfigured/, 'local room saves validate provider, account, model, and local-only routing')

const remoteHost = main.slice(main.indexOf('async function mutateRoomMemoryOnThisHost'), main.indexOf('async function archiveRoomOnThisHost'))
assert.match(remoteHost, /pickRoomManualMemoryCommand/)
assert.match(remoteHost, /applyRoomManualMemoryCommand/)
assert.match(remoteHost, /randomUUID/, 'remote host stamps memory ids and timestamps')
const remoteRoomUpdate = main.slice(main.indexOf('async function updateRoomOnThisHost'), main.indexOf('async function renameRoomOnThisHost'))
assert.match(remoteRoomUpdate, /assertRoomSeatRoutesConfigured/, 'remote room saves validate provider, account, model, and local-only routing')
assert.match(remote, /\/v1\\\/rooms\\\/\(\[\^\/\]\+\)\\\/memory/, 'remote REST exposes only the dedicated command route')

assert.match(store, /async mutateRoomMemory\(roomId, command: RoomManualMemoryCommand\)/)
assert.match(view, /mutateRoomMemory\(id, \{ action: 'add'/)
assert.match(view, /mutateRoomMemory\(id, \{ action: 'remove'/)
assert.doesNotMatch(view, /updateRoom\(id, \{ memory:/, 'renderer never replaces the whole memory array')

console.log('room-memory-boundary.test.ts passed')
