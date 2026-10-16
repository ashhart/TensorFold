import assert from 'node:assert/strict'
import { requireActiveRoomForCancel, reserveRoomStartup } from './workspace-runtime-ipc'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

const aborters = new Map<string, AbortController>()
const guard = new WorkspaceMutationGuard()
const first = reserveRoomStartup('room-1', aborters, guard, ['/repo'])
assert.ok(first, 'first room start reserves synchronously')
assert.equal(aborters.has('room-1'), true)
assert.equal(reserveRoomStartup('room-1', aborters, guard, ['/repo']), null, 'concurrent room start is rejected before any await')
assert.throws(() => guard.acquire({ workspaceKey: '/repo' }), /already in progress/, 'restore cannot overlap paused room worktree setup')

aborters.delete('room-1')
first!.runLease.release()
const retry = reserveRoomStartup('room-1', aborters, guard, ['/repo'])
assert.ok(retry, 'room start succeeds after failed startup releases reservation')
aborters.delete('room-1')
retry!.runLease.release()

const activeRoom = { id: 'active-room', archivedAt: null } as any
assert.equal(requireActiveRoomForCancel([activeRoom], activeRoom.id), activeRoom)
assert.throws(() => requireActiveRoomForCancel([], 'missing'), /Room not found/, 'missing room cancellation fails closed')
assert.throws(
  () => requireActiveRoomForCancel([{ id: 'archived-room', archivedAt: Date.now() } as any], 'archived-room'),
  /Restore this archived room/,
  'archived room cancellation cannot mutate the retained record',
)

console.log('room startup reservation and cancellation ownership boundaries passed')
