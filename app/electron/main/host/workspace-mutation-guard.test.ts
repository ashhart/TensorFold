import assert from 'node:assert/strict'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

const guard = new WorkspaceMutationGuard()
const lease = guard.acquire({ threadId: 'thread-1', workspaceKey: '/canonical/repo' })

assert.throws(
  () => guard.assertThreadStartAllowed('thread-1', '/canonical/repo'),
  /restoration is in progress/,
  'same-thread start is rejected while checkpoint restore holds its lease',
)
assert.throws(
  () => guard.assertThreadStartAllowed('thread-2', '/canonical/repo'),
  /restoration is in progress/,
  'a second thread sharing the local cwd cannot start during restore',
)
assert.throws(
  () => guard.assertRoomStartAllowed(['/other', '/canonical/repo']),
  /restoration is in progress/,
  'a room sharing the local cwd cannot start during restore',
)

lease.release()
assert.doesNotThrow(() => guard.assertThreadStartAllowed('thread-1', '/canonical/repo'), 'thread start succeeds after lease release')
assert.doesNotThrow(() => guard.assertRoomStartAllowed(['/canonical/repo']), 'room start succeeds after lease release')

const startReservation = guard.reserveStart({ threadId: 'thread-starting', workspaceKeys: ['/canonical/repo'] })
assert.throws(() => guard.acquire({ workspaceKey: '/canonical/repo' }), /already in progress/, 'restore cannot acquire while startup owns the workspace reservation')
assert.throws(() => guard.acquire({ threadId: 'thread-starting' }), /already in progress/, 'same-thread restore cannot acquire while chat/thread startup is reserved')
startReservation.release()
assert.doesNotThrow(() => guard.acquire({ workspaceKey: '/canonical/repo' }).release(), 'restore can acquire after startup reservation releases')

const blocked = guard.acquire({ workspaceKey: '/locked' })
assert.throws(
  () => guard.reserveStart({ threadId: 'no-leak', workspaceKeys: ['/free', '/locked'] }),
  /restoration is in progress/,
  'mixed-key start reservation fails atomically when any workspace is exclusively locked',
)
blocked.release()
const noLeak = guard.acquire({ threadId: 'no-leak', workspaceKey: '/free' })
noLeak.release()

const liveRun = guard.reserveStart({ threadId: 'live-run', workspaceKeys: ['/purge-target'] })
const globalPending = guard.acquireGlobal()
assert.throws(
  () => guard.reserveStart({ threadId: 'late-run', workspaceKeys: ['/other'] }),
  /security purge is in progress/,
  'the purge gate closes synchronously before waiting for admitted runs to settle',
)
let globalAcquired = false
void globalPending.then(() => { globalAcquired = true })
await Promise.resolve()
assert.equal(globalAcquired, false, 'global purge waits for an existing workspace lease')
liveRun.release()
const globalLease = await globalPending
assert.equal(globalAcquired, true)
assert.throws(() => guard.acquire({ workspaceKey: '/purge-target' }), /security purge is in progress/, 'global lease excludes restores too')
globalLease.release()
assert.doesNotThrow(() => guard.reserveStart({ threadId: 'after-purge', workspaceKeys: ['/purge-target'] }).release())

const shutdownWork = guard.reserveStart({ threadId: 'saving-question', workspaceKeys: ['/question'] })
const quitPause = guard.pauseAdmission('TensorFold is quitting')
assert.throws(() => guard.reserveStart({ threadId: 'new-chat' }), /TensorFold is quitting/, 'deferred question/device shutdown cannot admit a fresh run')
assert.throws(() => guard.assertRoomStartAllowed(['/room']), /TensorFold is quitting/)
assert.throws(() => guard.acquire({ workspaceKey: '/restore' }), /TensorFold is quitting/)
shutdownWork.release()
quitPause.release(); quitPause.release()
assert.doesNotThrow(() => guard.reserveStart({ threadId: 'recovery-after-failed-quit' }).release(), 'failed quit can reopen admission for recovery')
const updaterLease = await guard.acquireGlobal('App update installation')
const updateQuitPause = guard.pauseAdmission('TensorFold is quitting')
assert.throws(() => guard.reserveStart({ threadId: 'late-update-chat' }), /TensorFold is quitting/, 'Quit can close admission while the updater already owns its barrier')
updateQuitPause.release()
assert.throws(() => guard.reserveStart({ threadId: 'still-updating' }), /App update installation/, 'releasing Quit never removes an updater barrier')
updaterLease.release()
assert.doesNotThrow(() => guard.reserveStart({ threadId: 'after-update' }).release())

console.log('workspace mutation leases and the global purge barrier serialize starts, restores, and purge')
