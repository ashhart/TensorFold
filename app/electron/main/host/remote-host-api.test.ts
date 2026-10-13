// Run: jiti electron/main/host/remote-host-api.test.ts
//
// /v1/files directory listing confinement: a paired-device session token (the
// least-privilege tier) may only browse inside the user's home subtree; the owner token
// keeps unrestricted local browsing.
import assert from 'node:assert/strict'
import { mkdtempSync, promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Isolate persistence to a temp dir so we touch neither ~/.vontra nor an Electron app.
process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'vontra-remote-host-api-'))

const { createRemoteHostApi } = await import('./remote-host-api')
const { DEFAULT_ROOM_COLLABORATION_MODE } = await import('../../agent/room-collaboration')
const { store } = await import('../state')
const { defaultSettings } = await import('../../store/defaults')
store.settings = defaultSettings()
store.settings.activeProviderId = store.settings.providers[0].id
store.settings.activeModel = 'fixture-model'
store.settings.providers[0].models = ['fixture-model']
store.projects = []
store.threads = []
store.rooms = []

const api = createRemoteHostApi({
  sendToRenderer: () => {},
  persistProjects: () => {},
  persistThreads: () => {},
  persistRooms: () => {},
  persistTickets: () => {},
  defaultRoomCollaborationMode: DEFAULT_ROOM_COLLABORATION_MODE,
  defaultRoomWorkerBudget: { maxActive: 2, maxStarts: 4 },
  ensureProjectInstructionFiles: async () => {},
  remoteServer: () => ({ publish: () => {} }),
  aborters: () => new Map(),
  roomAborters: () => new Map(),
  emitTicket: () => {},
  emitRoom: () => {},
  abortAndWaitMissionsForArchive: () => async () => {},
  loopHost: () => ({ stopLoopsForArchive: async () => {} }),
  ticketWorkerRuntime: () => ({ isTicketStarting: () => false }),
  startThreadRun: async () => ({ runId: 'run' }),
  guideThreadRun: async () => ({ queued: false, queueDepth: 0 }),
  cancelThreadRun: () => {},
  startRoomRun: async () => ({ ok: true }),
  cancelRoomRun: () => {},
})

const home = os.homedir()

// Confined: the home root itself is listable, but exposes no parent link to navigate above.
const atHome = await api.listHostDirectories('~', { confineToHome: true })
assert.equal(atHome.ok, true)
assert.equal(atHome.path, home)
assert.equal(atHome.parent, null, 'a confined listing at the home root has no parent link')

// Confined: a subdirectory of home is listable.
const underHome = await api.listHostDirectories('~/..', { confineToHome: true }).then(
  () => { throw new Error('~/.. must be rejected when confined') },
  (err: any) => err,
)
assert.equal(underHome.code, 403, 'tilde escapes out of home are rejected when confined')
assert.match(underHome.message, /home folder/)

// Confined: absolute paths outside home are rejected.
await assert.rejects(
  () => api.listHostDirectories('/tmp', { confineToHome: true }),
  (err: any) => err.code === 403 && /home folder/.test(err.message),
)
await assert.rejects(
  () => api.listHostDirectories('/', { confineToHome: true }),
  (err: any) => err.code === 403,
)

// Unconfined (owner token): absolute paths outside home still list.
const tmp = await api.listHostDirectories('/tmp')
assert.equal(tmp.ok, true, 'the owner tier keeps unrestricted directory listing')
assert.equal(tmp.path, '/tmp')

const projectHome = await fs.mkdtemp(path.join(home, '.studio-policy-fixture-'))
const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-policy-outside-'))
try {
  const inside = path.join(projectHome, 'project')
  await fs.mkdir(inside)
  const alias = path.join(projectHome, 'alias')
  await fs.symlink(outside, alias)
  await assert.rejects(() => api.listHostDirectories(alias, { confineToHome: true }), (err: any) => err.code === 403)
  await assert.rejects(() => api.createProjectOnThisHost({ name: 'Fixture', cwd: inside, workspace: { kind: 'local', path: outside } }, { confineToHome: true }), /home folder/)
  await assert.rejects(() => api.createProjectOnThisHost({ name: 'Fixture', cwd: inside, workspace: { kind: 'local', path: alias } }, { confineToHome: true }), /home folder/)
  const project = await api.createProjectOnThisHost({ name: 'Fixture', cwd: inside }, { confineToHome: true })
  assert.equal(project.workspaceBoundary, 'home')
  assert.equal(project.workspace?.path, await fs.realpath(inside))
  const thread = await api.createThreadOnThisHost({ projectId: project.id }, { confineToHome: true })
  assert.equal(thread.workspaceBoundary, 'home')
  assert.equal(thread.cwd, project.workspace?.path)
  await assert.rejects(() => api.createThreadOnThisHost({ projectId: project.id, cwd: outside }, { confineToHome: true }), /home folder/)
  const ownerProject = await api.createProjectOnThisHost({ name: 'Outside fixture', cwd: outside })
  const ownerRoom = await api.createRoomOnThisHost({ projectId: ownerProject.id })
  await assert.rejects(() => api.steerRoomOnThisHost(ownerRoom.id, 'fixture', { confineToHome: true }), /home folder/)
  await assert.rejects(() => api.startRemoteRoomOnThisHost(ownerRoom.id, 'fixture', undefined, { confineToHome: true }), /home folder/)
  // A persisted project cannot widen later execution after its directory is replaced.
  await fs.rmdir(inside)
  await fs.symlink(outside, inside)
  await assert.rejects(() => api.startRemoteThreadOnThisHost({ threadId: thread.id, userText: 'fixture' }), /home folder/)
  await assert.rejects(() => api.createThreadOnThisHost({ projectId: project.id }), /home folder/)
} finally {
  await fs.rm(projectHome, { recursive: true, force: true })
  await fs.rm(outside, { recursive: true, force: true })
}

console.log('remote-host-api directory confinement tests passed')
