import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const remoteControlIpc = readFileSync(new URL('./ipc/remote-control-ipc.ts', import.meta.url), 'utf8')
// The monolithic store.ts was decomposed into per-domain slices under src/lib/store/slices/.
// Reassemble the relevant slice bodies into one string so the projection-boundary assertions
// can keep slicing between methods the same way they did against the old barrel file.
const sliceFiles = ['projectsSlice.ts', 'threadsSlice.ts', 'roomsSlice.ts', 'ticketsSlice.ts', 'automationsSlice.ts'] as const
const store = sliceFiles.map((name) => readFileSync(new URL(`../../../src/lib/store/slices/${name}`, import.meta.url), 'utf8')).join('\n')

const dispatch = remoteControlIpc.slice(remoteControlIpc.indexOf('function dispatchRemoteSnapshot'), remoteControlIpc.indexOf('function ensureRemoteEventStream'))
assert.match(dispatch, /orchestratorHostId: host\.id[\s\S]*trace:/, 'remote agent events carry their immutable source host')
assert.match(dispatch, /room:event', \{ \.\.\.data, orchestratorHostId: host\.id \}/, 'remote room events carry their immutable source host')
assert.doesNotMatch(dispatch, /webContents\.send\('job:update'/, 'remote jobs are not merged into locally actionable state')
assert.doesNotMatch(dispatch, /webContents\.send\('automation:update'/, 'remote automations are not merged into locally actionable state')

for (const method of ['applyProjectUpdate', 'applyThreadUpdate', 'applyRoomUpdate', 'applyTicketUpdate']) {
  const start = store.lastIndexOf(`  ${method}(`)
  const end = store.indexOf('\n  },', start)
  assert.match(store.slice(start, end), /projectionOriginsMatch/, `${method} rejects cross-origin raw-id collisions`)
}
assert.match(store.slice(store.indexOf('applyEvent(e: AgentEvent)'), store.indexOf('\n  async createAutomation')), /orchestratorHostId/, 'agent events are origin checked')
assert.match(store.slice(store.indexOf('applyRoomEvent(e)'), store.indexOf('\n  async createTicket')), /orchestratorHostId/, 'room events are origin checked')

console.log('origin-projection-boundary.test.ts passed')
