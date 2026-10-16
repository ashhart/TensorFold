import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { assertMissionWorkspaceSupported, registerMissionApprovalResolver } from './mission-runtime'

const localProject = { id: 'p', workspace: { kind: 'local', path: '/repo' }, archivedAt: null } as any
const localThread = { id: 't', projectId: 'p', workspace: { kind: 'local', path: '/repo' }, archivedAt: null } as any
const state = { projects: [localProject], threads: [localThread] }

assert.doesNotThrow(() => assertMissionWorkspaceSupported({ projectId: 'p', threadId: 't' }, state))
assert.throws(() => assertMissionWorkspaceSupported({ projectId: 'missing', threadId: null }, state), /project not found/i)
assert.throws(() => assertMissionWorkspaceSupported({ projectId: null, threadId: 'missing' }, state), /source thread not found/i)
assert.throws(
  () => assertMissionWorkspaceSupported({ projectId: 'p', threadId: 'projectless' }, {
    projects: [localProject],
    threads: [{ id: 'projectless', projectId: null, workspace: null, archivedAt: null } as any],
  }),
  /does not match/i,
)
assert.throws(
  () => assertMissionWorkspaceSupported({ projectId: 'ssh', threadId: null }, {
    projects: [{ id: 'ssh', workspace: { kind: 'ssh', hostId: 'h', path: '/repo' }, archivedAt: null } as any],
    threads: [],
  }),
  /does not support SSH/i,
)

const missionsIpc = readFileSync(fileURLToPath(new URL('./ipc/missions-ipc.ts', import.meta.url)), 'utf8')
const start = missionsIpc.indexOf("handle('missions:create'")
const end = missionsIpc.indexOf("handle('prLoop:createLocal'", start)
assert.ok(start >= 0 && end > start, 'missions:create handler remains discoverable')
const handler = missionsIpc.slice(start, end)
assert.ok(
  handler.indexOf('assertMissionWorkspaceSupported') < handler.indexOf('createMission('),
  'mission source ownership is validated before mission construction',
)
assert.ok(
  handler.indexOf('assertMissionWorkspaceSupported') < handler.indexOf('store.missions.unshift'),
  'mission source ownership is validated before durable persistence',
)
assert.match(handler, /projectId:\s*input\.projectId\s*\?\?\s*null/, 'the helper validates renderer-supplied project ownership')
assert.match(handler, /const projectId = input\.projectId \?\? sourceThread\?\.projectId \?\? null/, 'omitted project ownership is derived from the source thread')

console.log('mission creation rejects invalid or unsupported ownership before persistence')

// A queued fan-out approval request starts executing only after an earlier approval
// settles — possibly via the mission's abort. An already-aborted signal never fires its
// abort listener, so the resolver must settle deny immediately instead of leaving the
// run's await requestApproval() hung forever with the mission unrecoverable.
{
  const controller = new AbortController()
  const resolvers = new Map<string, (decision: 'allow' | 'allow_always' | 'deny') => void>()
  const meta = new Map<string, { threadId: string }>()
  controller.abort()
  let decision: string | null = null
  registerMissionApprovalResolver('call-1', controller, resolvers, meta, (d) => { decision = d })
  assert.equal(decision, 'deny', 'already-aborted mission approval must settle deny immediately')
  assert.equal(resolvers.size, 0, 'settled resolver must deregister itself')
  assert.equal(meta.size, 0, 'settled resolver must clear pending approval meta')
  console.log('already-aborted mission approvals settle deny instead of hanging the run')
}
