import assert from 'node:assert/strict'
import { resolveThreadInstructionPreview } from './context-preview'

let calls = 0
const resolver = async () => {
  calls++
  return { rootPath: '/local/lookalike', providerId: null, nodes: [], cycles: [], resolvedContent: 'LOCAL SECRET', tokenEstimate: 3 }
}

const remote = await resolveThreadInstructionPreview({
  cwd: '/srv/repo', providerId: null, workspace: { kind: 'ssh', hostId: 'pgx', path: '/srv/repo' },
}, resolver)
assert.equal(calls, 0, 'SSH preview does not call IPC/local instruction resolution')
assert.equal(remote.graph, null)
assert.match(remote.unavailableReason ?? '', /Remote instruction preview unavailable/)

const local = await resolveThreadInstructionPreview({ cwd: '/local/repo', providerId: 'p', workspace: null }, resolver)
assert.equal(calls, 1)
assert.match(local.graph?.resolvedContent ?? '', /LOCAL SECRET/)
assert.equal(local.unavailableReason, null)

const inheritedRemote = await resolveThreadInstructionPreview(
  { cwd: '/srv/repo', providerId: null, projectId: 'project', workspace: null },
  resolver,
  { id: 'project', cwd: '/local/lookalike', workspace: { kind: 'ssh', hostId: 'pgx', path: '/srv/repo' } },
)
assert.equal(calls, 1, 'project-inherited SSH preview also avoids the local resolver')
assert.equal(inheritedRemote.graph, null)

const folderless = await resolveThreadInstructionPreview({ cwd: '', providerId: 'p', workspace: null }, resolver)
assert.equal(calls, 1, 'a chat with no folder never asks for an instruction graph (no error toast)')
assert.deepEqual(folderless, { graph: null, unavailableReason: null })

console.log('context preview keeps SSH instruction resolution out of local IPC')
