import assert from 'node:assert/strict'
import { effectiveThreadCwd, effectiveThreadWorkspace } from './thread-workspace'

const project = { id: 'project', cwd: '/local/lookalike', workspace: { kind: 'ssh' as const, hostId: 'pgx', path: '/srv/repo' } }
const inherited = { id: 'thread', projectId: 'project', cwd: '/local/lookalike', workspace: null }
assert.deepEqual(effectiveThreadWorkspace(inherited as any, project), project.workspace)
assert.equal(effectiveThreadCwd(inherited as any, project), '/srv/repo', 'inherited SSH path outranks a same-named/local legacy cwd')

const override = { ...inherited, workspace: { kind: 'local' as const, path: '/worktree' }, cwd: '/worktree' }
assert.deepEqual(effectiveThreadWorkspace(override as any, project), override.workspace)
assert.equal(effectiveThreadCwd(override as any, project), '/worktree')

assert.equal(effectiveThreadWorkspace({ ...inherited, projectId: 'other' } as any, project), null, 'a mismatched project cannot lend its workspace')

console.log('thread workspace UI helpers mirror host project fallback without local SSH lookalikes')
