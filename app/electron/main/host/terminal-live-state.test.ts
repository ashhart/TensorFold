import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { clearLiveTerminalSessions, liveTerminalActivity, registerLiveTerminalSession, removeLiveTerminalSession } from './terminal-live-state'
import { spawnLocalPtyForProject } from './terminal-runtime'

clearLiveTerminalSessions()

let spawnCalls = 0
const fakeSpawn = () => { spawnCalls++; return {} }
assert.throws(() => spawnLocalPtyForProject({ workspace: { kind: 'ssh', hostId: 'pgx', path: '/srv/repo' } } as any, fakeSpawn), /do not support SSH workspaces/)
assert.throws(() => spawnLocalPtyForProject({ workspace: null, archivedAt: 1 } as any, fakeSpawn), /Restore the archived project/)
assert.equal(spawnCalls, 0, 'SSH and archived projects fail before node-pty spawn')
const stalePersisted = { id: 'stale', projectId: 'p', cliId: 'shell', title: 'Stale', status: 'busy', createdAt: 1 } as const
const live = { id: 'live', projectId: 'p', cliId: 'shell', title: 'Live', status: 'idle' as 'idle' | 'busy' | 'attention' | 'done', createdAt: 2 }
void stalePersisted // Deliberately never registered: persisted rows are not live PTYs.
registerLiveTerminalSession(live)
assert.deepEqual(liveTerminalActivity(), { total: 1, running: 0, attention: 0 })
live.status = 'attention'
assert.deepEqual(liveTerminalActivity(), { total: 1, running: 0, attention: 1 }, 'summary reads the live session object, not a persisted copy')
removeLiveTerminalSession(live.id)
assert.deepEqual(liveTerminalActivity(), { total: 0, running: 0, attention: 0 })
clearLiveTerminalSessions()

const runtimeSource = readFileSync(new URL('./terminal-runtime.ts', import.meta.url), 'utf8')
assert.ok((runtimeSource.match(/spawnSession\(cli, project/g) ?? []).length >= 3, 'terminal start, restore, and remoteApi start share the guarded spawn path')
const archiveSource = readFileSync(new URL('./archive.ts', import.meta.url), 'utf8')
assert.match(archiveSource, /abortProjectTerminals\?\.\(id\)/, 'project archive stops already-live project terminals')
const resizeHandler = runtimeSource.match(/handle\('terminal:resize',[\s\S]*?\n\s*}\)/)?.[0] ?? ''
assert.match(resizeHandler, /requireSecurityRole\(\s*'owner'\s*,\s*'resize a project terminal'\s*\)/, 'terminal resize is owner-gated like every other PTY mutation')
assert.doesNotMatch(runtimeSource, /\.pty\.kill\(/, 'terminal teardown never signals only the PTY leader')
assert.match(runtimeSource, /terminatePtyProcess\(t\.pty, \{ force: true \}\)/, 'app shutdown and security purge synchronously reap the PTY process group')
assert.ok((runtimeSource.match(/terminatePtyProcess\(t\.pty\)/g) ?? []).length >= 2, 'local and remote user kills use bounded process-group termination')
assert.match(runtimeSource, /detached:\s*detachedProcessGroup\(\)/, 'CLI version probes launch in a reapable POSIX process group')
assert.match(runtimeSource, /terminateChildProcess\(child, \{ processGroup: detachedProcessGroup\(\) \}\)/, 'timed-out CLI version probes reap descendants')

console.log('terminal live-state tests passed')
