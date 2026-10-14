// Run: jiti electron/main/host/checkpoint-runtime.test.ts
import assert from 'node:assert/strict'
import { agentOwnedPathsFromTurn, createCheckpointService, threadTurnNumber, type CheckpointBackend } from './checkpoint-runtime.ts'
import type { GitRunner } from '../../agent/workspace-checkpoints.ts'
import type { PersistedTurnCheckpoints } from '../../agent/turn-checkpoints.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const ownedTimeline: any[] = [
  { id: 'u1', kind: 'user' }, { id: 't1', kind: 'tool', state: 'ok', changedPaths: ['src/turn-1.ts'] },
  { id: 'u2', kind: 'user' }, { id: 't2', kind: 'tool', state: 'ok', changedPaths: ['src/turn-2.ts'] },
  { id: 'u3', kind: 'user' }, { id: 't3', kind: 'tool', state: 'ok', changedPaths: ['src/turn-3.ts'] },
]

interface Call { command: string; cwd?: string }

// Fake runner: records commands, replies from a matcher covering every probe the
// checkpoint ops make. No subprocess / git / network.
function fakeRunner(reply: (command: string) => { code?: number; stdout?: string; stderr?: string }) {
  const calls: Call[] = []
  const runner: GitRunner = {
    async exec(command, options) {
      calls.push({ command, cwd: options?.cwd })
      const r = reply(command)
      return { code: r.code ?? 0, stdout: r.stdout ?? '', stderr: r.stderr ?? '' }
    },
  }
  return { runner, calls }
}

function captureReply(command: string) {
  if (command.includes('--is-inside-work-tree')) return { stdout: 'true' }
  if (command.includes('rev-parse --verify --quiet HEAD')) return { stdout: 'deadbeefHEAD' }
  if (command.includes('--git-common-dir')) return { stdout: '/repo/.git' }
  if (command.includes('write-tree')) return { stdout: 'tree111' }
  if (command.includes('commit-tree')) return { stdout: 'commit222' }
  if (command.includes('^{commit}')) return { stdout: 'res333' }
  if (command.includes('for-each-ref')) return { stdout: 'refs/vontra/checkpoints/T/turn/1\nrefs/vontra/checkpoints/T/turn/2\n' }
  if (command.includes('git cat-file -t')) return { stdout: 'blob\n' }
  if (command.includes('git diff')) return { stdout: 'LIVE-DIFF' }
  return {}
}

console.log('checkpoint-runtime tests:')

{
  // Turn number = count of user messages (assistant/tool/notice items don't count).
  const timeline: any[] = [
    { kind: 'user', text: 'a' },
    { kind: 'assistant', text: 'b' },
    { kind: 'tool', text: 'c' },
    { kind: 'user', text: 'd' },
  ]
  assert.equal(threadTurnNumber(timeline), 2)
  assert.equal(threadTurnNumber([]), 0)
  ok('threadTurnNumber: counts only user items')
}

{
  assert.deepEqual(agentOwnedPathsFromTurn(ownedTimeline, 2), ['src/turn-2.ts', 'src/turn-3.ts'])
  assert.deepEqual(agentOwnedPathsFromTurn([
    { id: 'u', kind: 'user' },
    { id: 'failed-no-mutation', kind: 'tool', state: 'error' },
    { id: 'failed-mutated', kind: 'tool', state: 'error', changedPaths: ['partial-write.txt'] },
    { id: 'forged', kind: 'assistant', state: 'ok', changedPaths: ['whole-tree'] },
    { id: 'legacy', kind: 'tool', state: 'ok' },
  ] as any, 1), ['partial-write.txt'], 'failed tools retain only explicit host-recorded mutation evidence')
  ok('agentOwnedPathsFromTurn scopes restore to host-recorded mutations, including failed partial writes')
}

{
  // capture on a LOCAL workspace uses the injected local runner + resolved cwd,
  // and drives createCheckpoint (writes only the hidden turn ref).
  const { runner, calls } = fakeRunner(captureReply)
  const svc = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    localRunner: runner,
  })
  const res = await svc.capture('T', 3, { cwd: '/repo', kind: 'task' })
  assert.equal(res.ok, true)
  assert.equal(res.sha, 'commit222')
  const cmds = calls.map((c) => c.command)
  const updateRef = cmds.find((c) => c.includes('update-ref') && !c.includes('-d'))
  assert.ok(updateRef && updateRef.includes("'refs/vontra/checkpoints/T/turn/3'"), 'writes the hidden turn ref')
  assert.ok(calls.every((c) => c.cwd === '/repo'), 'runs against the resolved local cwd')
  ok('capture: local workspace -> createCheckpoint on the hidden turn ref')
}

{
  // capture prefers a connected SSH backend runner + its remote cwd over local.
  const { runner: backendRunner, calls } = fakeRunner(captureReply)
  const backend: CheckpointBackend = Object.assign(backendRunner, { rootCwd: '/remote/root' })
  let localUsed = false
  const svc = createCheckpointService({
    resolveBackend: () => backend,
    resolveCwd: async () => { localUsed = true; return '/local' },
    localRunner: fakeRunner(captureReply).runner,
  })
  const res = await svc.capture('T', 1, { workspace: { kind: 'ssh', hostId: 'h', path: '/remote/proj' } as any, kind: 'task' })
  assert.equal(res.ok, true)
  assert.equal(localUsed, false, 'local cwd resolver not used for an SSH backend')
  assert.ok(calls.every((c) => c.cwd === '/remote/proj'), 'runs against the workspace remote path')
  ok('capture: SSH backend -> backend runner + remote path')
}

{
  // A disconnected SSH target is still remote. It must never be reinterpreted as a local cwd
  // merely because backendForWorkspace has no live session after restart/disconnect.
  const { runner: localRunner, calls: localCalls } = fakeRunner(captureReply)
  let localResolved = false
  const svc = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => { localResolved = true; return '/remote/proj' },
    localRunner,
  })
  const target = { workspace: { kind: 'ssh', hostId: 'h', path: '/remote/proj' } as any, kind: 'task' as const }
  const captured = await svc.capture('T', 1, target)
  assert.equal(captured.ok, false)
  assert.match(captured.error ?? '', /SSH workspace is disconnected/)
  assert.equal(localResolved, false, 'disconnected SSH checkpoint never resolves a local cwd')
  assert.equal(localCalls.length, 0, 'disconnected SSH checkpoint never invokes the local Git runner')
  await assert.rejects(() => svc.undoLast('T', target), /SSH workspace is disconnected/)
  assert.equal(localCalls.length, 0, 'disconnected SSH restore never invokes the local Git runner')
  ok('disconnected SSH checkpoint/restore fails closed without local fallback')
}

{
  // chat threads are never checkpointed (no code workspace).
  const { calls } = fakeRunner(captureReply)
  const svc = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    localRunner: fakeRunner(captureReply).runner,
  })
  const res = await svc.capture('T', 1, { cwd: '/repo', kind: 'chat' })
  assert.equal(res.ok, false)
  assert.equal(calls.length, 0, 'no git commands for a chat thread')
  ok('capture: chat thread -> skipped, no git')
}

{
  // capture is best-effort: a throwing cwd resolver yields ok:false, never throws.
  const svc = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => { throw new Error('bad cwd') },
    localRunner: fakeRunner(captureReply).runner,
  })
  const res = await svc.capture('T', 1, { cwd: '/nope', kind: 'task' })
  assert.equal(res.ok, false)
  assert.ok((res.error || '').includes('bad cwd'))
  ok('capture: resolver throws -> ok:false, no throw')
}

{
  // non-git workspace -> ok:false, no checkpoint write.
  const { runner, calls } = fakeRunner((command) => (command.includes('--is-inside-work-tree') ? { code: 128 } : captureReply(command)))
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/x', localRunner: runner })
  const res = await svc.capture('T', 1, { cwd: '/x', kind: 'task' })
  assert.equal(res.ok, false)
  assert.ok(!calls.some((c) => c.command.includes('update-ref')), 'no ref written outside a repo')
  ok('capture: non-git workspace -> ok:false')
}

{
  // undoLast restores the HIGHEST turn then deletes that ref (progressive undo).
  const { runner, calls } = fakeRunner(captureReply)
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/repo', localRunner: runner })
  const res = await svc.undoLast('T', { cwd: '/repo', kind: 'task', timeline: ownedTimeline })
  assert.equal(res.ok, true)
  assert.equal(res.turn, 2, 'undoes the most recent turn')
  const cmds = calls.map((c) => c.command)
  assert.ok(cmds.some((c) => /git --literal-pathspecs restore --source res333 --worktree --staged/.test(c)), 'restores only recorded agent-owned paths')
  assert.ok(cmds.some((c) => c.includes("update-ref -d 'refs/vontra/checkpoints/T/turn/2'")), 'drops the consumed checkpoint')
  ok('undoLast: restore highest turn + delete its ref')
}

{
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes('git cat-file -t')) return { code: 1 }
    return captureReply(command)
  })
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/repo', localRunner: runner })
  const res = await svc.undoLast('T', { cwd: '/repo', kind: 'task', timeline: ownedTimeline })
  assert.equal(res.ok, true)
  assert.match(res.warning ?? '', /Preserved 2 post-checkpoint paths/)
  assert.deepEqual(res.preservedPaths, ['src/turn-2.ts', 'src/turn-3.ts'])
  assert.ok(!calls.some((c) => c.command.includes(' restore --source ')), 'a new path with no checkpoint blob is never deleted/restored')
  ok('undoLast: preserved post-checkpoint paths return an explicit warning')
}

{
  // undoLast with no checkpoints -> ok:false, nothing restored.
  const { runner, calls } = fakeRunner((command) => (command.includes('for-each-ref') ? { stdout: '' } : captureReply(command)))
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/repo', localRunner: runner })
  const res = await svc.undoLast('T', { cwd: '/repo', kind: 'task' })
  assert.equal(res.ok, false)
  assert.ok(!calls.some((c) => c.command.includes('git restore')), 'nothing restored')
  ok('undoLast: no checkpoints -> ok:false')
}

{
  // diffLast diffs the highest turn against the live worktree.
  const { runner } = fakeRunner(captureReply)
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/repo', localRunner: runner })
  const res = await svc.diffLast('T', { cwd: '/repo', kind: 'task' })
  assert.equal(res.ok, true)
  assert.equal(res.turn, 2)
  assert.equal(res.diff, 'LIVE-DIFF')
  ok('diffLast: diff highest turn vs worktree')
}

{
  // listTurns passes through the enumerated turns.
  const { runner } = fakeRunner(captureReply)
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/repo', localRunner: runner })
  assert.deepEqual(await svc.listTurns('T', { cwd: '/repo', kind: 'task' }), [1, 2])
  ok('listTurns: passthrough of enumerated turns')
}

// ---- unified turn checkpoints (VST-036): files + conversation ----
{
  const { runner, calls } = fakeRunner(captureReply)
  const svc = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    localRunner: runner,
    conversationDeps: { clock: { now: () => 1_000 }, newId: (() => { let n = 0; return () => `cvk_${++n}` })() },
  })
  const target = { cwd: '/repo', kind: 'task' as const, timeline: ownedTimeline }

  svc.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'first' }, { role: 'assistant', content: 'A' }], timeline: [{ id: 'i1', kind: 'user' }] }, 'sha1')
  svc.recordTurn('T', 2, { apiMessages: [{ role: 'user', content: 'first' }, { role: 'assistant', content: 'A' }, { role: 'user', content: 'second' }, { role: 'assistant', content: 'B' }], timeline: [{ id: 'i1', kind: 'user' }, { id: 'i2', kind: 'user' }] }, 'sha2')

  const turns = svc.turnCheckpoints('T')
  assert.deepEqual(turns.map((t) => t.turn), [1, 2], 'both turns recorded, ordered')
  assert.equal(turns[0].fileRef, 'refs/vontra/checkpoints/T/turn/1', 'file ref derived from the turn')
  assert.equal(turns[0].fileSha, 'sha1')

  const revert = await svc.revertTurn('T', 1, target)
  assert.equal(revert.ok, true)
  assert.equal(revert.conversation?.apiMessages.length, 2, 'conversation restored to turn 1 (later messages dropped)')
  assert.deepEqual((revert.conversation?.apiMessages as any[]).map((m) => m.content), ['first', 'A'])
  assert.ok(calls.some((c) => /git --literal-pathspecs restore/.test(c.command)), 'a git restore was issued for the file side')

  const bad = await svc.revertTurn('T', 9, target)
  assert.equal(bad.ok, false)
  assert.match(bad.error ?? '', /no turn checkpoint/)
  ok('revertTurn: restores files + conversation together, guards unknown turns')
}

{
  const { runner } = fakeRunner(captureReply)
  const svc = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    localRunner: runner,
    conversationDeps: { clock: { now: () => 1_000 }, newId: (() => { let n = 0; return () => `cvk_${++n}` })() },
  })
  svc.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'shared' }], timeline: [] }, 'sha1')
  const fork = await svc.forkTurn('T', 1, { cwd: '/repo', kind: 'task' }, { label: 'alt' })
  assert.equal(fork.ok, true)
  assert.notEqual(fork.fork?.branch, 'main', 'fork opens a new branch')
  assert.equal(fork.fork?.fileRef, 'refs/vontra/checkpoints/T/turn/1')
  assert.equal(fork.worktreePath, undefined, 'no worktree unless asked for')
  fork.fork!.conversation.apiMessages.push({ role: 'user', content: 'only in fork' } as any)
  const revert = await svc.revertTurn('T', 1, { cwd: '/repo', kind: 'task', timeline: ownedTimeline })
  assert.equal(revert.conversation?.apiMessages.length, 1, 'the parent turn-1 checkpoint is untouched by fork mutation')
  assert.equal((await svc.forkTurn('T', 5, { kind: 'chat' })).ok, false, 'forking an unknown turn fails')
  ok('forkTurn: isolated clone; parent untouched; guards unknown turns')
}

{
  // Fork WORKTREE (VST-036): worktree:true materializes the fork point into its own
  // detached worktree via `git worktree add`, homed at the injected destination.
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes('worktree add')) return { code: 0 }
    return captureReply(command)
  })
  const svc = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    localRunner: runner,
    forkWorktreePath: ({ threadId, turn, branch }) => `/worktrees/${threadId}-t${turn}-${branch}`,
  })
  svc.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'x' }], timeline: [] }, 'sha1')
  const fork = await svc.forkTurn('T', 1, { cwd: '/repo', kind: 'task' }, { worktree: true })
  assert.equal(fork.ok, true)
  assert.equal(fork.worktreeError, undefined)
  assert.equal(fork.worktreePath, `/worktrees/T-t1-${fork.fork!.branch}`)
  const add = calls.map((c) => c.command).find((c) => c.includes('worktree add'))
  assert.ok(add?.includes('--detach') && add.includes('res333'), 'detached worktree at the checkpoint commit')
  ok('forkTurn: worktree fork adds a detached worktree at the snapshot commit')
}

{
  // Worktree failure modes degrade to the shared-cwd fork with an explicit reason —
  // the fork itself must never fail because of the worktree.
  const disk = { persistence: { load: async () => null, save: () => {} } }
  // 1) SSH backend → local-only.
  const backend = Object.assign(fakeRunner(captureReply).runner, { rootCwd: '/remote' }) as CheckpointBackend
  const sshSvc = createCheckpointService({
    resolveBackend: () => backend,
    resolveCwd: async () => '/repo',
    localRunner: fakeRunner(captureReply).runner,
    forkWorktreePath: () => '/worktrees/x',
    persistence: disk.persistence,
  })
  sshSvc.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'x' }], timeline: [] })
  const ssh = await sshSvc.forkTurn('T', 1, { workspace: { kind: 'ssh', hostId: 'h', path: '/remote/p' } as any, kind: 'task' }, { worktree: true })
  assert.equal(ssh.ok, true)
  assert.match(ssh.worktreeError ?? '', /local-only/)
  // 2) No destination factory configured.
  const bare = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/repo', localRunner: fakeRunner(captureReply).runner })
  bare.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'x' }], timeline: [] })
  const noDep = await bare.forkTurn('T', 1, { cwd: '/repo', kind: 'task' }, { worktree: true })
  assert.equal(noDep.ok, true)
  assert.match(noDep.worktreeError ?? '', /no fork worktree directory/)
  // 3) Not a git repository.
  const noRepo = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/plain',
    localRunner: fakeRunner((c) => (c.includes('--is-inside-work-tree') ? { stdout: 'false' } : {})).runner,
    forkWorktreePath: () => '/worktrees/x',
  })
  noRepo.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'x' }], timeline: [] }, 'sha1')
  const plain = await noRepo.forkTurn('T', 1, { cwd: '/plain', kind: 'task' }, { worktree: true })
  assert.equal(plain.ok, true)
  assert.match(plain.worktreeError ?? '', /no git repository/)
  ok('forkTurn: worktree failures degrade with explicit reasons, fork survives')
}

{
  // A chat thread (no workspace) reverts conversation-only — no git restore attempted.
  const { runner, calls } = fakeRunner(() => ({}))
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '', localRunner: runner })
  svc.recordTurn('C', 1, { apiMessages: [{ role: 'user', content: 'hi' }], timeline: [] })
  const revert = await svc.revertTurn('C', 1, { kind: 'chat' })
  assert.equal(revert.ok, true)
  assert.equal(revert.conversation?.apiMessages.length, 1, 'chat revert restores conversation only')
  assert.equal(calls.length, 0, 'no git commands for a chat revert')
  ok('revertTurn: chat threads revert conversation-only, no git')
}

{
  const { runner, calls } = fakeRunner(captureReply)
  const svc = createCheckpointService({ resolveBackend: () => undefined, resolveCwd: async () => '/repo', localRunner: runner })
  svc.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'hi' }], timeline: [] }, null)
  assert.equal(svc.turnCheckpoints('T')[0].fileRef, null, 'a failed/absent capture never fabricates a file ref')
  const revert = await svc.revertTurn('T', 1, { cwd: '/repo', kind: 'task', timeline: ownedTimeline })
  assert.equal(revert.ok, true)
  assert.match(revert.warning ?? '', /No file checkpoint was captured/)
  assert.ok(!calls.some((c) => c.command.includes(' restore --source ')), 'conversation-only record never attempts a missing git ref')
  ok('revertTurn: absent file capture is honest conversation-only state with warning')
}

// ---- ledger persistence (VST-036): ledgers + plan links survive a restart ----

// In-memory "disk" that mimics the real persister: a JSON round-trip per save, so
// anything that wouldn't survive serialization fails here too.
function fakeDisk() {
  let disk: PersistedTurnCheckpoints | null = null
  return {
    read: () => disk,
    persistence: {
      load: async () => disk,
      save: (state: PersistedTurnCheckpoints) => { disk = JSON.parse(JSON.stringify(state)) as PersistedTurnCheckpoints },
    },
  }
}

function persistedSvc(disk: ReturnType<typeof fakeDisk>, opts: { persistable?: (id: string) => boolean; maxTurns?: number } = {}) {
  return createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    localRunner: fakeRunner(captureReply).runner,
    conversationDeps: { clock: { now: () => 1_000 }, newId: (() => { let n = 0; return () => `p_${++n}_${Math.random().toString(36).slice(2, 6)}` })() },
    persistence: disk.persistence,
    isPersistableThread: opts.persistable,
    maxPersistedTurnsPerThread: opts.maxTurns,
  })
}

{
  // recordTurn writes a bounded, filtered state; a fresh service restores it and can
  // still revert files+conversation — the full restart round-trip.
  const disk = fakeDisk()
  const svc = persistedSvc(disk, { persistable: (id) => id !== 'GHOST', maxTurns: 2 })
  const msgs = (n: number) => ({ apiMessages: Array.from({ length: n }, (_, i) => ({ role: 'user', content: `m${i + 1}` })), timeline: [] })
  svc.recordTurn('T', 1, msgs(1))
  svc.recordTurn('T', 2, msgs(2))
  svc.recordTurn('T', 3, msgs(3))
  svc.recordTurn('GHOST', 1, msgs(1)) // incognito/deleted → filtered by the persistable predicate

  const saved = disk.read()
  assert.ok(saved, 'a save landed')
  assert.deepEqual(saved!.ledgers.T.entries.map((e) => e.turn.turn), [2, 3], 'disk keeps only the newest maxTurns turns')
  assert.equal(saved!.ledgers.GHOST, undefined, 'non-persistable threads never touch disk')

  const svc2 = persistedSvc(disk, {})
  const restored = await svc2.restorePersisted()
  assert.equal(restored.threads, 1)
  assert.equal(restored.turns, 2)
  const revert = await svc2.revertTurn('T', 3, { cwd: '/repo', kind: 'task', timeline: ownedTimeline })
  assert.equal(revert.ok, true)
  assert.equal(revert.conversation?.apiMessages.length, 3, 'a restored ledger reverts the conversation across a restart')
  ok('persistence: record → restart → restore → revert round-trip, bounded + filtered')
}

{
  // restorePersisted applies the persistable filter at LOAD too (prunes deleted threads)
  // and an unreadable disk restores to empty instead of failing.
  const disk = fakeDisk()
  persistedSvc(disk).recordTurn('DEAD', 1, { apiMessages: [{ role: 'user', content: 'x' }], timeline: [] })
  const svc = persistedSvc(disk, { persistable: () => false })
  const restored = await svc.restorePersisted()
  assert.equal(restored.threads, 0, 'ledgers of non-persistable threads are pruned at load')
  const broken = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    localRunner: fakeRunner(captureReply).runner,
    persistence: { load: async () => { throw new Error('disk on fire') }, save: () => {} },
  })
  assert.deepEqual(await broken.restorePersisted(), { threads: 0, turns: 0, links: 0 }, 'a failing load degrades to empty, never throws')
  ok('persistence: load-time pruning + unreadable disk degrades to empty')
}

{
  // adoptFork seeds the forked thread's ledger at the fork point (and persists it),
  // so the new thread can immediately revert to where it diverged.
  const disk = fakeDisk()
  const svc = persistedSvc(disk)
  svc.recordTurn('SRC', 1, { apiMessages: [{ role: 'user', content: 'shared' }], timeline: [] })
  const fork = await svc.forkTurn('SRC', 1, { kind: 'chat' }, { label: 'alt take' })
  assert.equal(fork.ok, true)
  const adopted = svc.adoptFork('NEW', fork.fork!)
  assert.equal(adopted.turn, 1)
  assert.equal(adopted.threadId, 'NEW')
  assert.equal(svc.turnCheckpoints('NEW').length, 1, 'the fork point is the new thread\'s first checkpoint')
  const revert = await svc.revertTurn('NEW', 1, { kind: 'chat' })
  assert.deepEqual((revert.conversation?.apiMessages as any[]).map((m) => m.content), ['shared'])
  assert.ok(disk.read()?.ledgers.NEW, 'the adopted ledger persists')
  ok('adoptFork: forked thread can revert to its fork point, persisted')
}

{
  // Plan → implementation links persist alongside ledgers, restore once, and dedupe.
  const disk = fakeDisk()
  const svc = persistedSvc(disk)
  svc.recordPlanLink({ planId: 'p1', planThreadId: 'PLAN', implementationThreadId: 'IMPL', ticketId: 'tk1' })
  assert.equal(disk.read()?.planLinks.length, 1)
  const svc2 = persistedSvc(disk)
  await svc2.restorePersisted()
  await svc2.restorePersisted() // double restore must not duplicate
  const links = svc2.planImplementationLinks()
  assert.equal(links.length, 1)
  assert.equal(links[0].planId, 'p1')
  assert.equal(links[0].ticketId, 'tk1')
  assert.throws(() => svc2.recordPlanLink({ planId: 'p2', planThreadId: 'X', implementationThreadId: 'X' }), /own plan thread/, 'self-implementation guard still applies through the service')
  ok('plan links: persisted, restored once, deduped, guarded')
}

{
  // Security purge must clear both live structures and persist the canonical empty wire
  // shape synchronously; unlike ordinary best-effort checkpoint writes, a failure surfaces.
  const disk = fakeDisk()
  const svc = persistedSvc(disk)
  svc.recordTurn('T', 1, { apiMessages: [{ role: 'user', content: 'secret' }], timeline: [] })
  svc.recordPlanLink({ planId: 'p1', planThreadId: 'PLAN', implementationThreadId: 'IMPL' })
  assert.equal(svc.recordCount(), 2, 'count includes turn snapshots and plan links')
  await svc.clearAll()
  assert.equal(svc.recordCount(), 0, 'clearAll removes every live checkpoint/link')
  assert.deepEqual(disk.read(), { version: 1, ledgers: {}, planLinks: [] }, 'clearAll durably writes the canonical empty state')

  const broken = createCheckpointService({
    resolveBackend: () => undefined,
    resolveCwd: async () => '/repo',
    persistence: { load: async () => null, save: async () => { throw new Error('disk full') } },
  })
  await assert.rejects(() => broken.clearAll(), /disk full/, 'security purge can surface checkpoint persistence failures')
  ok('clearAll: live + durable checkpoint erasure, failures surface')
}

console.log(`\n${passed} passed`)
