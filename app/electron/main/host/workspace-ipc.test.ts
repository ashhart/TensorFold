import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { checkpointWorkspaceKey, hostCheckpointTarget, parsePorcelainZ, parseGitNumstat, requireNoRunningLocalWriterAtWorkspace, requireNoRunningWriterAtWorkspaceKey, requireThreadStopped, sameCheckpointTarget } from './workspace-ipc'

{
  const timeline: any[] = [{ id: 'tool-1', kind: 'tool', state: 'ok', changedPaths: ['host-owned.ts'] }]
  const threads: any[] = [{
    id: 'thread-1', cwd: '/host/repo', workspace: { kind: 'ssh', hostId: 'host-1', path: '/remote/repo' }, kind: 'task', timeline,
  }]
  const target = hostCheckpointTarget(threads, 'thread-1')
  assert.equal(target?.cwd, '/host/repo')
  assert.deepEqual(target?.workspace, { kind: 'ssh', hostId: 'host-1', path: '/remote/repo' })
  assert.deepEqual(target?.timeline?.[0].changedPaths, ['host-owned.ts'])
  timeline[0].changedPaths.push('.')
  assert.deepEqual(target?.timeline?.[0].changedPaths, ['host-owned.ts'], 'target timeline is snapshotted at the host boundary')
  assert.equal(hostCheckpointTarget(threads, 'missing'), null, 'a missing thread cannot fall back to renderer-supplied cwd/workspace')
  assert.equal(hostCheckpointTarget([{ ...threads[0], archivedAt: 1 }], 'thread-1'), null, 'archived thread ownership cannot be checkpointed/restored')
  assert.equal(sameCheckpointTarget(target!, { ...target!, cwd: '/other' }), false, 'a cwd swap invalidates the authorized checkpoint target')
  assert.equal(sameCheckpointTarget(target!, { ...target!, workspace: { kind: 'ssh', hostId: 'host-2', path: '/remote/repo' } }), false, 'an SSH host swap invalidates the authorized checkpoint target')
  assert.equal(sameCheckpointTarget(target!, { ...target!, timeline: [] }), true, 'timeline growth does not retarget the filesystem authorization')
  const inherited = hostCheckpointTarget([
    { id: 'inherited', projectId: 'remote-project', cwd: '/srv/repo', workspace: null, kind: 'task', timeline: [] } as any,
  ], 'inherited', [
    { id: 'remote-project', cwd: '/local/lookalike', workspace: { kind: 'ssh', hostId: 'pgx', path: '/srv/repo' } } as any,
  ])
  assert.deepEqual(inherited?.workspace, { kind: 'ssh', hostId: 'pgx', path: '/srv/repo' }, 'checkpoint target inherits the host-owned project workspace when the thread has no override')
  assert.equal(hostCheckpointTarget([
    { id: 'archived-owner', projectId: 'archived-project', cwd: '/repo', kind: 'task', timeline: [] } as any,
  ], 'archived-owner', [{ id: 'archived-project', archivedAt: 1 } as any]), null, 'an archived project cannot lend a checkpoint target to an active-looking thread')
  console.log('ok: checkpoint target is host-owned and missing threads fail closed')
}

{
  assert.throws(() => requireThreadStopped(undefined, 'thread-1'), /run-state checks are unavailable/, 'missing host run-state wiring fails closed')
  assert.throws(() => requireThreadStopped(() => true, 'thread-1'), /Stop the running thread/, 'a running checkpoint owner cannot be restored')
  assert.doesNotThrow(() => requireThreadStopped(() => false, 'thread-1'))

  const local = { id: 'local-running', cwd: '/repo', workspace: null } as any
  const ssh = { id: 'ssh-running', cwd: '/repo', workspace: { kind: 'ssh', hostId: 'h', path: '/repo' } } as any
  await assert.rejects(
    () => requireNoRunningLocalWriterAtWorkspace([local], [], '/repo', (id) => id === 'local-running', () => false),
    /Stop active threads/,
    'a full-file restore rejects a running local thread sharing its workspace',
  )
  await assert.doesNotReject(
    () => requireNoRunningLocalWriterAtWorkspace([ssh], [], '/repo', () => true, () => false),
    'an SSH thread with a coincidentally equal path is not a local workspace owner',
  )
  const localRoom = { id: 'room-running', workspace: null, rootCwd: '/repo', workspaceCwd: null, scratchpadCwd: null } as any
  await assert.rejects(
    () => requireNoRunningLocalWriterAtWorkspace([], [localRoom], '/repo', () => false, (id) => id === 'room-running'),
    /Stop active rooms/,
    'a full-file restore rejects a running room sharing its local root cwd',
  )
  const sshKey = await checkpointWorkspaceKey({ kind: 'task', workspace: { kind: 'ssh', hostId: 'host-1', path: '/srv/repo/../repo' } } as any)
  assert.equal(sshKey, 'ssh:host-1:/srv/repo')
  await assert.rejects(
    () => requireNoRunningWriterAtWorkspaceKey([
      { id: 'ssh-owner', kind: 'task', workspace: { kind: 'ssh', hostId: 'host-1', path: '/srv/repo' } } as any,
    ], [], sshKey!, () => true, () => false),
    /Stop active threads/,
    'checkpoint restore rejects another running SSH thread sharing host and normalized path',
  )
  console.log('ok: restore run-state gates fail closed and distinguish local from SSH ownership')
}

{
  // IPC registration uses Electron's real ipcMain and cannot be invoked from this Node
  // test. Verify the concrete handler bodies keep their native gate ahead of mutation;
  // requireNativeConfirmation's allow/deny/fail-closed behavior has its own unit test.
  const source = readFileSync(fileURLToPath(new URL('./workspace-ipc.ts', import.meta.url)), 'utf8')
  const undoStart = source.indexOf("handle('checkpoint:undoLast'")
  const undoEnd = source.indexOf("handle('checkpoint:diffLast'", undoStart)
  const undo = source.slice(undoStart, undoEnd)
  assert.ok(undoStart >= 0 && undoEnd > undoStart, 'undoLast handler remains discoverable')
  assert.ok(
    undo.indexOf('requireNativeConfirmation') >= 0 && undo.indexOf('requireNativeConfirmation') < undo.indexOf('checkpoints.undoLast'),
    'native main-process confirmation must complete before undoLast restores workspace files',
  )
  const undoConfirm = undo.indexOf('await requireNativeConfirmation')
  assert.ok(undo.indexOf('requireThreadStopped') < undoConfirm && undo.lastIndexOf('requireThreadStopped') > undoConfirm, 'undoLast checks host run state before and after its dialog')
  assert.ok(undo.indexOf('workspaceMutationGuard.acquire') < undo.indexOf('checkpoints.undoLast'), 'undoLast acquires its exclusive mutation lease before restore')
  assert.ok(undo.indexOf('sameCheckpointTarget') > undo.indexOf('workspaceMutationGuard.acquire') && undo.indexOf('sameCheckpointTarget') < undo.indexOf('checkpoints.undoLast'), 'undoLast revalidates the host target while holding the live lease')

  const revertStart = source.indexOf("handle('checkpoint:revertTurn'")
  const revertEnd = source.indexOf("handle('checkpoint:forkTurn'", revertStart)
  const revert = source.slice(revertStart, revertEnd)
  assert.ok(revertStart >= 0 && revertEnd > revertStart, 'revertTurn handler remains discoverable')
  assert.ok(
    revert.indexOf('requireNativeConfirmation') >= 0 && revert.indexOf('requireNativeConfirmation') < revert.indexOf('checkpoints.revertTurn'),
    'native main-process confirmation must complete before revertTurn restores files or conversation',
  )
  const revertConfirm = revert.indexOf('await requireNativeConfirmation')
  assert.ok(revert.indexOf('requireThreadStopped') < revertConfirm && revert.lastIndexOf('requireThreadStopped') > revertConfirm, 'revertTurn checks host run state before and after its dialog')
  assert.ok(revert.indexOf('workspaceMutationGuard.acquire') < revert.indexOf('checkpoints.revertTurn'), 'revertTurn acquires its exclusive mutation lease before restore')
  assert.ok(revert.indexOf('sameCheckpointTarget') > revert.indexOf('workspaceMutationGuard.acquire') && revert.indexOf('sameCheckpointTarget') < revert.indexOf('checkpoints.revertTurn'), 'revertTurn revalidates the host target while holding the live lease')

  const forkStart = source.indexOf("handle('checkpoint:forkTurn'")
  const forkEnd = source.indexOf("handle('workspace:branch'", forkStart)
  const fork = source.slice(forkStart, forkEnd)
  assert.ok(fork.indexOf('workspaceMutationGuard.reserveStart') < fork.indexOf('checkpoints.forkTurn'), 'checkpoint fork reserves its source thread/workspace before creating a worktree')
  assert.ok(fork.indexOf('sameCheckpointTarget') < fork.indexOf('checkpoints.forkTurn'), 'checkpoint fork rejects a host target swap before filesystem work')

  const filesStart = source.indexOf("handle('workspace:revertFiles'")
  const filesEnd = source.indexOf("handle('workspace:commit'", filesStart)
  const files = source.slice(filesStart, filesEnd)
  assert.ok(filesStart >= 0 && filesEnd > filesStart, 'revertFiles handler remains discoverable')
  assert.ok(
    files.indexOf('validateRelativeGitPaths') < files.indexOf('requireNativeConfirmation'),
    'revertFiles validates its bounded relative-path set before asking for confirmation',
  )
  const filesConfirm = files.indexOf('await requireNativeConfirmation')
  assert.ok(
    files.indexOf('requireNoRunningLocalWriterAtWorkspace') < filesConfirm && files.lastIndexOf('requireNoRunningLocalWriterAtWorkspace') > filesConfirm,
    'revertFiles checks active local thread/room workspace owners before and after its dialog',
  )
  assert.ok(
    files.indexOf('requireNativeConfirmation') < files.indexOf("auditEvent('workspace.revert_files'") &&
      files.indexOf('requireNativeConfirmation') < files.indexOf('git checkout --'),
    'native main-process confirmation must complete before revertFiles is audited or mutates Git state',
  )
  assert.doesNotMatch(
    files.slice(files.indexOf('await requireNativeConfirmation'), files.indexOf('const list = safePaths')),
    /safePaths\.(?:join|map|slice)|auditPreview\(workspace\)/,
    'the native confirmation reports only a bounded count, not sensitive full paths',
  )
  const coreSource = readFileSync(fileURLToPath(new URL('./ipc/workspace-ipc.ts', import.meta.url)), 'utf8')
  const workspaceStart = coreSource.indexOf("handle('threads:setWorkspace'")
  const workspaceEnd = coreSource.indexOf("handle('threads:appendNotice'", workspaceStart)
  const setWorkspace = coreSource.slice(workspaceStart, workspaceEnd)
  assert.ok(setWorkspace.indexOf('workspaceMutationGuard.reserveStart') < setWorkspace.indexOf('await dirExists'), 'thread workspace retargeting reserves the thread before asynchronous validation')
  assert.match(setWorkspace, /finally[\s\S]*targetLease\.release/, 'thread workspace retarget reservation always releases')
  const projectStart = coreSource.indexOf("handle('threads:setProject'")
  // threads:setProject is the final handler registered in ipc/workspace-ipc.ts (the surrounding
  // registerWorkspaceIpc() function closes right after it), so slice through to the end of the
  // handler body — the file's closing brace — rather than to a following handler registration.
  const projectEnd = coreSource.indexOf('\n}', projectStart)
  const setProject = coreSource.slice(projectStart, projectEnd)
  assert.ok(setProject.indexOf('workspaceMutationGuard.reserveStart') < setProject.indexOf('t.projectId ='), 'thread project retargeting cannot interleave with a checkpoint lease')
  console.log('ok: destructive workspace/checkpoint restores are native-confirmed before mutation')
}

// `-z` output uses NUL (\u0000) record/field separators and emits raw (un-C-quoted) paths.
// These fixtures mirror real git output captured from `git status --porcelain=v1 -z`
// and `git diff --numstat -z HEAD` on a repo with a non-ASCII path and a rename.

// --- parsePorcelainZ ---------------------------------------------------------

{
  // non-ASCII path is raw UTF-8, not `"caf\303\251.txt"`
  const out = parsePorcelainZ('?? café.txt\u0000')
  assert.deepEqual(out, [{ path: 'café.txt', status: '?' }])
  console.log('ok: porcelain non-ASCII untracked path is raw UTF-8')
}

{
  // path with a literal space survives intact (the old slice/`->` parse mangled these)
  const out = parsePorcelainZ('A  with space.txt\u0000')
  assert.deepEqual(out, [{ path: 'with space.txt', status: 'A' }])
  console.log('ok: porcelain path with embedded space preserved')
}

{
  // rename: NEW path is in the record, OLD path is the next NUL field and must be skipped
  const out = parsePorcelainZ('R  renamed file.txt\u0000tracked.txt\u0000 M other.ts\u0000')
  assert.deepEqual(out, [
    { path: 'renamed file.txt', status: 'R' },
    { path: 'other.ts', status: 'M' },
  ])
  console.log('ok: porcelain rename keeps NEW path and skips OLD field')
}

{
  // copy (C) behaves like rename: new path in record, old in next field
  const out = parsePorcelainZ('C  copy.txt\u0000orig.txt\u0000')
  assert.deepEqual(out, [{ path: 'copy.txt', status: 'R' }])
  console.log('ok: porcelain copy treated like rename')
}

{
  // empty / trailing tokens are ignored, status codes mapped
  const out = parsePorcelainZ(' M a.ts\u0000 D gone.ts\u0000')
  assert.deepEqual(out, [
    { path: 'a.ts', status: 'M' },
    { path: 'gone.ts', status: 'D' },
  ])
  console.log('ok: porcelain modify/delete mapping + trailing NUL ignored')
}

// --- parseGitNumstat ---------------------------------------------------------

{
  const m = parseGitNumstat('5\t2\tserver.ts\u0000')
  assert.deepEqual(m.get('server.ts'), { a: 5, d: 2 })
  console.log('ok: numstat normal path')
}

{
  // non-ASCII path key matches the porcelain path key (so counts line up)
  const m = parseGitNumstat('1\t0\tcafé.txt\u0000')
  assert.deepEqual(m.get('café.txt'), { a: 1, d: 0 })
  console.log('ok: numstat non-ASCII key matches porcelain')
}

{
  // rename: empty path field, then OLD\u0000NEW — keyed on NEW to match porcelain
  const m = parseGitNumstat('1\t0\t\u0000tracked.txt\u0000renamed file.txt\u0000')
  assert.deepEqual(m.get('renamed file.txt'), { a: 1, d: 0 })
  assert.equal(m.get('tracked.txt'), undefined)
  console.log('ok: numstat rename keyed on NEW path')
}

{
  // binary files report `-`/`-` -> 0/0
  const m = parseGitNumstat('-\t-\timg.png\u0000')
  assert.deepEqual(m.get('img.png'), { a: 0, d: 0 })
  console.log('ok: numstat binary dash -> 0/0')
}

{
  // a rename followed by a normal entry both parse correctly (cursor advances right)
  const m = parseGitNumstat('1\t0\t\u0000old.txt\u0000new.txt\u00003\t4\tplain.ts\u0000')
  assert.deepEqual(m.get('new.txt'), { a: 1, d: 0 })
  assert.deepEqual(m.get('plain.ts'), { a: 3, d: 4 })
  console.log('ok: numstat rename then normal entry both parse')
}

console.log('all workspace-ipc parser tests passed')
