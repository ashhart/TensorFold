// Run: jiti electron/agent/workspace-checkpoints.test.ts
import assert from 'node:assert/strict'
import { exec as cpExec } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import {
  checkpointRef,
  CHECKPOINT_REFS_PREFIX,
  isGitRepository,
  createCheckpoint,
  createForkWorktree,
  restoreCheckpoint,
  diffCheckpoints,
  diffCheckpointToWorktree,
  listCheckpointTurns,
  deleteCheckpoints,
  type GitRunner,
} from './workspace-checkpoints.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

interface Call { command: string; cwd?: string }

// A fake GitRunner that records every command and replies from a per-command
// matcher table. No subprocess, git, or network is ever touched.
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

// Default replies covering every probe createCheckpoint makes.
function defaultReply(command: string) {
  if (command.includes('--is-inside-work-tree')) return { stdout: 'true' }
  if (command.includes('rev-parse --verify --quiet HEAD')) return { stdout: 'deadbeefHEAD' }
  if (command.includes('--git-common-dir')) return { stdout: '/repo/.git' }
  if (command.includes('write-tree')) return { stdout: 'tree111' }
  if (command.includes('commit-tree')) return { stdout: 'commit222' }
  if (command.includes('rev-parse --verify --quiet') && command.includes('^{commit}')) return { stdout: 'res333' }
  if (command.includes('git cat-file -t')) return { stdout: 'blob\n' }
  return {}
}

console.log('workspace-checkpoints tests:')

{
  // Ref is built under the hidden vontra namespace, never refs/heads/*.
  const ref = checkpointRef('thread-A', 7)
  assert.equal(ref, `${CHECKPOINT_REFS_PREFIX}/thread-A/turn/7`)
  assert.ok(ref.startsWith('refs/vontra/checkpoints/'))
  assert.ok(!ref.startsWith('refs/heads/'))
  // Slashes / unsafe chars in a threadId are encoded so they can't escape the ref dir.
  assert.equal(checkpointRef('a/b spaced', 1), `${CHECKPOINT_REFS_PREFIX}/a%2Fb%20spaced/turn/1`)
  ok('checkpointRef: hidden refs/vontra/checkpoints/<thread>/turn/<n>, encoded, never refs/heads')
}

{
  // isGitRepository gates on --is-inside-work-tree == true.
  const yes = fakeRunner(() => ({ stdout: 'true' }))
  assert.equal(await isGitRepository(yes.runner, '/repo'), true)
  const no = fakeRunner(() => ({ code: 128, stderr: 'not a git repository' }))
  assert.equal(await isGitRepository(no.runner, '/x'), false)
  ok('isGitRepository: true only when inside a work tree')
}

{
  const { runner, calls } = fakeRunner(defaultReply)
  const res = await createCheckpoint(runner, '/repo', 'thread-A', 3)
  assert.equal(res.ok, true)
  assert.equal(res.sha, 'commit222')

  const cmds = calls.map((c) => c.command)
  const joined = cmds.join('\n')

  // Proof 1: uses an ISOLATED GIT_INDEX_FILE on every index-touching command.
  const indexCmds = cmds.filter((c) => /git (read-tree|add|write-tree|commit-tree)/.test(c))
  assert.ok(indexCmds.length >= 4, 'expected read-tree/add/write-tree/commit-tree')
  for (const c of indexCmds) {
    assert.ok(/GIT_INDEX_FILE='[^']*vontra-checkpoint-index-/.test(c), `missing isolated index env: ${c}`)
  }
  // Temp index sits in the git common dir, never in the worktree.
  assert.ok(joined.includes(`/repo/.git/vontra-checkpoint-index-`))

  // Proof 2: the exact command sequence / shapes.
  assert.ok(cmds.some((c) => /GIT_INDEX_FILE=\S+ .*git read-tree HEAD$/.test(c)), 'read-tree HEAD')
  assert.ok(cmds.some((c) => /git add -A -- \.$/.test(c)), 'add -A -- .')
  assert.ok(cmds.some((c) => /git write-tree$/.test(c)), 'write-tree')
  assert.ok(cmds.some((c) => /git commit-tree tree111 -m /.test(c)), 'commit-tree of written tree')

  // Proof 3: writes ONLY a hidden checkpoint ref via update-ref (not branch/tag).
  const updateRef = cmds.find((c) => c.includes('update-ref') && !c.includes('-d'))
  assert.ok(updateRef, 'update-ref present')
  assert.ok(updateRef!.includes(`'refs/vontra/checkpoints/thread-A/turn/3'`), 'updates only the hidden ref')
  assert.ok(updateRef!.includes('commit222'), 'points ref at the commit-tree sha')

  // Proof 4: NEVER 'git commit', NEVER 'git checkout <branch>', NEVER moves HEAD.
  // 'git commit' (which moves HEAD) is forbidden; 'git commit-tree' (which does not) is required.
  assert.ok(!cmds.some((c) => /\bgit commit(?!-tree)\b/.test(c)), 'must not run git commit')
  assert.ok(!cmds.some((c) => /git checkout/.test(c)), 'must not checkout')
  assert.ok(!cmds.some((c) => /update-ref\s+(--)?\s*HEAD\b/.test(c)), 'must not move HEAD')
  assert.ok(!cmds.some((c) => /refs\/heads\//.test(c)), 'must not touch refs/heads/*')

  // Proof 5: pinned identity, not the user's git config.
  assert.ok(indexCmds.every((c) => c.includes('GIT_AUTHOR_NAME=') && c.includes('GIT_COMMITTER_EMAIL=')))

  // Proof 6: the throwaway index is cleaned up.
  assert.ok(cmds.some((c) => /^rm -f .*vontra-checkpoint-index-/.test(c)), 'temp index removed')
  ok('createCheckpoint: isolated GIT_INDEX_FILE; only refs/vontra/*; no commit/checkout/HEAD move')
}

{
  // A repo with no HEAD commit must SKIP read-tree but still write a checkpoint.
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes('rev-parse --verify --quiet HEAD')) return { code: 1, stdout: '' }
    return defaultReply(command)
  })
  const res = await createCheckpoint(runner, '/repo', 'fresh', 0)
  assert.equal(res.ok, true)
  assert.ok(!calls.some((c) => c.command.includes('read-tree')), 'no HEAD -> skip read-tree')
  ok('createCheckpoint: empty repo (no HEAD) skips read-tree, still captures')
}

{
  // restore rewrites ONLY agent-owned paths and re-syncs those index entries.
  const { runner, calls } = fakeRunner(defaultReply)
  const res = await restoreCheckpoint(runner, '/repo', 'refs/vontra/checkpoints/t/turn/2', ['src/a.ts', 'src/b.ts'])
  assert.equal(res.ok, true)
  const cmds = calls.map((c) => c.command)
  assert.ok(cmds.some((c) => /git --literal-pathspecs restore --source res333 --worktree --staged -- 'src\/a\.ts' 'src\/b\.ts'$/.test(c)), 'restore only owned paths')
  assert.ok(!cmds.some((c) => /git clean/.test(c)), 'never broad-clean untracked user work')
  assert.ok(cmds.some((c) => /git --literal-pathspecs reset --quiet -- 'src\/a\.ts' 'src\/b\.ts'$/.test(c)), 'reset only owned index entries')
  // Guards: never checkout a branch, never move HEAD, never touch refs/heads.
  assert.ok(!cmds.some((c) => /git checkout/.test(c)), 'restore must not checkout')
  assert.ok(!cmds.some((c) => /update-ref.*HEAD/.test(c)), 'restore must not move HEAD')
  assert.ok(!cmds.some((c) => /refs\/heads\//.test(c)), 'restore must not touch refs/heads/*')
  ok('restoreCheckpoint: owned-path restore+reset, never clean/checkout/HEAD')
}

{
  // A forged/bad changedPaths entry must never widen a leaf-scoped restore into
  // the whole worktree or a directory subtree. Git tree objects are preserved,
  // while only concrete blob paths may reach git restore/reset.
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes("'res333:src'")) return { stdout: 'tree\n' }
    return defaultReply(command)
  })
  const res = await restoreCheckpoint(runner, '/repo', 'refs/vontra/checkpoints/t/turn/2', [
    '.', 'src', 'src/a.ts', '.git/config', '../outside', 'src/a.ts',
  ])
  assert.equal(res.ok, true)
  assert.deepEqual(res.restoredPaths, ['src/a.ts'])
  assert.deepEqual(res.preservedPaths, ['src'])
  const mutation = calls.map((c) => c.command).find((c) => c.includes(' restore --source ')) ?? ''
  assert.match(mutation, /-- 'src\/a\.ts'$/)
  assert.ok(!mutation.includes("-- '.'"), 'dot can never become a whole-worktree pathspec')
  assert.ok(!mutation.includes("'.git/config'"), 'git metadata can never enter restore scope')

  const refused = await restoreCheckpoint(runner, '/repo', 'refs/vontra/checkpoints/t/turn/2', ['.'])
  assert.equal(refused.ok, false)
  assert.match(refused.error ?? '', /No agent-owned file paths/)
  ok('restoreCheckpoint: rejects dot/metadata/traversal and never restores tree objects')
}

{
  // REAL git, real subprocess, zero commits: hasHeadCommit's own gate must not
  // skip the post-restore index re-sync. A fresh `git init` repo (no commits at
  // all) has no HEAD, yet `git reset --quiet -- .` is empirically fine to run
  // with zero commits (it un-stages without erroring) -- restoreCheckpoint must
  // run it regardless of HEAD existing, or every restored file is left staged.
  const root = await mkdtemp(path.join(tmpdir(), 'vontra-checkpoint-nohead-'))
  const realRunner: GitRunner = {
    exec: (command, opts) => new Promise((resolve) => {
      cpExec(command, { cwd: opts?.cwd ?? root, timeout: 25_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
        const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
        resolve({ code, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
      })
    }),
  }
  try {
    const init = await realRunner.exec('git init -q', { cwd: root })
    assert.equal(init.code, 0, init.stderr)
    assert.equal(await isGitRepository(realRunner, root), true)

    // Zero real commits -> HEAD does not exist (the no-commit-repo case the
    // finding describes).
    const headCheck = await realRunner.exec('git rev-parse --verify --quiet HEAD', { cwd: root })
    assert.notEqual(headCheck.code, 0, 'fresh repo must have no HEAD commit')

    await writeFile(path.join(root, 'file.txt'), 'hello\n', 'utf8')
    const cp = await createCheckpoint(realRunner, root, 'nohead-thread', 1)
    assert.equal(cp.ok, true, cp.error)

    // Mutate the worktree after the checkpoint so restore has something to do.
    await writeFile(path.join(root, 'file.txt'), 'changed\n', 'utf8')
    await writeFile(path.join(root, 'extra.txt'), 'new file\n', 'utf8')

    const ref = checkpointRef('nohead-thread', 1)
    const restore = await restoreCheckpoint(realRunner, root, ref, ['file.txt', 'extra.txt'])
    assert.equal(restore.ok, true, restore.error)
    assert.deepEqual(restore.preservedPaths, ['extra.txt'], 'a post-checkpoint created path is preserved without a content ownership proof')
    assert.equal(await readFile(path.join(root, 'extra.txt'), 'utf8'), 'new file\n', 'untracked user/concurrent work survives restore')

    // Step 1 of restore (`git restore --staged`) stages the restored files into
    // the real index. The finding: with zero commits, hasHeadCommit() is false,
    // so the `git reset -- .` re-sync was skipped entirely and files were left
    // staged. Assert they are NOT staged after restore.
    const status = await realRunner.exec('git status --porcelain', { cwd: root })
    const stagedLines = status.stdout.split('\n').filter((line) => line.length > 0 && line[0] !== ' ' && line[0] !== '?')
    assert.deepEqual(stagedLines, [], `expected no staged entries after restore, got: ${JSON.stringify(status.stdout)}`)
    ok('restoreCheckpoint: zero-commit repo still re-syncs the index (no staged leftovers)')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

{
  // Restoring a missing ref is guarded: no mutation runs.
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes('^{commit}')) return { code: 1, stdout: '' }
    return defaultReply(command)
  })
  const res = await restoreCheckpoint(runner, '/repo', 'refs/vontra/checkpoints/t/turn/99', ['src/a.ts'])
  assert.equal(res.ok, false)
  assert.ok((res.error || '').includes('not found'))
  assert.ok(!calls.some((c) => /git (restore|clean|reset)/.test(c.command)), 'no mutation on missing ref')
  ok('restoreCheckpoint: missing ref -> ok:false, no worktree mutation')
}

{
  // diff compares two resolved checkpoint commits, read-only.
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes('fromRef') || command.includes('turn/1')) return { stdout: 'aaa111' }
    if (command.includes('turn/2')) return { stdout: 'bbb222' }
    if (command.includes('git diff')) return { stdout: 'DIFF-PATCH' }
    return defaultReply(command)
  })
  const res = await diffCheckpoints(
    runner, '/repo',
    'refs/vontra/checkpoints/t/turn/1',
    'refs/vontra/checkpoints/t/turn/2',
  )
  assert.equal(res.ok, true)
  assert.equal(res.diff, 'DIFF-PATCH')
  const diffCmd = calls.map((c) => c.command).find((c) => c.startsWith('git diff'))
  assert.ok(diffCmd && /git diff --patch --no-color --no-ext-diff --no-textconv aaa111 bbb222$/.test(diffCmd), diffCmd)
  // Read-only: no mutation of worktree/index/refs.
  assert.ok(!calls.some((c) => /git (restore|clean|reset|update-ref|checkout|commit)/.test(c.command)), 'diff is read-only')
  ok('diffCheckpoints: read-only git diff between two checkpoint commits')
}

{
  // delete is best-effort update-ref -d over the hidden refs only.
  const { runner, calls } = fakeRunner(() => ({}))
  await deleteCheckpoints(runner, '/repo', ['refs/vontra/checkpoints/t/turn/1', 'refs/vontra/checkpoints/t/turn/2'])
  const dels = calls.map((c) => c.command).filter((c) => c.includes('update-ref -d'))
  assert.equal(dels.length, 2)
  assert.ok(dels.every((c) => c.includes('refs/vontra/checkpoints/')), 'deletes only hidden refs')
  ok('deleteCheckpoints: update-ref -d over hidden refs, best-effort')
}

{
  // listCheckpointTurns enumerates a thread's hidden turn refs and returns turn
  // numbers in ascending numeric order (git for-each-ref lists lexically: 1,10,2…).
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes('for-each-ref')) {
      return { stdout: 'refs/vontra/checkpoints/t/turn/1\nrefs/vontra/checkpoints/t/turn/10\nrefs/vontra/checkpoints/t/turn/2\n' }
    }
    return defaultReply(command)
  })
  const turns = await listCheckpointTurns(runner, '/repo', 't')
  assert.deepEqual(turns, [1, 2, 10], 'turns sorted numerically, not lexically')
  const cmd = calls.map((c) => c.command).find((c) => c.includes('for-each-ref'))
  assert.ok(cmd && cmd.includes('refs/vontra/checkpoints/t/turn'), 'scoped to the thread namespace')
  // Read-only: never mutates worktree/index/refs.
  assert.ok(!calls.some((c) => /git (restore|clean|reset|update-ref|checkout|commit)\b/.test(c.command)), 'list is read-only')
  ok('listCheckpointTurns: numeric-sorted turns scoped to the thread, read-only')
}

{
  // No refs for the thread -> empty list, not an error.
  const { runner } = fakeRunner((command) => (command.includes('for-each-ref') ? { stdout: '' } : defaultReply(command)))
  assert.deepEqual(await listCheckpointTurns(runner, '/repo', 'empty'), [])
  ok('listCheckpointTurns: no checkpoints -> []')
}

{
  // diffCheckpointToWorktree diffs a checkpoint commit against the CURRENT worktree
  // (tracked + untracked) by snapshotting the worktree into an EPHEMERAL index tree,
  // never touching the real index/HEAD/refs.
  const { runner, calls } = fakeRunner((command) => {
    if (command.includes('write-tree')) return { stdout: 'treeNOW' }
    if (command.startsWith('git diff') || command.includes(' git diff')) return { stdout: 'LIVE-DIFF' }
    return defaultReply(command)
  })
  const res = await diffCheckpointToWorktree(runner, '/repo', 'refs/vontra/checkpoints/t/turn/2')
  assert.equal(res.ok, true)
  assert.equal(res.diff, 'LIVE-DIFF')
  const diffCmd = calls.map((c) => c.command).find((c) => c.includes('git diff'))
  assert.ok(diffCmd && /git diff --patch --no-color --no-ext-diff --no-textconv res333 treeNOW$/.test(diffCmd), diffCmd)
  // The worktree snapshot used an isolated GIT_INDEX_FILE (real index untouched)…
  assert.ok(calls.some((c) => /GIT_INDEX_FILE='[^']*vontra-checkpoint-index-/.test(c.command)), 'ephemeral index for snapshot')
  // …and the temp index is cleaned up; nothing mutates worktree/HEAD/refs.
  assert.ok(calls.some((c) => /^rm -f .*vontra-checkpoint-index-/.test(c.command)), 'temp index removed')
  assert.ok(!calls.some((c) => /git (restore|clean|reset|update-ref|checkout)\b/.test(c.command)), 'no worktree/ref mutation')
  assert.ok(!calls.some((c) => /\bgit commit(?!-tree)\b/.test(c.command)), 'never git commit')
  ok('diffCheckpointToWorktree: checkpoint vs live worktree via ephemeral index, read-only')
}

{
  // Missing checkpoint ref -> ok:false, no snapshot work.
  const { runner, calls } = fakeRunner((command) => (command.includes('^{commit}') ? { code: 1, stdout: '' } : defaultReply(command)))
  const res = await diffCheckpointToWorktree(runner, '/repo', 'refs/vontra/checkpoints/t/turn/99')
  assert.equal(res.ok, false)
  assert.ok((res.error || '').includes('not found'))
  assert.ok(!calls.some((c) => c.command.includes('write-tree')), 'no snapshot on missing ref')
  ok('diffCheckpointToWorktree: missing ref -> ok:false, no snapshot')
}

{
  // REGRESSION: capturing/restoring from a workspace cwd that is a SUBDIRECTORY of the repo must
  // still snapshot/restore the WHOLE worktree, not just the subtree. Previously read-tree/add/restore
  // ran with the subdir as cwd, so parent/sibling files were dropped from the checkpoint and a
  // restore left out-of-subdir turn edits un-reverted.
  const root = await mkdtemp(path.join(tmpdir(), 'vontra-checkpoint-subdir-'))
  const realRunner: GitRunner = {
    exec: (command, opts) => new Promise((resolve) => {
      cpExec(command, { cwd: opts?.cwd ?? root, timeout: 25_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
        const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
        resolve({ code, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
      })
    }),
  }
  try {
    assert.equal((await realRunner.exec('git init -q', { cwd: root })).code, 0)
    await realRunner.exec('git config user.email t@t.t', { cwd: root })
    await realRunner.exec('git config user.name t', { cwd: root })
    // A nested working subdirectory the agent might be cwd'd into.
    const sub = path.join(root, 'pkg', 'app')
    await realRunner.exec(`mkdir -p ${sub}`, { cwd: root })
    await writeFile(path.join(root, 'root.txt'), 'root-orig\n', 'utf8')
    await writeFile(path.join(sub, 'app.txt'), 'app-orig\n', 'utf8')
    await realRunner.exec('git add -A', { cwd: root })
    await realRunner.exec('git commit -qm init', { cwd: root })

    // Capture a checkpoint with cwd = the SUBDIRECTORY (not the repo root).
    const cp = await createCheckpoint(realRunner, sub, 'subdir-thread', 1)
    assert.equal(cp.ok, true, cp.error)

    // The checkpoint tree must contain the PARENT-level file too, re-rooted at the repo top — proving
    // the snapshot was anchored at the worktree root, not the subdir.
    const lsTree = await realRunner.exec(`git ls-tree -r --name-only ${cp.sha}`, { cwd: root })
    const names = lsTree.stdout.split('\n').filter(Boolean)
    assert.ok(names.includes('root.txt'), `checkpoint must include parent-level root.txt, got: ${JSON.stringify(names)}`)
    assert.ok(names.includes('pkg/app/app.txt'), `checkpoint must include subdir file at repo-rooted path, got: ${JSON.stringify(names)}`)

    // Now mutate BOTH a parent file and the subdir file, then restore from the subdir cwd.
    await writeFile(path.join(root, 'root.txt'), 'root-CHANGED\n', 'utf8')
    await writeFile(path.join(sub, 'app.txt'), 'app-CHANGED\n', 'utf8')
    // changedPaths are relative to the thread cwd, not the repo root. A shell
    // command may legitimately report a parent path that still resolves inside
    // the repo; both forms must normalize to repo-root pathspecs safely.
    const restore = await restoreCheckpoint(realRunner, sub, checkpointRef('subdir-thread', 1), ['../../root.txt', 'app.txt'])
    assert.equal(restore.ok, true, restore.error)

    // Both files must be reverted — including the out-of-subdir parent file.
    const rootAfter = (await realRunner.exec('cat root.txt', { cwd: root })).stdout
    const appAfter = (await realRunner.exec('cat pkg/app/app.txt', { cwd: root })).stdout
    assert.equal(rootAfter, 'root-orig\n', 'parent-level file must be reverted by a subdir-cwd restore')
    assert.equal(appAfter, 'app-orig\n', 'subdir file must be reverted')
    ok('createCheckpoint/restoreCheckpoint: subdir cwd snapshots & restores the WHOLE worktree')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

// ---- createForkWorktree (VST-036): fake-runner failure modes + real-git materialization ----

{
  const missing = fakeRunner((command) => {
    if (command.includes('--show-toplevel')) return { stdout: '/repo' }
    if (command.includes('^{commit}')) return { code: 1, stderr: '' }
    return {}
  })
  const noRef = await createForkWorktree(missing.runner, '/repo', 'refs/vontra/checkpoints/T/turn/9', '/wt/dest')
  assert.equal(noRef.ok, false)
  assert.match(noRef.error ?? '', /checkpoint ref not found/)
  assert.ok(!missing.calls.some((c) => c.command.includes('worktree add')), 'no worktree op without a resolvable ref')

  const addFails = fakeRunner((command) => {
    if (command.includes('--show-toplevel')) return { stdout: '/repo' }
    if (command.includes('^{commit}')) return { stdout: 'abc123' }
    if (command.includes('worktree add')) return { code: 128, stderr: "fatal: '/wt/dest' already exists" }
    return {}
  })
  const clash = await createForkWorktree(addFails.runner, '/repo', 'refs/vontra/checkpoints/T/turn/1', '/wt/dest')
  assert.equal(clash.ok, false)
  assert.match(clash.error ?? '', /already exists/)
  ok('createForkWorktree: missing ref and add failures surface as explicit errors')
}

{
  // Real git: checkpoint a dirty worktree, fork it into a NEW worktree, and prove the
  // fork materializes at exactly the snapshot (including a then-untracked file) while
  // the source worktree stays untouched.
  const root = await mkdtemp(path.join(tmpdir(), 'vontra-fork-wt-'))
  const realRunner: GitRunner = {
    exec: (command, opts) => new Promise((resolve) => {
      cpExec(command, { cwd: opts?.cwd ?? root, timeout: 25_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
        const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
        resolve({ code, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
      })
    }),
  }
  try {
    assert.equal((await realRunner.exec('git init -q', { cwd: root })).code, 0)
    await realRunner.exec('git config user.email t@t.t', { cwd: root })
    await realRunner.exec('git config user.name t', { cwd: root })
    await writeFile(path.join(root, 'tracked.txt'), 'v1\n')
    await realRunner.exec('git add -A', { cwd: root })
    await realRunner.exec('git commit -qm init', { cwd: root })
    // Dirty state at the fork point: an edit + an untracked file.
    await writeFile(path.join(root, 'tracked.txt'), 'v2-at-fork\n')
    await writeFile(path.join(root, 'untracked.txt'), 'scratch\n')
    const cp = await createCheckpoint(realRunner, root, 'fork-thread', 1)
    assert.equal(cp.ok, true)
    // The source moves on after the snapshot.
    await writeFile(path.join(root, 'tracked.txt'), 'v3-later\n')

    const dest = path.join(root, '..', path.basename(root) + '-fork')
    const made = await createForkWorktree(realRunner, root, checkpointRef('fork-thread', 1), dest)
    assert.equal(made.ok, true, made.error)
    try {
      const tracked = (await realRunner.exec('cat tracked.txt', { cwd: dest })).stdout
      const untracked = (await realRunner.exec('cat untracked.txt', { cwd: dest })).stdout
      assert.equal(tracked, 'v2-at-fork\n', 'fork worktree holds the snapshot content')
      assert.equal(untracked, 'scratch\n', 'the then-untracked file materializes too')
      const sourceNow = (await realRunner.exec('cat tracked.txt', { cwd: root })).stdout
      assert.equal(sourceNow, 'v3-later\n', 'the source worktree is untouched')
      ok('createForkWorktree: real git — fork materializes the exact snapshot in its own worktree')
    } finally {
      await realRunner.exec(`git worktree remove --force ${JSON.stringify(dest)}`, { cwd: root }).catch(() => {})
      await rm(dest, { recursive: true, force: true }).catch(() => {})
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

console.log(`\n${passed} passed`)
