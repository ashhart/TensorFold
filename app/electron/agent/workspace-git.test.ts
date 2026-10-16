import assert from 'node:assert/strict'
import { exec as childExec } from 'node:child_process'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { addWorktree, commitWithoutHooks, createBranchInPlace, gitBranch, gitStatus, gitDiff, parsePorcelain, parseNumstat, parseWorktreeList, listBranches, listWorktrees, checkoutBranch, type GitRunner } from './workspace-git'

const exec = promisify(childExec)

function runnerFrom(map: Record<string, { code?: number; stdout?: string; stderr?: string }>): GitRunner {
  return {
    async exec(command) {
      const m = map[command] ?? {}
      return { code: m.code ?? 0, stdout: m.stdout ?? '', stderr: m.stderr ?? '' }
    },
  }
}

// --- pure parsing ---
assert.deepEqual(
  parsePorcelain(' M src/a.ts\n?? new.txt\nA  added.ts\n D del.ts\nR  old.ts -> renamed.ts'),
  [
    { path: 'src/a.ts', status: 'M' },
    { path: 'new.txt', status: '?' },
    { path: 'added.ts', status: 'A' },
    { path: 'del.ts', status: 'D' },
    { path: 'renamed.ts', status: 'R' },
  ],
)

const nm = parseNumstat('3\t1\tsrc/a.ts\n-\t-\tbin.png')
assert.deepEqual(nm.get('src/a.ts'), { additions: 3, deletions: 1 })
assert.deepEqual(nm.get('bin.png'), { additions: 0, deletions: 0 }) // binary

// renames: numstat keys counts under the "old => new" path (and the brace form). parsePorcelain
// keeps the NEW path, so parseNumstat must resolve to the new path or rename counts are lost.
const nmRename = parseNumstat('5\t2\told.txt => new.txt\n1\t0\tsrc/{a => b}/f.ts')
assert.deepEqual(nmRename.get('new.txt'), { additions: 5, deletions: 2 })
assert.deepEqual(nmRename.get('src/b/f.ts'), { additions: 1, deletions: 0 })

// worktree porcelain: blank-line-separated blocks; branch short name; detached
assert.deepEqual(
  parseWorktreeList('worktree /repo\nHEAD abc\nbranch refs/heads/main\n\nworktree /repo-feat\nHEAD def\nbranch refs/heads/feature-x\n\nworktree /repo-det\nHEAD 789\ndetached\n'),
  [
    { path: '/repo', head: 'abc', branch: 'main', detached: false, bare: false },
    { path: '/repo-feat', head: 'def', branch: 'feature-x', detached: false, bare: false },
    { path: '/repo-det', head: '789', branch: null, detached: true, bare: false },
  ],
)

async function run() {
  // branch
  assert.equal(await gitBranch(runnerFrom({ 'git rev-parse --abbrev-ref HEAD': { stdout: 'main\n' } }), '/x'), 'main')
  assert.equal(await gitBranch(runnerFrom({ 'git rev-parse --abbrev-ref HEAD': { stdout: 'HEAD\n' } }), '/x'), null) // detached

  // status: combines porcelain + numstat; untracked defaults to 0/0
  const st = await gitStatus(runnerFrom({
    'git rev-parse --abbrev-ref HEAD': { stdout: 'main\n' },
    'git status --porcelain=v1': { stdout: ' M src/a.ts\n?? new.txt\n' },
    'git diff HEAD --numstat': { stdout: '3\t1\tsrc/a.ts\n' },
  }), '/x')
  assert.equal(st.branch, 'main')
  assert.equal(st.files.length, 2)
  assert.deepEqual(st.files.find((f) => f.path === 'src/a.ts'), { path: 'src/a.ts', status: 'M', additions: 3, deletions: 1 })
  assert.deepEqual(st.files.find((f) => f.path === 'new.txt'), { path: 'new.txt', status: '?', additions: 0, deletions: 0 })
  assert.equal(st.additions, 3)
  assert.equal(st.deletions, 1)

  // diff: tracked uses `git diff HEAD`, untracked uses `--no-index`
  const diffs = await gitDiff(runnerFrom({
    'git rev-parse --abbrev-ref HEAD': { stdout: 'main\n' },
    'git status --porcelain=v1': { stdout: ' M src/a.ts\n?? new.txt\n' },
    'git diff HEAD --numstat': { stdout: '3\t1\tsrc/a.ts\n' },
    "git diff HEAD --no-color -- 'src/a.ts'": { stdout: 'diff --git a/src/a.ts b/src/a.ts\n+added line\n' },
    "git diff --no-index --no-color -- /dev/null 'new.txt'": { stdout: 'new file\n+hello\n' },
  }), '/x')
  assert.equal(diffs.length, 2)
  assert.match(diffs.find((d) => d.path === 'src/a.ts')!.diff, /\+added line/)
  assert.match(diffs.find((d) => d.path === 'new.txt')!.diff, /\+hello/)

  // diff with path filter
  const filtered = await gitDiff(runnerFrom({
    'git rev-parse --abbrev-ref HEAD': { stdout: 'main\n' },
    'git status --porcelain=v1': { stdout: ' M src/a.ts\n M src/b.ts\n' },
    'git diff HEAD --numstat': { stdout: '1\t0\tsrc/a.ts\n2\t0\tsrc/b.ts\n' },
    "git diff HEAD --no-color -- 'src/b.ts'": { stdout: 'b diff\n' },
  }), '/x', ['src/b.ts'])
  assert.equal(filtered.length, 1)
  assert.equal(filtered[0].path, 'src/b.ts')

  // branches + worktrees + checkout
  assert.deepEqual(await listBranches(runnerFrom({ "git for-each-ref --format='%(refname:short)' refs/heads": { stdout: 'main\nfeature-x\ndev\n' } }), '/x'), ['main', 'feature-x', 'dev'])
  const wts = await listWorktrees(runnerFrom({ 'git worktree list --porcelain': { stdout: 'worktree /repo\nHEAD abc\nbranch refs/heads/main\n' } }), '/x')
  assert.equal(wts.length, 1)
  assert.equal(wts[0].branch, 'main')
  assert.equal((await checkoutBranch(runnerFrom({ "git checkout 'feature-x'": { code: 0 } }), '/x', 'feature-x')).ok, true)
  const coFail = await checkoutBranch(runnerFrom({ "git checkout 'nope'": { code: 1, stderr: 'error: pathspec' } }), '/x', 'nope')
  assert.equal(coFail.ok, false)
  assert.match(coFail.error || '', /pathspec/)

  // Worktree create (VST-017 spec test): a NEW branch gets `worktree add -b … HEAD`; an
  // EXISTING branch is attached without -b; a failed add reports the git error.
  const freshAdd = await addWorktree(runnerFrom({
    "git branch --list 'feat-1'": { stdout: '' },
    "git -C '/repo' worktree add -b 'feat-1' '/repo-wt/feat-1' HEAD": { code: 0 },
  }), '/repo', 'feat-1', '/repo-wt/feat-1')
  assert.deepEqual(freshAdd, { ok: true, path: '/repo-wt/feat-1' })

  const attachAdd = await addWorktree(runnerFrom({
    "git branch --list 'feat-1'": { stdout: '  feat-1\n' },
    "git -C '/repo' worktree add '/repo-wt/feat-1' 'feat-1'": { code: 0 },
  }), '/repo', 'feat-1', '/repo-wt/feat-1')
  assert.equal(attachAdd.ok, true)

  const addFail = await addWorktree(runnerFrom({
    "git branch --list 'feat-1'": { stdout: '' },
    "git -C '/repo' worktree add -b 'feat-1' '/repo-wt/feat-1' HEAD": { code: 128, stderr: 'fatal: already exists' },
  }), '/repo', 'feat-1', '/repo-wt/feat-1')
  assert.equal(addFail.ok, false)
  assert.match(addFail.error || '', /already exists/)

  assert.equal((await createBranchInPlace(runnerFrom({ "git checkout -b 'feat-2'": { code: 0 } }), '/repo', 'feat-2')).ok, true)

  // TensorFold Studio's commit path must never execute repository or globally configured hooks. Test both
  // pre-commit (also covered by --no-verify) and post-commit (requires core.hooksPath isolation).
  const repo = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-no-hooks-'))
  const preMarker = path.join(repo, 'pre-hook-ran')
  const postMarker = path.join(repo, 'post-hook-ran')
  try {
    await exec('git init', { cwd: repo })
    await exec('git config user.name "TensorFold Studio Test"', { cwd: repo })
    await exec('git config user.email "vontra-test@local"', { cwd: repo })
    await fs.writeFile(path.join(repo, 'tracked.txt'), 'safe\n', 'utf8')
    await exec('git add tracked.txt', { cwd: repo })
    const hooks = path.join(repo, '.git', 'hooks')
    await fs.mkdir(hooks, { recursive: true })
    await fs.writeFile(path.join(hooks, 'pre-commit'), `#!/bin/sh\nprintf planted > '${preMarker.replace(/'/g, `'\\''`)}'\n`, { mode: 0o755 })
    await fs.writeFile(path.join(hooks, 'post-commit'), `#!/bin/sh\nprintf planted > '${postMarker.replace(/'/g, `'\\''`)}'\n`, { mode: 0o755 })
    const realRunner: GitRunner = {
      exec: async (command, options) => {
        try {
          const out = await exec(command, { cwd: options?.cwd })
          return { code: 0, stdout: out.stdout, stderr: out.stderr }
        } catch (error: any) {
          return { code: typeof error?.code === 'number' ? error.code : 1, stdout: String(error?.stdout ?? ''), stderr: String(error?.stderr ?? error?.message ?? '') }
        }
      },
    }
    const committed = await commitWithoutHooks(realRunner, repo, 'safe commit')
    assert.deepEqual(committed, { ok: true })
    await assert.rejects(() => fs.access(preMarker), 'pre-commit hook was suppressed')
    await assert.rejects(() => fs.access(postMarker), 'post-commit hook was suppressed')
    assert.match((await exec('git log -1 --pretty=%s', { cwd: repo })).stdout, /safe commit/)
  } finally {
    await fs.rm(repo, { recursive: true, force: true })
  }

  console.log('workspace-git tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
