import assert from 'node:assert/strict'
import os from 'node:os'
import path from 'node:path'
import { promises as fs } from 'node:fs'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { cleanupMissionAttemptWorktrees, createMissionAttemptWorktrees, integrateMissionAttemptWinner, missionFanOutCount } from './mission-worktree-fanout'

const run = promisify(execFile)
const sandbox = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-mission-fanout-'))
const root = path.join(sandbox, 'repo')
await fs.mkdir(root)
process.env.VONTRA_HOME = path.join(sandbox, 'vontra-home')
const git = async (...args: string[]) => (await run('git', args, { cwd: root })).stdout.trim()

try {
  await git('init', '-b', 'main')
  await git('config', 'user.name', 'Test')
  await git('config', 'user.email', 'test@example.com')
  await fs.writeFile(path.join(root, 'base.txt'), 'base\n')
  await git('add', 'base.txt')
  await git('commit', '-m', 'base')

  assert.equal(missionFanOutCount('single'), 1)
  assert.equal(missionFanOutCount('paired'), 2)
  assert.equal(missionFanOutCount('council'), 3)
  assert.equal(missionFanOutCount('swarm'), 4)

  const batch = await createMissionAttemptWorktrees(root, 'mission-test', 1, 3)
  assert.equal(batch.attempts.length, 3)
  assert.equal(new Set(batch.attempts.map((attempt) => attempt.cwd)).size, 3)
  await fs.writeFile(path.join(batch.attempts[1].cwd, 'winner.txt'), 'winner\n')
  const integrated = await integrateMissionAttemptWinner(batch, 'attempt-2', true)
  assert.equal(integrated.ok, true)
  assert.equal(integrated.merged, true)
  assert.equal(await fs.readFile(path.join(root, 'winner.txt'), 'utf8'), 'winner\n')
  assert.equal((await git('worktree', 'list', '--porcelain')).includes(batch.root), false)

  await fs.mkdir(path.join(root, 'nested'))
  await fs.writeFile(path.join(root, 'nested', 'tracked.txt'), 'nested\n')
  await git('add', 'nested/tracked.txt')
  await git('commit', '-m', 'nested source')
  const nested = await createMissionAttemptWorktrees(path.join(root, 'nested'), 'mission-nested', 2, 2)
  assert.equal(path.basename(nested.attempts[0].cwd), 'nested')
  assert.notEqual(nested.attempts[0].cwd, nested.attempts[0].worktreeRoot)
  assert.equal(await fs.readFile(path.join(nested.attempts[0].cwd, 'tracked.txt'), 'utf8'), 'nested\n')
  await cleanupMissionAttemptWorktrees(nested, true)

  const readOnly = await createMissionAttemptWorktrees(root, 'mission-readonly', 3, 2)
  await fs.writeFile(path.join(readOnly.attempts[0].cwd, 'ignored.txt'), 'ignored\n')
  const selected = await integrateMissionAttemptWinner(readOnly, 'attempt-1', false)
  assert.equal(selected.ok, true)
  assert.equal(selected.merged, false)
  await assert.rejects(fs.stat(path.join(root, 'ignored.txt')))

  await fs.writeFile(path.join(root, 'dirty.txt'), 'dirty\n')
  await assert.rejects(() => createMissionAttemptWorktrees(root, 'mission-dirty', 4, 2), /clean base worktree/)
  await fs.rm(path.join(root, 'dirty.txt'))

  const retained = await createMissionAttemptWorktrees(root, 'mission-retained', 5, 2)
  await fs.writeFile(path.join(retained.attempts[0].cwd, 'candidate.txt'), 'candidate\n')
  await fs.writeFile(path.join(root, 'base-race.txt'), 'changed during run\n')
  const race = await integrateMissionAttemptWinner(retained, 'attempt-1', true)
  assert.equal(race.ok, false)
  assert.equal(race.retained, true)
  assert.equal(await fs.stat(retained.attempts[0].cwd).then(() => true), true)
  await fs.rm(path.join(root, 'base-race.txt'))
  await cleanupMissionAttemptWorktrees(retained, true)

  console.log('mission worktree fan-out tests passed')
} finally {
  await fs.rm(sandbox, { recursive: true, force: true })
}
