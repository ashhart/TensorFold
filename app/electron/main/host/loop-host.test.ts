// Tests for waitForThreadSettled — the fix for Loop Mode capturing role/critic output before
// the agent run finishes. startThreadRun is fire-and-forget (sets thread.status='running'
// synchronously, resolves with {runId}, then settles asynchronously flipping status to
// 'idle'/'error'). Loop Mode must wait for the run to actually settle before slicing the
// timeline, otherwise it captures an empty (user-only) slice every cycle.
//
// waitForThreadSettled polls the observable status until it leaves 'running'. It returns the
// final observed status: 'idle'/'error' when the run settles, undefined if the thread vanished,
// and 'running' if the wall-clock cap elapsed (so a stuck run can't hang the loop forever).

import assert from 'node:assert/strict'
import { exec as cpExec } from 'node:child_process'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { assertLoopSurfaceSupported, createLoopHost, prepareLoopBaseline, reserveLoopWorkspace, rollbackLoopWorkspace, waitForThreadSettled, loopRoleApprovalMode, type LoopCommandRunner } from './loop-host'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

const shellRunner: LoopCommandRunner = (command, cwd) => new Promise((resolve) => {
  cpExec(command, { cwd }, (error, stdout, stderr) => resolve({
    code: error ? (typeof (error as NodeJS.ErrnoException).code === 'number' ? (error as any).code : 1) : 0,
    stdout: String(stdout ?? ''),
    stderr: String(stderr ?? ''),
  }))
})

async function run() {
  // 1) The core regression: status starts 'running' (as startThreadRun leaves it) and only
  //    flips to 'idle' after a few polls. waitForThreadSettled must NOT return until then.
  {
    let status: 'idle' | 'running' | 'error' | undefined = 'running'
    let ticks = 0
    const sleep = async () => { ticks++; if (ticks >= 3) status = 'idle' }
    const result = await waitForThreadSettled(() => status, { pollMs: 1, sleep })
    assert.equal(result, 'idle', 'should wait through running and return once the run settles to idle')
    assert.ok(ticks >= 3, `should have polled until settle (ticks=${ticks})`)
  }

  // 2) A failed run settles to 'error' — surfaced as the return value, not swallowed.
  {
    let status: 'idle' | 'running' | 'error' | undefined = 'running'
    const sleep = async () => { status = 'error' }
    const result = await waitForThreadSettled(() => status, { pollMs: 1, sleep })
    assert.equal(result, 'error', 'should return error when the run settles to error')
  }

  // 3) Already-idle status returns immediately without sleeping (no spurious poll).
  {
    let slept = false
    const result = await waitForThreadSettled(() => 'idle', { pollMs: 1, sleep: async () => { slept = true } })
    assert.equal(result, 'idle')
    assert.equal(slept, false, 'should not sleep when already settled')
  }

  // 4) Thread removed underneath us (getStatus -> undefined) returns immediately, never hangs.
  {
    let slept = false
    const result = await waitForThreadSettled(() => undefined, { pollMs: 1, sleep: async () => { slept = true } })
    assert.equal(result, undefined, 'a vanished thread settles to undefined, not an infinite loop')
    assert.equal(slept, false, 'a vanished thread must not poll forever')
  }

  // 5) A run that never settles must hit the timeout (returns the last observed 'running')
  //    rather than hanging the loop forever. Use an injected clock so the test stays fast.
  {
    let clock = 0
    const result = await waitForThreadSettled(() => 'running', {
      pollMs: 1,
      timeoutMs: 5,
      now: () => clock,
      sleep: async () => { clock += 2 }, // advance past the 5ms budget within a few polls
    })
    assert.equal(result, 'running', 'a stuck run must hit the timeout cap and return rather than hang')
  }

  console.log('PASS loop-host waitForThreadSettled: waits for fire-and-forget run to settle; surfaces idle/error, returns on gone/timeout')

  // [MEDIUM #16] Non-read-only Loop roles must run with the user's ACTUAL saved global
  // approvalMode, not a hardcoded 'auto' literal — a hardcoded 'auto' silently suppresses safe-
  // class prompts for every implementing/reviewing role, even when the user chose 'suggest'
  // (approve every mutation) as their global posture. Read-only roles keep their existing
  // 'readonly' behavior regardless of the global setting, since they cannot mutate anything.
  {
    // Non-read-only role ('dev') must inherit the user's global approvalMode verbatim.
    assert.equal(loopRoleApprovalMode('dev', 'suggest'), 'suggest', 'a non-read-only role must respect a saved suggest posture, not be silently escalated to auto')
    assert.equal(loopRoleApprovalMode('dev', 'auto'), 'auto', 'a non-read-only role mirrors a saved auto posture')
    assert.equal(loopRoleApprovalMode('qa', 'suggest'), 'suggest', 'every non-read-only phase (qa) must respect suggest, not just dev')
    assert.equal(loopRoleApprovalMode('review', 'suggest'), 'suggest', 'every non-read-only phase (review) must respect suggest, not just dev')
    // The independent critic has no phase and is an auxiliary verifier, so it is always readonly.
    assert.equal(loopRoleApprovalMode(undefined, 'suggest'), 'readonly', 'the independent critic cannot mutate the workspace')

    // Read-only roles (product/architect) are unaffected by this fix — they stay 'readonly'
    // regardless of what the global approvalMode is, since they never mutate source.
    assert.equal(loopRoleApprovalMode('product', 'suggest'), 'readonly', 'read-only role product stays readonly under suggest')
    assert.equal(loopRoleApprovalMode('architect', 'suggest'), 'readonly', 'read-only role architect stays readonly under suggest')
    assert.equal(loopRoleApprovalMode('architect', 'auto'), 'readonly', 'read-only role architect stays readonly even under a global auto posture')
  }
  console.log('PASS loop-host loopRoleApprovalMode: non-read-only roles inherit the saved global approvalMode instead of a hardcoded auto; read-only roles unchanged')

  // SSH ownership is derived from the real surface/project, not renderer-supplied projectId.
  // The gate runs before baseCwd is resolved, so a local lookalike (or nonexistent path) cannot
  // receive any local git/process work for a remote room.
  {
    const state: any = {
      projects: [{ id: 'remote-project', workspace: { kind: 'ssh', hostId: 'pgx', path: '/srv/repo' } }],
      threads: [],
      rooms: [{ id: 'remote-room', projectId: 'remote-project', workspace: null }],
      tickets: [],
    }
    assert.throws(
      () => assertLoopSurfaceSupported({ source: 'room', surfaceId: 'remote-room', projectId: 'remote-project', baseCwd: '/local/lookalike' }, state),
      /does not support SSH workspaces/,
    )
    assert.throws(
      () => assertLoopSurfaceSupported({ source: 'room', surfaceId: 'remote-room-override', projectId: 'local-project', baseCwd: '/managed/local' }, {
        projects: [{ id: 'local-project', cwd: '/local/project', workspace: null }],
        threads: [], tickets: [],
        rooms: [{ id: 'remote-room-override', projectId: 'local-project', workspaceCwd: '/managed/local', workspace: { kind: 'ssh', hostId: 'pgx', path: '/srv/repo' } }],
      } as any),
      /does not support SSH workspaces/,
      'a managed-looking workspaceCwd cannot mask an SSH room override',
    )
    assert.throws(
      () => assertLoopSurfaceSupported({ source: 'room', surfaceId: 'remote-room', projectId: 'spoofed-local-project', baseCwd: '/local/lookalike' }, state),
      /does not match its source surface/,
    )
  }
  console.log('PASS loop-host SSH ownership: actual surface/project state fails closed')

  {
    const state: any = {
      projects: [{ id: 'p', cwd: '/owned/repo', workspace: null }],
      threads: [{ id: 't', projectId: 'p', cwd: '/owned/repo', workspace: null }],
      rooms: [], tickets: [],
    }
    assert.throws(
      () => assertLoopSurfaceSupported({ source: 'solo', surfaceId: 't', projectId: 'p', baseCwd: '/unrelated/repo' }, state),
      /workspace does not match its source surface/,
      'renderer baseCwd cannot retarget a loop outside its host-owned surface',
    )
    assert.equal(assertLoopSurfaceSupported({ source: 'solo', surfaceId: 't', projectId: 'p', baseCwd: '/owned/repo' }, state).cwd, '/owned/repo')

    const guard = new WorkspaceMutationGuard()
    const lease = reserveLoopWorkspace(guard, '/owned/repo')
    assert.throws(() => guard.acquire({ workspaceKey: 'local:/owned/repo' }), /already in progress/, 'restore cannot overlap a Loop lifetime reservation')
    lease.release()
    assert.doesNotThrow(() => guard.acquire({ workspaceKey: 'local:/owned/repo' }).release(), 'restore can proceed after Loop releases')
  }
  console.log('PASS loop-host target + lease: baseCwd is owned and restore cannot race the loop')

  // A failed dirty-tree snapshot must restore the exact staged/unstaged/untracked state and must
  // not publish a running Loop record. Inject only the commit failure; all other commands hit a
  // real temporary git repository.
  {
    const repo = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-loop-baseline-'))
    const run = (command: string) => shellRunner(command, repo)
    try {
      assert.equal((await run('git init -q')).code, 0)
      await fs.writeFile(path.join(repo, 'tracked.txt'), 'original\n')
      assert.equal((await run('git add tracked.txt')).code, 0)
      assert.equal((await run('git -c user.name=Test -c user.email=test@example.test commit -q -m initial')).code, 0)
      await fs.writeFile(path.join(repo, 'staged.txt'), 'staged\n')
      assert.equal((await run('git add staged.txt')).code, 0)
      await fs.writeFile(path.join(repo, 'tracked.txt'), 'unstaged change\n')
      await fs.writeFile(path.join(repo, 'untracked.txt'), 'untracked\n')

      const beforeStatus = (await run('git status --porcelain')).stdout
      const beforeHead = (await run('git rev-parse HEAD')).stdout.trim()
      const beforeIndex = await fs.readFile(path.join(repo, '.git', 'index'))
      const failingRunner: LoopCommandRunner = async (command, cwd) => command.includes(' commit ')
        ? { code: 1, stdout: '', stderr: 'injected commit failure' }
        : shellRunner(command, cwd)

      await assert.rejects(() => prepareLoopBaseline(repo, failingRunner), /git commit failed.*injected commit failure/s)
      assert.equal((await run('git status --porcelain')).stdout, beforeStatus, 'staged/unstaged/untracked status is exact after failure')
      assert.equal((await run('git rev-parse HEAD')).stdout.trim(), beforeHead, 'HEAD is unchanged after failure')
      assert.deepEqual(await fs.readFile(path.join(repo, '.git', 'index')), beforeIndex, 'the original index bytes are restored')
      assert.equal(await fs.readFile(path.join(repo, 'tracked.txt'), 'utf8'), 'unstaged change\n')
      assert.equal(await fs.readFile(path.join(repo, 'untracked.txt'), 'utf8'), 'untracked\n')
      const source = await fs.readFile(path.join(process.cwd(), 'electron/main/host/loop-host.ts'), 'utf8')
      const start = source.indexOf('async function startLoop')
      const baselineCall = source.indexOf('const baseline = await prepareLoopBaseline', start)
      const publishRecord = source.indexOf('loops.set(rec.id, rec)', start)
      const publishTicket = source.indexOf("reflectOnTicket(rec, 'running')", start)
      assert.ok(start >= 0 && baselineCall > start, 'startLoop invokes the baseline proof')
      assert.ok(baselineCall < publishRecord && baselineCall < publishTicket, 'no loop/ticket running record is published before baseline proof')

      await assert.rejects(() => prepareLoopBaseline(repo, failingRunner, 'checkpoint'), /git commit failed/, 'checkpoint commit failure is surfaced')
      assert.equal((await run('git status --porcelain')).stdout, beforeStatus, 'failed checkpoint restores the original index/worktree status')

      const rollbackCalls: string[] = []
      await assert.rejects(() => rollbackLoopWorkspace(repo, beforeHead, async (command) => {
        rollbackCalls.push(command)
        return command.startsWith('git reset') ? { code: 1, stdout: '', stderr: 'injected reset failure' } : shellRunner(command, repo)
      }), /rollback reset failed/)
      assert.equal(rollbackCalls.some((command) => command.startsWith('git clean')), false, 'a failed reset never proceeds to destructive clean')
    } finally {
      await fs.rm(repo, { recursive: true, force: true })
    }
  }
  console.log('PASS loop-host baseline failure: exact Git state retained and no running record published')

  // Purge lifecycle: restored loop records are cleared in memory, persistence is quiesced, and
  // deleting loops.json after the barrier cannot be undone by a late writer.
  {
    const home = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-loop-purge-'))
    const loopsFile = path.join(home, 'loops.json')
    process.env.VONTRA_HOME = home
    await fs.writeFile(loopsFile, JSON.stringify([{
      id: 'old-loop', source: 'thread', surfaceId: 't1', goal: 'SECRET LOOP GOAL', mode: 'build',
      status: 'running', cwd: home, branch: 'vontra/old', phase: null, cycle: 1, maxCycles: 2,
      checkpoints: 0, lastVerdict: null, blockers: [], log: ['SECRET LOOP LOG'], roleThreadIds: [],
      createdAt: 1, updatedAt: 1,
    }]), { mode: 0o600 })
    const host = createLoopHost({ startThreadRun: async () => ({ runId: 'unused' }), workspaceMutationGuard: new WorkspaceMutationGuard() })
    await host.init()
    assert.equal(host.getLoops().length, 1, 'legacy loops.json restores before purge')
    assert.equal(host.getLoops()[0].status, 'stopped', 'restart marks a formerly-running loop stopped')
    assert.doesNotMatch(await fs.readFile(loopsFile, 'utf8'), /SECRET LOOP GOAL|SECRET LOOP LOG/, 'legacy plaintext loops migrate to protected state on read')
    await host.quiesceAndClear()
    await fs.rm(loopsFile, { force: true })
    await new Promise((resolve) => setTimeout(resolve, 10))
    assert.equal(host.getLoops().length, 0, 'purge clears restored Loop Mode runtime state')
    await assert.rejects(fs.stat(loopsFile), /ENOENT/, 'no queued Loop Mode writer resurrects loops.json after the barrier')
    host.resumeAfterPurge()
    delete process.env.VONTRA_HOME
    await fs.rm(home, { recursive: true, force: true })
  }
  console.log('PASS loop-host purge lifecycle: protected migration, quiesce, clear, and no writer resurrection')
}

run().catch((e) => { console.error(e); process.exit(1) })
