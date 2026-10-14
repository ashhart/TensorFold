// Room worktree merge-back tests. Drives the real merge path against temporary Git
// repos: clean merge lands on the base branch, gated rooms refuse, and a conflict
// aborts cleanly (base worktree untouched) with a ticket draft naming the files.

import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import type { Room, Ticket, TicketExecutionRecord } from '../../agent/types'
import { roomFinalGateBlockers } from '../../agent/tickets'
import {
  localRoomMergeRunner as runner,
  mergeRoomWorktreeBranches,
  roomBranchName,
  roomMergeBranches,
  roomMergeConflictTicket,
  roomMergeGateBlockers,
} from './room-merge'

async function git(cwd: string, cmd: string): Promise<string> {
  const r = await runner.exec(`git ${cmd}`, { cwd })
  assert.equal(r.code, 0, `git ${cmd} failed: ${r.stderr || r.stdout}`)
  return r.stdout.trim()
}

function mkRoom(id: string): Room {
  return {
    id,
    projectId: 'project-1',
    name: 'Merge test room',
    seats: [{ id: 'seat-builder', name: 'builder', role: 'builder', providerId: 'openai', model: 'gpt-5.5' }],
    problem: 'test',
    state: 'idle',
    events: [],
    finalText: '',
    createdAt: 1,
    updatedAt: 1,
  } as unknown as Room
}

function mkExecution(patch: Partial<TicketExecutionRecord>): TicketExecutionRecord {
  return { target: 'room', startedAt: 1, finishedAt: 2, checks: [], blockers: [], reviewVerdict: null, ...patch }
}

function mkTicket(patch: Partial<Ticket>): Ticket {
  return {
    id: patch.id ?? `t-${Math.random().toString(36).slice(2, 10)}`,
    projectId: 'project-1',
    roomId: null,
    title: 'Ticket',
    body: '',
    type: 'task',
    status: 'todo',
    priority: 'medium',
    effort: 'medium',
    tags: [],
    risk: 'low',
    parentId: null,
    assignee: null,
    threadId: null,
    runState: null,
    order: 1,
    createdAt: 1,
    updatedAt: 1,
    ...patch,
  } as Ticket
}

async function main(): Promise<void> {
  const root = await mkdtemp(path.join(tmpdir(), 'vontra-room-merge-'))
  try {
    await git(root, 'init -b main')
    await git(root, 'config user.name "TensorFold Studio Test"')
    await git(root, 'config user.email "vontra-test@local"')
    await writeFile(path.join(root, 'base.txt'), 'base\n', 'utf8')
    await git(root, 'add base.txt')
    await git(root, 'commit -m initial')

    // 1) Clean merge lands the room branch's changes on the base branch.
    const room = mkRoom('abcd1234efgh')
    const branch = roomBranchName(room)
    assert.equal(branch, 'vontra/room-abcd1234')
    await git(root, `checkout -b ${branch}`)
    await writeFile(path.join(root, 'room.txt'), 'from room\n', 'utf8')
    await git(root, 'add room.txt')
    await git(root, `commit -m 'room work'`)
    await git(root, 'checkout main')

    const clean = await mergeRoomWorktreeBranches(runner, root, 'main', [branch])
    assert.equal(clean.status, 'merged')
    assert.equal(clean.ok, true)
    assert.deepEqual(clean.merged, [branch])
    assert.deepEqual(clean.remaining, [])
    assert.equal(await readFile(path.join(root, 'room.txt'), 'utf8'), 'from room\n')
    assert.equal(await git(root, 'status --porcelain'), '', 'base worktree clean after merge')

    // 2) Ungated rooms refuse: open tickets block, no tickets block, green room passes.
    assert.ok(roomMergeGateBlockers(room, [mkTicket({ roomId: room.id, status: 'in_progress' })]).length > 0, 'open ticket blocks the gate')
    assert.ok(roomMergeGateBlockers(room, []).some((b) => /no tickets/i.test(b)), 'empty room blocks the gate')
    const greenTicket = mkTicket({
      roomId: room.id,
      status: 'completed',
      requiredEvidence: ['Summary'],
      evidence: [{ id: 'e1', kind: 'note', summary: 'Investigation summary.', createdBy: 'builder', createdAt: 10 }],
    })
    assert.deepEqual(roomMergeGateBlockers(room, [greenTicket]), [], 'completed low-risk ticket passes the gate')

    // 2b) A merge-conflict ticket must NOT block the MERGE gate (it is the merge's own
    //     recovery path), but it still blocks the room FINAL gate.
    const conflictTicket = mkTicket({ roomId: room.id, status: 'todo', tags: ['merge-conflict'], title: 'Merge conflict: vontra/room-x' })
    assert.deepEqual(roomMergeGateBlockers(room, [greenTicket, conflictTicket]), [], 'merge-conflict ticket is excluded from the merge gate')
    assert.ok(roomFinalGateBlockers({ tickets: [greenTicket, conflictTicket] }).length > 0, 'merge-conflict ticket still blocks the room final gate')
    const stillBlocked = roomMergeGateBlockers(room, [greenTicket, conflictTicket, mkTicket({ roomId: room.id, status: 'in_progress' })])
    assert.ok(stillBlocked.length > 0, 'ordinary open tickets still block the merge gate')

    // 3) Conflict: merge aborts cleanly, base worktree untouched, ticket draft names the files.
    const room2 = mkRoom('bbbb2222cccc')
    const branch2 = roomBranchName(room2)
    await git(root, `checkout -b ${branch2} main`)
    await writeFile(path.join(root, 'base.txt'), 'room change\n', 'utf8')
    await git(root, `commit -am 'room edit'`)
    await git(root, 'checkout main')
    await writeFile(path.join(root, 'base.txt'), 'main change\n', 'utf8')
    await git(root, `commit -am 'main edit'`)

    const conflict = await mergeRoomWorktreeBranches(runner, root, 'main', [branch2, 'vontra/room-ffff9999'])
    assert.equal(conflict.status, 'conflict')
    assert.equal(conflict.ok, false)
    assert.equal(conflict.conflict?.branch, branch2)
    assert.deepEqual(conflict.conflict?.files, ['base.txt'])
    assert.deepEqual(conflict.merged, [])
    assert.deepEqual(conflict.remaining, ['vontra/room-ffff9999'], 'later branches stay unmerged after a conflict')
    assert.equal(await readFile(path.join(root, 'base.txt'), 'utf8'), 'main change\n', 'abort restored the base file')
    assert.equal(await git(root, 'status --porcelain'), '', 'abort left a clean base worktree')
    const mergeHead = await runner.exec('git rev-parse -q --verify MERGE_HEAD', { cwd: root })
    assert.notEqual(mergeHead.code, 0, 'no merge left in progress')

    const draft = roomMergeConflictTicket(room2, [mkTicket({ roomId: room2.id, status: 'completed', assignee: 'builder' })], branch2, conflict.conflict!.files)
    assert.equal(draft.roomId, room2.id)
    assert.equal(draft.assignee, 'builder', 'conflict ticket goes to the builder seat')
    assert.ok(draft.body!.includes('- base.txt'), 'ticket body lists the conflicted files')
    assert.ok(draft.title!.includes(branch2))
    assert.ok(draft.body!.includes('mark this ticket completed'), 'ticket body states the recovery steps explicitly')
    assert.ok(draft.body!.includes('run the room merge again'))
    assert.deepEqual(draft.tags, ['merge-conflict'])

    // 3b) Partial merge: the conflict ticket lists the branches already merged onto base.
    const partial = roomMergeConflictTicket(room2, [], branch2, ['a.ts'], ['vontra/room-aaaa0000'])
    assert.ok(partial.body!.includes('Already merged onto the base branch'), 'partial-merge ticket names the landed branches')
    assert.ok(partial.body!.includes('- vontra/room-aaaa0000'))
    const noPartial = roomMergeConflictTicket(room2, [], branch2, ['a.ts'])
    assert.ok(!noPartial.body!.includes('Already merged'), 'no merged-branches section when nothing landed')

    // 4) Several room branches merge in ticket-completion order.
    const tLate = mkTicket({ roomId: room.id, status: 'completed', execution: mkExecution({ branch: 'vontra/room-aaaa1111', finishedAt: 200 }) })
    const tEarly = mkTicket({ roomId: room.id, status: 'completed', execution: mkExecution({ branch: 'vontra/room-bbbb2222', finishedAt: 100 }) })
    assert.deepEqual(roomMergeBranches(room, [tLate, tEarly]), ['vontra/room-bbbb2222', 'vontra/room-aaaa1111'])
    assert.deepEqual(roomMergeBranches(room, [mkTicket({ roomId: room.id, status: 'completed' })]), [branch], 'falls back to the room worktree branch')

    // 4b) Equal finishedAt ties break on createdAt, then id (stable, single-clock keys).
    const tieB = mkTicket({ id: 't-b', roomId: room.id, status: 'completed', createdAt: 5, execution: mkExecution({ branch: 'vontra/room-bbbb0002', finishedAt: 100 }) })
    const tieA = mkTicket({ id: 't-a', roomId: room.id, status: 'completed', createdAt: 5, execution: mkExecution({ branch: 'vontra/room-aaaa0002', finishedAt: 100 }) })
    const tieC = mkTicket({ id: 't-c', roomId: room.id, status: 'completed', createdAt: 3, execution: mkExecution({ branch: 'vontra/room-cccc0002', finishedAt: 100 }) })
    assert.deepEqual(
      roomMergeBranches(room, [tieB, tieA, tieC]),
      ['vontra/room-cccc0002', 'vontra/room-aaaa0002', 'vontra/room-bbbb0002'],
      'ties order by createdAt, then id',
    )

    // 5) A dirty base checkout refuses before touching anything.
    await writeFile(path.join(root, 'base.txt'), 'dirty\n', 'utf8')
    const dirty = await mergeRoomWorktreeBranches(runner, root, 'main', [branch])
    assert.equal(dirty.status, 'error')
    assert.match(dirty.message, /uncommitted changes/)
    await git(root, 'checkout -- base.txt')

    // 6) Non-room branches are never merged through this path.
    const refused = await mergeRoomWorktreeBranches(runner, root, 'main', ['feature/not-a-room'])
    assert.equal(refused.status, 'error')
    assert.match(refused.message, /non-room branch/)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

main()
  .then(() => console.log('room-merge tests passed'))
  .catch((err) => {
    console.error(err)
    process.exitCode = 1
  })
