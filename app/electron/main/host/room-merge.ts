// Room worktree merge-back: the convergence step of the room workflow. Room ticket
// workers commit to isolated `vontra/room-*` branches; once the final gates are green
// this module merges those branches back into the project's base branch. Conflicts are
// never forced: the merge is aborted and a ticket is opened for the builder seat.
// Kept free of Electron imports so the jiti-run unit tests can drive it against real
// temporary Git repos.

import os from 'node:os'
import { exec } from 'node:child_process'
import type { Room, RoomMergeResult, Ticket } from '../../agent/types'
import { sanitizeCommitMessage } from '../../agent/git-hygiene'
import { hostGitEnvForWorkspace } from '../../agent/host-git-env'
import { roomFinalGateBlockers } from '../../agent/tickets'
import type { GitRunner } from '../../agent/workspace-git'

export const ROOM_BRANCH_PREFIX = 'vontra/room-'

export function roomBranchName(room: Pick<Room, 'id'>): string {
  return `${ROOM_BRANCH_PREFIX}${room.id.slice(0, 8)}`
}

// Local runner that captures the exit code (the workspace `sh` helper drops it).
export const localRoomMergeRunner: GitRunner = {
  exec: async (command, opts) => {
    const env = await hostGitEnvForWorkspace(opts?.cwd)
    return new Promise((resolve) => {
      exec(command, { cwd: opts?.cwd, timeout: 30_000, maxBuffer: 16 * 1024 * 1024, env }, (err, stdout, stderr) => {
        const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
        resolve({ code, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
      })
    })
  },
}

function qq(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

function mergeError(message: string, baseBranch: string | null, branches: string[]): RoomMergeResult {
  return { ok: false, status: 'error', baseBranch, merged: [], remaining: [...branches], conflict: null, blockers: [], message }
}

/** Final-gate blockers that must clear before a room worktree may merge back. */
export function roomMergeGateBlockers(room: Room, tickets: Ticket[]): string[] {
  const roomTickets = tickets.filter((ticket) => ticket.roomId === room.id)
  // Merge-conflict tickets are the merge's own recovery mechanism: they must not block
  // the MERGE gate (the documented recovery re-runs the merge), though they still count
  // against the room FINAL gate.
  const gateTickets = roomTickets.filter((ticket) => !(ticket.tags ?? []).includes('merge-conflict'))
  const blockers = roomFinalGateBlockers({ tickets: gateTickets })
  if (!roomTickets.length) blockers.unshift('Room has no tickets; nothing to merge.')
  return blockers
}

/**
 * Branches to merge, in ticket-completion order. Ticket workers record their branch on
 * the ticket execution record; a room usually shares one branch, but when a room produced
 * several, the earliest-finished ticket's branch lands first. Falls back to the room
 * worktree branch when no execution record names one.
 */
export function roomMergeBranches(room: Room, tickets: Ticket[]): string[] {
  const completed = tickets
    .filter((ticket) => ticket.roomId === room.id && ticket.status === 'completed')
    // finishedAt/updatedAt mix clocks, so ties break on stable keys (createdAt, then id).
    .sort((a, b) =>
      (a.execution?.finishedAt ?? a.updatedAt) - (b.execution?.finishedAt ?? b.updatedAt) ||
      a.createdAt - b.createdAt ||
      a.id.localeCompare(b.id))
  const ordered: string[] = []
  for (const ticket of completed) {
    const branch = ticket.execution?.branch?.trim()
    if (branch && branch.startsWith(ROOM_BRANCH_PREFIX) && !ordered.includes(branch)) ordered.push(branch)
  }
  return ordered.length ? ordered : [roomBranchName(room)]
}

/** Seat that built the merged work: the latest completed ticket's assignee, else the first seat. */
export function roomMergeBuilderSeat(room: Room, tickets: Ticket[]): string | null {
  const latest = tickets
    .filter((ticket) => ticket.roomId === room.id && ticket.status === 'completed')
    .sort((a, b) =>
      (b.execution?.finishedAt ?? b.updatedAt) - (a.execution?.finishedAt ?? a.updatedAt) ||
      b.createdAt - a.createdAt ||
      b.id.localeCompare(a.id))[0]
  return latest?.assignee ?? room.seats[0]?.name ?? null
}

/** Ticket draft opened on the room board when a merge conflicts; assigned to the builder seat. */
export function roomMergeConflictTicket(room: Room, tickets: Ticket[], branch: string, files: string[], merged: string[] = []): Partial<Ticket> {
  const list = files.length ? files.map((file) => `- ${file}`).join('\n') : '- (conflicted paths could not be determined)'
  const alreadyMerged = merged.length
    ? ['', 'Already merged onto the base branch (their merge commits are in place; do not redo them):', ...merged.map((b) => `- ${b}`)]
    : []
  return {
    projectId: room.projectId,
    roomId: room.id,
    type: 'bug',
    status: 'todo',
    priority: 'high',
    risk: 'high',
    title: `Merge conflict: ${branch}`,
    body: [
      `Merging ${branch} back into the project branch conflicted. The merge was aborted; the base working tree is untouched.`,
      '',
      'Conflicted files:',
      list,
      ...alreadyMerged,
      '',
      'Recovery: resolve the conflicts in the room worktree, mark this ticket completed, then run the room merge again.',
      'This ticket is tagged merge-conflict and does not block the merge gate.',
    ].join('\n'),
    tags: ['merge-conflict'],
    assignee: roomMergeBuilderSeat(room, tickets),
  }
}

/**
 * Merge each room branch into the checked-out base branch, in order. Stops at the first
 * conflict: the in-flight merge is aborted (base worktree restored) and the conflicted
 * file list is reported; branches after it stay unmerged.
 */
export async function mergeRoomWorktreeBranches(
  runner: GitRunner,
  repoTop: string,
  baseBranch: string,
  branches: string[],
): Promise<RoomMergeResult> {
  for (const branch of branches) {
    if (!branch.startsWith(ROOM_BRANCH_PREFIX) || branch.startsWith('-') || branch.includes('..')) {
      return mergeError(`Refusing to merge a non-room branch: ${branch}`, baseBranch, branches)
    }
  }
  if (!branches.length) {
    return { ok: true, status: 'nothing', baseBranch, merged: [], remaining: [], conflict: null, blockers: [], message: 'No room branches to merge.' }
  }
  // A dirty base checkout would mix uncommitted user work into the merge; refuse.
  const dirty = await runner.exec('git status --porcelain', { cwd: repoTop })
  if (dirty.stdout.trim()) {
    return mergeError('Base working tree has uncommitted changes. Commit or stash them before merging room work.', baseBranch, branches)
  }
  const merged: string[] = []
  for (const branch of branches) {
    const message = sanitizeCommitMessage(`Merge room branch ${branch} into ${baseBranch}`)
    const r = await runner.exec(`git -c core.hooksPath=${qq(os.devNull)} merge --no-ff -m ${qq(message)} ${qq(branch)}`, { cwd: repoTop })
    if (r.code === 0) {
      merged.push(branch)
      continue
    }
    const files = (await runner.exec('git diff --no-ext-diff --no-textconv --name-only --diff-filter=U', { cwd: repoTop })).stdout
      .split('\n').map((line) => line.trim()).filter(Boolean)
    const inMerge = (await runner.exec('git rev-parse -q --verify MERGE_HEAD', { cwd: repoTop })).code === 0
    if (inMerge) await runner.exec('git merge --abort', { cwd: repoTop })
    return {
      ok: false,
      status: 'conflict',
      baseBranch,
      merged,
      remaining: branches.slice(merged.length + 1),
      conflict: { branch, files, ticketId: null },
      blockers: [],
      message: `Merge conflict merging ${branch} into ${baseBranch}. The merge was aborted.`,
    }
  }
  return { ok: true, status: 'merged', baseBranch, merged, remaining: [], conflict: null, blockers: [], message: '' }
}
