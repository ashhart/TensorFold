import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import type { FanOutBudget } from '../../agent/types'
import { auditPreview } from '../../agent/security'
import { safeSegment, vontraPath } from '../../store'
import { auditEvent } from './audit'
import { qq, sh, shStrict } from './workspace'

export interface MissionAttemptWorkspace {
  id: string
  index: number
  branch: string
  worktreeRoot: string
  /** Directory corresponding to the mission's source cwd (which may be below the repo root). */
  cwd: string
}

export interface MissionAttemptBatch {
  missionId: string
  pass: number
  sourceCwd: string
  repoTop: string
  baseBranch: string
  baseCommit: string
  root: string
  attempts: MissionAttemptWorkspace[]
}

export interface MissionAttemptIntegration {
  ok: boolean
  merged: boolean
  commit: string | null
  retained: boolean
  message: string
}

const ATTEMPT_COUNTS: Record<FanOutBudget, number> = {
  single: 1,
  paired: 2,
  council: 3,
  swarm: 4,
}

export function missionFanOutCount(budget: FanOutBudget | null | undefined): number {
  return ATTEMPT_COUNTS[budget ?? 'single'] ?? 1
}

export async function createMissionAttemptWorktrees(
  sourceCwd: string,
  missionId: string,
  pass: number,
  count: number,
): Promise<MissionAttemptBatch> {
  const boundedCount = Math.max(2, Math.min(4, Math.floor(count)))
  const sourcePath = path.resolve(String(sourceCwd || '').trim())
  const sourceStat = await fs.stat(sourcePath).catch(() => null)
  if (!sourceStat?.isDirectory()) throw new Error('Choose an existing mission workspace before starting fan-out.')
  const source = await fs.realpath(sourcePath).catch(() => sourcePath)
  const inside = (await sh('git rev-parse --is-inside-work-tree', source, 5_000)).trim() === 'true'
  const repoTopRaw = inside ? (await sh('git rev-parse --show-toplevel', source, 5_000)).trim() : ''
  const repoTop = repoTopRaw ? await fs.realpath(repoTopRaw).catch(() => repoTopRaw) : ''
  const hasCommits = inside && !!(await sh('git rev-parse --verify HEAD', source, 5_000)).trim()
  if (!repoTop || !hasCommits) {
    throw new Error('Multi-attempt missions require a Git repository with at least one commit.')
  }
  const branchRaw = (await sh('git rev-parse --abbrev-ref HEAD', repoTop, 5_000)).trim()
  const baseBranch = branchRaw && branchRaw !== 'HEAD' ? branchRaw : ''
  if (!baseBranch) throw new Error('Multi-attempt missions require a checked-out base branch, not detached HEAD.')
  const dirty = (await sh('git status --porcelain', repoTop, 5_000)).trim()
  if (dirty) throw new Error('Multi-attempt mission fan-out requires a clean base worktree. Commit or stash existing changes first.')

  const baseCommit = (await shStrict(`git -C ${qq(repoTop)} rev-parse --verify HEAD`, repoTop, 10_000)).trim()
  const relativeSource = path.relative(repoTop, source)
  if (relativeSource.startsWith('..') || path.isAbsolute(relativeSource)) throw new Error('Mission workspace must be inside its Git worktree.')
  const shortMission = safeSegment(missionId).slice(0, 28) || 'mission'
  const batchId = randomUUID().slice(0, 8)
  const root = vontraPath('mission-attempts', `${shortMission}-pass-${Math.max(1, pass)}-${batchId}`)
  await fs.mkdir(root, { recursive: true })
  const attempts: MissionAttemptWorkspace[] = []

  try {
    for (let index = 0; index < boundedCount; index++) {
      const id = `attempt-${index + 1}`
      const branch = `vontra/mission-${shortMission}-p${Math.max(1, pass)}-a${index + 1}-${batchId}`
      const worktreeRoot = path.join(root, id)
      const cwd = path.join(worktreeRoot, relativeSource)
      await shStrict(`git -C ${qq(repoTop)} worktree add -b ${qq(branch)} ${qq(worktreeRoot)} ${qq(baseCommit)}`, repoTop, 30_000)
      attempts.push({ id, index, branch, worktreeRoot, cwd })
    }
  } catch (error) {
    await cleanupMissionAttemptWorktrees({
      missionId, pass, sourceCwd, repoTop, baseBranch, baseCommit, root, attempts,
    }, true)
    throw error
  }

  await auditEvent('mission.fanout_worktrees_created', {
    missionId,
    pass,
    source: auditPreview(repoTop),
    attempts: attempts.map((attempt) => ({ id: attempt.id, branch: attempt.branch, cwd: auditPreview(attempt.cwd) })),
  })
  return { missionId, pass, sourceCwd, repoTop, baseBranch, baseCommit, root, attempts }
}

export async function integrateMissionAttemptWinner(
  batch: MissionAttemptBatch,
  winnerId: string,
  allowMutations: boolean,
): Promise<MissionAttemptIntegration> {
  const winner = batch.attempts.find((attempt) => attempt.id === winnerId)
  if (!winner) throw new Error(`Mission fan-out winner "${winnerId}" was not found.`)

  if (!allowMutations) {
    await cleanupMissionAttemptWorktrees(batch, true)
    return { ok: true, merged: false, commit: null, retained: false, message: `Selected ${winner.id}; read-only attempts produced no branch to merge.` }
  }

  const sourceHead = (await shStrict(`git -C ${qq(batch.repoTop)} rev-parse --verify HEAD`, batch.repoTop, 10_000)).trim()
  const sourceDirty = (await sh('git status --porcelain', batch.repoTop, 5_000)).trim()
  if (sourceDirty || sourceHead !== batch.baseCommit) {
    return {
      ok: false, merged: false, commit: null, retained: true,
      message: 'The base worktree changed while mission attempts were running. Attempt worktrees were retained; review and merge the winner manually.',
    }
  }

  const changed = (await sh('git status --porcelain', winner.cwd, 5_000)).trim()
  let commit: string | null = null
  if (changed) {
    await shStrict(`git -C ${qq(winner.cwd)} add -A`, winner.cwd, 10_000)
    const message = `Mission ${batch.missionId} pass ${batch.pass}: select ${winner.id}`
    await shStrict(
      `git -C ${qq(winner.cwd)} -c core.hooksPath=${qq(os.devNull)} -c user.name="TensorFold" -c user.email=tensorfold-studio@example.test commit --no-verify -m ${qq(message)}`,
      winner.cwd,
      30_000,
    )
    commit = (await shStrict(`git -C ${qq(winner.cwd)} rev-parse HEAD`, winner.cwd, 10_000)).trim()
  }

  try {
    if (commit) {
      const mergeMessage = `Merge mission winner ${winner.branch} into ${batch.baseBranch}`
      await shStrict(
        `git -C ${qq(batch.repoTop)} -c core.hooksPath=${qq(os.devNull)} merge --no-ff -m ${qq(mergeMessage)} ${qq(winner.branch)}`,
        batch.repoTop,
        30_000,
      )
    }
  } catch (error) {
    const inMerge = (await sh('git rev-parse -q --verify MERGE_HEAD', batch.repoTop, 5_000)).trim()
    if (inMerge) await sh('git merge --abort', batch.repoTop, 10_000)
    return {
      ok: false, merged: false, commit, retained: true,
      message: `Mission winner merge failed and was aborted. Attempt worktrees were retained: ${String((error as Error)?.message ?? error)}`,
    }
  }

  await auditEvent('mission.fanout_winner_integrated', {
    missionId: batch.missionId,
    pass: batch.pass,
    winnerId,
    branch: winner.branch,
    commit,
    merged: !!commit,
  })
  await cleanupMissionAttemptWorktrees(batch, true)
  return {
    ok: true,
    merged: !!commit,
    commit,
    retained: false,
    message: commit ? `Merged ${winner.id} (${commit.slice(0, 8)}) into ${batch.baseBranch}.` : `Selected ${winner.id}; it made no file changes.`,
  }
}

export async function cleanupMissionAttemptWorktrees(batch: MissionAttemptBatch, deleteBranches: boolean): Promise<void> {
  for (const attempt of [...batch.attempts].reverse()) {
    await sh(`git -C ${qq(batch.repoTop)} worktree remove --force ${qq(attempt.worktreeRoot)}`, batch.repoTop, 30_000)
    if (deleteBranches) await sh(`git -C ${qq(batch.repoTop)} branch -D ${qq(attempt.branch)}`, batch.repoTop, 10_000)
  }
  await fs.rm(batch.root, { recursive: true, force: true }).catch(() => {})
}
