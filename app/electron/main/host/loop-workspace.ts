import { exec as cpExec } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'

interface LoopCommandResult {
  code: number
  stdout: string
  stderr: string
  timedOut?: boolean
  signal?: string | null
}

export function execCmd(command: string, cwd: string): Promise<LoopCommandResult> {
  return new Promise((resolve) => {
    cpExec(command, { cwd, timeout: 120_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
      const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
      resolve({
        code, stdout: String(stdout ?? ''), stderr: String(stderr ?? ''),
        timedOut: err?.killed === true && err.signal === 'SIGTERM' && err.code == null,
        signal: err?.signal ?? null,
      })
    })
  })
}

export type LoopCommandRunner = (command: string, cwd: string) => Promise<LoopCommandResult>

export interface LoopBaseline {
  baseSha: string
  snapshottedDirtyTree: boolean
}

function commandFailure(label: string, result: { stdout: string; stderr: string }): Error {
  const detail = (result.stderr || result.stdout).trim().slice(-1000)
  return new Error(`Could not create a safe Loop baseline (${label})${detail ? `: ${detail}` : '.'} No loop was started.`)
}

// Verify a baseline before Loop can reset files; snapshot failure restores the original HEAD and exact index.
export async function prepareLoopBaseline(cwd: string, run: LoopCommandRunner = execCmd, kind: 'baseline' | 'checkpoint' = 'baseline'): Promise<LoopBaseline> {
  const inside = await run('git rev-parse --is-inside-work-tree', cwd)
  if (inside.code !== 0 || inside.stdout.trim() !== 'true') throw commandFailure('this folder is not a Git worktree', inside)
  const conflicts = await run('git diff --name-only --diff-filter=U', cwd)
  if (conflicts.code !== 0) throw commandFailure('could not inspect unresolved conflicts', conflicts)
  if (conflicts.stdout.trim()) throw new Error('Resolve Git conflicts before starting Loop Mode. No loop was started.')

  const oldHeadResult = await run('git rev-parse --verify --quiet HEAD', cwd)
  const oldHead = oldHeadResult.code === 0 ? oldHeadResult.stdout.trim() : ''
  const status = await run('git status --porcelain', cwd)
  if (status.code !== 0) throw commandFailure('could not inspect the working tree', status)
  if (!status.stdout.trim()) return { baseSha: oldHead, snapshottedDirtyTree: false }

  const indexResult = await run('git rev-parse --git-path index', cwd)
  if (indexResult.code !== 0 || !indexResult.stdout.trim()) throw commandFailure('could not locate the Git index', indexResult)
  const rawIndexPath = indexResult.stdout.trim()
  const indexPath = path.isAbsolute(rawIndexPath) ? rawIndexPath : path.resolve(cwd, rawIndexPath)
  const originalIndex = await fs.readFile(indexPath).catch((error: NodeJS.ErrnoException) => {
    if (error?.code === 'ENOENT') return null
    throw error
  })
  const originalIndexMode = await fs.stat(indexPath).then((stat) => stat.mode).catch(() => 0o600)

  const restoreOriginalState = async (): Promise<void> => {
    const currentHeadResult = await run('git rev-parse --verify --quiet HEAD', cwd)
    const currentHead = currentHeadResult.code === 0 ? currentHeadResult.stdout.trim() : ''
    let refError: Error | null = null
    if (currentHead && currentHead !== oldHead) {
      const restoreRef = oldHead
        ? await run(`git update-ref HEAD ${oldHead} ${currentHead}`, cwd)
        : await run(`git update-ref -d HEAD ${currentHead}`, cwd)
      if (restoreRef.code !== 0) refError = commandFailure('could not restore the original HEAD after snapshot failure', restoreRef)
    }
    try {
      if (originalIndex == null) {
        await fs.rm(indexPath, { force: true })
      } else {
        await fs.mkdir(path.dirname(indexPath), { recursive: true })
        const tmp = `${indexPath}.vontra-restore-${randomUUID()}`
        await fs.writeFile(tmp, originalIndex, { mode: originalIndexMode })
        await fs.rename(tmp, indexPath)
      }
    } catch (error) {
      throw new Error(`Loop baseline failed and the original Git index could not be restored: ${error instanceof Error ? error.message : String(error)}`)
    }
    if (refError) throw refError
  }

  const add = await run('git add -A', cwd)
  if (add.code !== 0) {
    await restoreOriginalState()
    throw commandFailure('git add failed', add)
  }
  const message = kind === 'checkpoint' ? 'loop checkpoint' : 'loop baseline (pre-existing uncommitted work)'
  const commit = await run(`git -c user.name="TensorFold" -c user.email=tensorfold-studio@example.test commit -q -m "${message}" --no-verify --no-gpg-sign`, cwd)
  if (commit.code !== 0) {
    await restoreOriginalState()
    throw commandFailure('git commit failed', commit)
  }

  const [newHeadResult, cleanResult] = await Promise.all([
    run('git rev-parse --verify --quiet HEAD', cwd),
    run('git status --porcelain', cwd),
  ])
  const newHead = newHeadResult.code === 0 ? newHeadResult.stdout.trim() : ''
  if (!newHead || newHead === oldHead || cleanResult.code !== 0 || cleanResult.stdout.trim()) {
    await restoreOriginalState()
    throw new Error('Could not verify a clean, new Loop baseline commit. The original Git state was restored and no loop was started.')
  }
  return { baseSha: newHead, snapshottedDirtyTree: true }
}

export async function rollbackLoopWorkspace(cwd: string, target: string, run: LoopCommandRunner = execCmd): Promise<void> {
  const reset = await run(target ? `git reset --hard ${target}` : 'git reset --hard', cwd)
  if (reset.code !== 0) throw commandFailure('rollback reset failed', reset)
  if (target) {
    const head = await run('git rev-parse --verify --quiet HEAD', cwd)
    if (head.code !== 0 || head.stdout.trim() !== target) throw commandFailure('rollback target could not be verified', head)
  }
  // Never clean untracked files if reset failed or landed on the wrong commit.
  const clean = await run('git clean -fd', cwd)
  if (clean.code !== 0) throw commandFailure('rollback clean failed', clean)
  const status = await run('git status --porcelain', cwd)
  if (status.code !== 0 || status.stdout.trim()) throw commandFailure('rollback did not leave a clean worktree', status)
}
