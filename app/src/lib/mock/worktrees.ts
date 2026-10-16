import type { TensorFoldStudioApi } from '../ipc'

type Worktree = Awaited<ReturnType<TensorFoldStudioApi['getWorktrees']>>['worktrees'][number]

// Browser-only worktree records. These simulate Git responses without touching disk.
export function createMockWorktreeApi(): Pick<TensorFoldStudioApi, 'getWorktrees' | 'createWorktree'> {
  const byProject = new Map<string, Worktree[]>()
  const rowsFor = (cwd: string): Worktree[] => {
    let rows = byProject.get(cwd)
    if (!rows) {
      const projectName = cwd.split('/').filter(Boolean).pop() || 'project'
      rows = [
        { path: cwd, head: 'abc1234', branch: 'main', detached: false, bare: false },
        { path: `/Users/example/.vontra/worktrees/${projectName}/feature-dark-mode`, head: 'def5678', branch: 'feature/dark-mode', detached: false, bare: false },
      ]
      byProject.set(cwd, rows)
    }
    return rows
  }
  return {
    getWorktrees: async ({ cwd, workspace }) => {
      const target = workspace?.path || cwd
      const rows = rowsFor(target)
      return { branches: [...new Set(rows.flatMap((row) => row.branch ? [row.branch] : []))], worktrees: rows.map((row) => ({ ...row })), current: rows.find((row) => row.path === target)?.branch ?? null }
    },
    createWorktree: async ({ cwd, workspace, name, mode }) => {
      const target = workspace?.path || cwd
      const branch = (name || '').trim().toLowerCase().replace(/[^a-z0-9._/-]+/g, '-').replace(/\/{2,}/g, '/').replace(/^[-/.]+|[-/.]+$/g, '').slice(0, 80) || 'feature'
      const rows = rowsFor(target)
      if (workspace?.kind === 'ssh' && mode === 'worktree') return { ok: false, branch, mode, path: '', error: 'Worktrees on SSH hosts are not supported.' }
      if (rows.some((row) => row.branch === branch)) return { ok: false, branch, mode, path: '', error: 'A worktree already uses this branch.' }
      if (mode === 'branch') {
        const main = rows.find((row) => row.path === target)!
        main.branch = branch
        return { ok: true, branch, mode, path: target }
      }
      const projectName = target.split('/').filter(Boolean).pop() || 'project'
      const path = `/Users/example/.vontra/worktrees/${projectName}/${branch.replace(/\//g, '-')}`
      rows.push({ path, branch, head: 'abc1234', detached: false, bare: false })
      return { ok: true, branch, mode, path }
    },
  }
}
