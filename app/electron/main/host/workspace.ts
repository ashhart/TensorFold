import { app } from 'electron'
import path from 'node:path'
import os from 'node:os'
import { promises as fs } from 'node:fs'
import { exec } from 'node:child_process'
import type { GitRepositoryState } from '../../agent/types'
import { auditPreview } from '../../agent/security'
import { hostGitEnvForWorkspace } from '../../agent/host-git-env'
import { auditEvent, requireSecurityRole } from './audit'

export async function resolveWorkspaceCwd(cwd: string | null | undefined, label = 'workspace'): Promise<string> {
  const raw = String(cwd || '').trim()
  if (!raw) throw new Error(`Choose a ${label} folder before running workspace tools.`)
  const resolved = path.resolve(raw)
  const stat = await fs.stat(resolved).catch(() => null)
  if (!stat?.isDirectory()) throw new Error(`${label} folder does not exist: ${raw}`)
  // Compare the canonical (symlink-resolved) paths so a symlink pointing at $HOME
  // can't slip past the home-folder guard.
  const canonical = await fs.realpath(resolved).catch(() => resolved)
  if (canonical === path.parse(canonical).root) throw new Error('Refusing to use a filesystem root as a workspace. Pick a project folder instead.')
  const home = await fs.realpath(app.getPath('home')).catch(() => path.resolve(app.getPath('home')))
  if (canonical === home) throw new Error('Refusing to use your home folder as a workspace. Pick a project folder instead.')
  return resolved
}

/** Non-throwing check that a path is an existing directory (for graceful cwd fallback). */
export async function dirExists(cwd: string | null | undefined): Promise<boolean> {
  const raw = String(cwd || '').trim()
  if (!raw) return false
  const stat = await fs.stat(path.resolve(raw)).catch(() => null)
  return !!stat?.isDirectory()
}

export function validateRelativeGitPaths(paths: string[]): string[] {
  const clean = (paths ?? []).map((p) => String(p || '').trim()).filter(Boolean)
  for (const p of clean) {
    if (path.isAbsolute(p) || p.split(/[\\/]+/).includes('..')) throw new Error(`Path "${p}" escapes the workspace.`)
  }
  return clean
}

export const sh = async (cmd: string, cwd: string, timeout = 5000) => {
  const env = await hostGitEnvForWorkspace(cwd)
  return new Promise<string>((resolve) => exec(cmd, { cwd, timeout, maxBuffer: 8_000_000, env }, (err, stdout) => resolve(err ? '' : (stdout || ''))))
}

export const shStrict = async (cmd: string, cwd: string, timeout = 10_000) => {
  const env = await hostGitEnvForWorkspace(cwd)
  return new Promise<string>((resolve, reject) => exec(cmd, { cwd, timeout, maxBuffer: 8_000_000, env }, (err, stdout, stderr) => err ? reject(new Error(stderr || stdout || String(err))) : resolve(stdout || '')))
}

export const qq = (p: string) => `'${p.replace(/'/g, "'\\''")}'`

export async function inspectGitRepository(cwd: string | null | undefined, label = 'workspace'): Promise<GitRepositoryState> {
  const workspace = await resolveWorkspaceCwd(cwd, label)
  const inside = (await sh('git rev-parse --is-inside-work-tree', workspace, 3000)).trim() === 'true'
  if (!inside) {
    return {
      cwd: workspace,
      exists: false,
      insideWorkTree: false,
      topLevel: null,
      branch: null,
      hasCommits: false,
      message: 'This folder is not initialized as a Git repository.',
    }
  }
  const topLevel = (await sh('git rev-parse --show-toplevel', workspace, 3000)).trim() || workspace
  const branchRaw = (await sh('git rev-parse --abbrev-ref HEAD', workspace, 3000)).trim()
  const branch = branchRaw && branchRaw !== 'HEAD' ? branchRaw : null
  const hasCommits = !!(await sh('git rev-parse --verify HEAD', workspace, 3000)).trim()
  return { cwd: workspace, exists: true, insideWorkTree: true, topLevel, branch, hasCommits }
}

export async function initGitRepository(cwd: string): Promise<GitRepositoryState> {
  requireSecurityRole('operator', 'initialize git repositories')
  const before = await inspectGitRepository(cwd)
  if (before.exists && before.hasCommits) return before
  if (!before.exists) {
    await auditEvent('workspace.git_init', { cwd: auditPreview(before.cwd) })
    await shStrict('git init', before.cwd, 10_000)
  }
  const afterInit = await inspectGitRepository(before.cwd)
  if (!afterInit.hasCommits) {
    await auditEvent('workspace.git_initial_commit', { cwd: auditPreview(before.cwd) })
    await shStrict(`git -c core.hooksPath=${qq(os.devNull)} -c user.name="TensorFold" -c user.email=tensorfold-studio@example.test commit --no-verify --allow-empty -m "Initial commit"`, before.cwd, 10_000)
  }
  return inspectGitRepository(before.cwd)
}
