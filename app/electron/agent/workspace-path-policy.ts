import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { blockedReadReason, blockedWriteReason } from './command-policy'

export type WorkspacePathAccess = 'read' | 'write'

async function readControlPointer(file: string, prefix = ''): Promise<string | null> {
  const canonical = await fs.realpath(file).catch(() => null)
  if (!canonical) return null
  const stat = await fs.lstat(canonical)
  if (!stat.isFile()) return null
  if (stat.size > 4096) throw new Error('Git control pointer exceeds the size limit.')
  const handle = await fs.open(canonical, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    if (!(await handle.stat()).isFile()) return null
    const buffer = Buffer.alloc(4097)
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)
    if (bytesRead > 4096) throw new Error('Git control pointer exceeds the size limit.')
    const value = buffer.subarray(0, bytesRead).toString('utf8').trim()
    if (!value.startsWith(prefix)) return null
    return fs.realpath(path.resolve(path.dirname(file), value.slice(prefix.length).trim())).catch(() => null)
  } finally { await handle.close() }
}

async function assertOutsideGitControlRoots(root: string, target: string): Promise<void> {
  const dotGit = path.join(root, '.git')
  const gitStat = await fs.stat(dotGit).catch(() => null)
  const gitDir = gitStat?.isDirectory() ? await fs.realpath(dotGit) : await readControlPointer(dotGit, 'gitdir:')
  if (!gitDir) return
  const commonDir = await readControlPointer(path.join(gitDir, 'commondir'))
  for (const control of [gitDir, commonDir]) {
    if (control && (target === control || target.startsWith(control + path.sep))) {
      throw new Error('Blocked by TensorFold safety policy: refuses accessing Git private metadata.')
    }
  }
}

export function assertWorkspacePathPolicy(root: string, target: string, access: WorkspacePathAccess, flavor = path): void {
  const relative = flavor.relative(root, target).replace(/\\/g, '/')
  if (relative === '..' || relative.startsWith('../') || flavor.isAbsolute(relative)) {
    throw new Error(`Path "${target}" escapes the workspace through a symlink.`)
  }
  // Include absolute Git components so choosing a directory inside .git as the
  // workspace cannot turn its control files into ordinary relative paths.
  const reason = blockedWriteReason(target) || (access === 'write' ? blockedWriteReason(relative) : blockedReadReason(relative))
  if (reason) throw new Error(`Blocked by TensorFold safety policy: ${reason}.`)
}

/** Canonicalize missing tails without creating directories or following dangling links. */
export async function canonicalWorkspacePath(rootPath: string, targetPath: string, access: WorkspacePathAccess): Promise<string> {
  const lexicalRoot = path.resolve(rootPath)
  const lexicalTarget = path.resolve(targetPath)
  assertWorkspacePathPolicy(lexicalRoot, lexicalTarget, access)
  const root = await fs.realpath(lexicalRoot)
  const missing: string[] = []
  let probe = lexicalTarget
  while (true) {
    try { await fs.lstat(probe); break } catch (error: any) {
      if (error?.code !== 'ENOENT') throw error
      const parent = path.dirname(probe)
      if (parent === probe) throw error
      missing.unshift(path.basename(probe))
      probe = parent
    }
  }
  const target = path.join(await fs.realpath(probe), ...missing)
  assertWorkspacePathPolicy(root, target, access)
  await assertOutsideGitControlRoots(root, target)
  return target
}
