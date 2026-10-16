import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
import { assertWorkspacePathPolicy, canonicalWorkspacePath } from './workspace-path-policy'
import { readOpenedTextFile, withFileReadBudget } from './bounded-file-read'

export interface AtomicWorkspaceWriteDeps {
  /** Test seam used to simulate a target swap at the former check/use boundary. */
  beforeRename?: (target: string) => Promise<void> | void
  /** Add/move operations must fail rather than replace a concurrently-created entry. */
  ifExists?: 'replace' | 'error'
  /** Optimistic digest preflight for edits staged from previously-read content. */
  expectedSha256?: string
}

export function workspaceContentSha256(content: string | Uint8Array): string {
  return createHash('sha256').update(content).digest('hex')
}

function isWithin(root: string, candidate: string): boolean {
  return candidate === root || candidate.startsWith(root + path.sep)
}

async function canonicalParent(root: string, lexicalParent: string): Promise<string> {
  const missing: string[] = []
  let probe = lexicalParent
  while (true) {
    try {
      const stat = await fs.lstat(probe)
      if (!stat.isDirectory() && !stat.isSymbolicLink()) throw new Error(`Write parent is not a directory: ${probe}`)
      break
    } catch (error: any) {
      if (error?.code !== 'ENOENT') throw error
      const parent = path.dirname(probe)
      if (parent === probe) throw error
      missing.unshift(path.basename(probe))
      probe = parent
    }
  }

  let current = await fs.realpath(probe)
  if (!isWithin(root, current)) throw new Error(`Path "${lexicalParent}" escapes the workspace through a symlink.`)
  for (const segment of missing) {
    const desired = path.join(current, segment)
    const temp = path.join(current, `.vontra-dir-${process.pid}-${randomUUID()}`)
    await fs.mkdir(temp, { mode: 0o700 })
    try {
      // Same random-source rename property as the file write: if the canonical
      // parent path is swapped, the unpredictable temp source disappears and the
      // operation fails instead of creating through the replacement symlink.
      await fs.rename(temp, desired)
    } catch (error: any) {
      await fs.rmdir(temp).catch(() => {})
      if (error?.code !== 'EEXIST' && error?.code !== 'ENOTEMPTY') throw error
    }
    current = await fs.realpath(desired)
    if (!isWithin(root, current)) throw new Error(`Path "${lexicalParent}" escapes the workspace through a symlink.`)
  }
  return current
}

interface DirectoryIdentity { dev: number; ino: number }

async function directoryIdentity(directory: string): Promise<DirectoryIdentity> {
  const stat = await fs.stat(directory)
  if (!stat.isDirectory()) throw new Error(`Write parent is not a directory: ${directory}`)
  return { dev: stat.dev, ino: stat.ino }
}

async function assertStableParent(
  lexicalParent: string,
  canonical: string,
  expected: DirectoryIdentity,
  targetPath: string,
): Promise<void> {
  const current = await fs.realpath(lexicalParent).catch(() => '')
  if (current !== canonical) throw new Error(`Path "${targetPath}" changed through a symlink during write.`)
  const identity = await directoryIdentity(current)
  if (identity.dev !== expected.dev || identity.ino !== expected.ino) {
    throw new Error(`Path "${targetPath}" changed parent directory during write.`)
  }
}

async function assertExpectedFile(target: string, expectedSha256: string): Promise<void> {
  const noFollow = typeof constants.O_NOFOLLOW === 'number' ? constants.O_NOFOLLOW : 0
  let handle: Awaited<ReturnType<typeof fs.open>> | undefined
  try {
    handle = await fs.open(target, constants.O_RDONLY | noFollow | constants.O_NONBLOCK)
    const before = await handle.stat()
    if (!before.isFile()) throw new Error('target is not a regular file')
    const opened = handle
    const bytes = await withFileReadBudget({}, (budget) => readOpenedTextFile({
      stat: async () => ({ isFile: before.isFile(), size: before.size }),
      read: (buffer, length, position) => opened.read(buffer, 0, length, position).then((result) => result.bytesRead),
      close: async () => {}, // The identity check below still needs this handle.
    }, budget)) as Buffer
    const digest = workspaceContentSha256(bytes)
    const after = await fs.lstat(target)
    if (
      after.isSymbolicLink() || before.dev !== after.dev || before.ino !== after.ino ||
      before.size !== after.size || before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs ||
      digest !== expectedSha256
    ) {
      throw new Error('target changed')
    }
  } catch {
    throw new Error(`Target "${target}" changed after it was read; refusing to overwrite concurrent work.`)
  } finally {
    await handle?.close().catch(() => {})
  }
}

/**
 * Write through an unpredictable, O_EXCL/O_NOFOLLOW temp inode in the verified
 * canonical parent, then rename it over the final directory entry. POSIX rename
 * replaces a last-moment symlink rather than following it, closing the old
 * lstat-then-write race that could overwrite the symlink target.
 */
export async function atomicWriteWorkspaceFile(
  workspaceRoot: string,
  targetPath: string,
  content: string | Uint8Array,
  deps: AtomicWorkspaceWriteDeps = {},
): Promise<void> {
  await canonicalWorkspacePath(workspaceRoot, targetPath, 'write')
  const root = await fs.realpath(workspaceRoot)
  const lexicalTarget = path.resolve(targetPath)
  const lexicalRoot = path.resolve(workspaceRoot)
  if (!isWithin(lexicalRoot, lexicalTarget)) throw new Error(`Path "${targetPath}" escapes the workspace.`)

  const lexicalParent = path.dirname(lexicalTarget)
  const parent = await canonicalParent(root, lexicalParent)
  if (!isWithin(root, parent)) throw new Error(`Path "${targetPath}" escapes the workspace through a symlink.`)
  const parentIdentity = await directoryIdentity(parent)

  const target = path.join(parent, path.basename(lexicalTarget))
  assertWorkspacePathPolicy(root, target, 'write')
  let mode = 0o666
  let preserveMode = false
  try {
    const stat = await fs.lstat(target)
    if (stat.isFile()) {
      mode = stat.mode & 0o777
      preserveMode = true
    }
  } catch { /* new file */ }

  const temp = path.join(parent, `.vontra-write-${process.pid}-${randomUUID()}`)
  const noFollow = typeof constants.O_NOFOLLOW === 'number' ? constants.O_NOFOLLOW : 0
  const handle = await fs.open(temp, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | noFollow, mode)
  let committed = false
  try {
    // open(2)'s mode is filtered through umask. Preserve an existing target's
    // executable/permission bits explicitly when this is a replacement.
    if (preserveMode) await handle.chmod(mode)
    await handle.writeFile(content)
    await handle.sync()
    await deps.beforeRename?.(target)
    // Revalidate both the canonical path and the directory inode immediately before
    // committing. The unpredictable source name then makes a parent swap fail closed:
    // the replacement directory cannot name our temp inode.
    await assertStableParent(lexicalParent, parent, parentIdentity, targetPath)
    // This protects the normal read/stage/commit window, but portable Node does
    // not expose an openat/renameat-style conditional replacement primitive.
    // A non-cooperating process can still replace the entry in the tiny gap
    // between this preflight and rename; see security-operations.md.
    if (deps.expectedSha256) await assertExpectedFile(target, deps.expectedSha256)
    if (deps.ifExists === 'error') {
      // link(2) is the portable atomic no-clobber primitive. It fails with EEXIST for
      // files, directories, and symlinks, including a target raced in after staging.
      await fs.link(temp, target)
      committed = true
      await fs.unlink(temp).catch(() => {})
    } else {
      // rename replaces a last-moment final symlink instead of following it.
      await fs.rename(temp, target)
      committed = true
    }
  } finally {
    if (!committed) {
      // If an attacker renamed the parent, the temp pathname may no longer be
      // reachable. Keep the handle open until here so failed writes can still be
      // truncated, preventing content from being stranded in a moved temp inode.
      await handle.truncate(0).catch(() => {})
      await handle.sync().catch(() => {})
    }
    await handle.close().catch(() => {})
    await fs.unlink(temp).catch(() => {})
  }
}
