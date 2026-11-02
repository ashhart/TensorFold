import { constants, promises as fs, type Stats } from 'node:fs'
import path from 'node:path'

export interface LocalPackageBin {
  relative: string
  dev: number
  ino: number
  size: number
  mtimeMs: number
  ctimeMs: number
  mode: number
}

export interface LocalBinPermissionChange {
  before: LocalPackageBin
  after: LocalPackageBin | null
  applied: boolean
}

function snapshot(relative: string, stat: Stats): LocalPackageBin {
  return { relative, dev: stat.dev, ino: stat.ino, size: stat.size, mtimeMs: stat.mtimeMs,
    ctimeMs: stat.ctimeMs, mode: stat.mode & 0o777 }
}

async function readLocalBin(root: string, relative: string): Promise<LocalPackageBin> {
  let current = root
  const segments = relative.split('/')
  for (const [index, segment] of segments.entries()) {
    if (!segment || segment === '.' || segment === '..') throw new Error('Invalid local package executable path.')
    current = path.join(current, segment)
    const stat = await fs.lstat(current)
    const last = index === segments.length - 1
    if (stat.isSymbolicLink() || (last ? !stat.isFile() || stat.nlink !== 1 : !stat.isDirectory())) {
      throw new Error(`Local package executable must be an owned regular file without symlink parents: ${relative}`)
    }
    if (last) {
      if (await fs.realpath(current) !== current) throw new Error(`Local package executable path changed: ${relative}`)
      return snapshot(relative, stat)
    }
  }
  throw new Error('A local package executable path is empty.')
}

function sameBin(left: LocalPackageBin, right: LocalPackageBin): boolean {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size
    && left.mtimeMs === right.mtimeMs && left.ctimeMs === right.ctimeMs && left.mode === right.mode
}

export async function localPackageBins(root: string, directory: string, manifest: Record<string, any>): Promise<LocalPackageBin[]> {
  if (!directory) return []
  if (manifest.bin === undefined) {
    if (manifest.directories?.bin !== undefined) throw new Error('Local packages must declare explicit bin entries instead of directories.bin.')
    return []
  }
  const bin = typeof manifest.bin === 'string' && typeof manifest.name === 'string'
    ? { [manifest.name.split('/').pop()!]: manifest.bin } : manifest.bin
  if (!bin || typeof bin !== 'object' || Array.isArray(bin) || Object.keys(bin).length > 256) {
    throw new Error('Local package bin must name a bounded map of executable paths.')
  }
  const paths = new Set<string>()
  for (const [name, target] of Object.entries(bin)) {
    if (!/^[a-z0-9._-]+$/i.test(name) || name === '.' || name === '..' || name.length > 214
      || typeof target !== 'string' || !target || target.length > 4096 || /[\\\x00-\x1f]/.test(target)) {
      throw new Error('Local package bin contains an invalid command name or path.')
    }
    const normalized = path.posix.normalize(target)
    if (path.posix.isAbsolute(normalized) || normalized === '.' || normalized === '..' || normalized.startsWith('../')) {
      throw new Error('Local package executable paths must stay inside their declaring package.')
    }
    if (normalized.split('/').length > 80 || normalized.split('/').some((segment) => segment.startsWith('.') || segment === 'node_modules')) {
      throw new Error('Local package executable paths cannot enter hidden directories or node_modules.')
    }
    paths.add(path.posix.join(directory, normalized))
  }
  if (paths.size && process.platform === 'win32') {
    throw new Error('Isolated local-package executables currently require a POSIX host; Windows command shims need a separate setup workflow.')
  }
  return Promise.all([...paths].map((relative) => readLocalBin(root, relative)))
}

export async function stageLocalPackageBins(stage: string, bins: readonly LocalPackageBin[]): Promise<void> {
  for (const bin of bins) {
    const target = path.join(stage, bin.relative)
    await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 })
    // npm needs a target to create relative bin links; workspace source bytes never enter this stage.
    await fs.writeFile(target, '#!/usr/bin/env node\n', { flag: 'wx', mode: 0o755 })
  }
}

export async function assertLocalPackageBins(
  root: string, bins: readonly LocalPackageBin[], changes: readonly LocalBinPermissionChange[] = [],
): Promise<void> {
  for (const bin of bins) {
    const change = changes.find((entry) => entry.applied && entry.before.relative === bin.relative)
    const expected = change?.after ?? bin
    if (!sameBin(expected, await readLocalBin(root, bin.relative))) {
      throw new Error(`Local package executable changed during installation: ${bin.relative}`)
    }
  }
}

async function withBinHandle<T>(root: string, expected: LocalPackageBin, action: (handle: Awaited<ReturnType<typeof fs.open>>) => Promise<T>): Promise<T> {
  const handle = await fs.open(path.join(root, expected.relative), constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    const stat = await handle.stat()
    if (!stat.isFile() || stat.nlink !== 1 || !sameBin(expected, snapshot(expected.relative, stat))
      || !sameBin(expected, await readLocalBin(root, expected.relative))) {
      throw new Error(`Local package executable changed before permission update: ${expected.relative}`)
    }
    return await action(handle)
  } finally { await handle.close() }
}

export async function publishLocalBinPermissions(root: string, bins: readonly LocalPackageBin[], changes: LocalBinPermissionChange[]): Promise<void> {
  for (const bin of bins) {
    if ((bin.mode & 0o111) === 0o111) continue
    await withBinHandle(root, bin, async (handle) => {
      const change: LocalBinPermissionChange = { before: bin, after: null, applied: false }
      changes.push(change)
      await handle.chmod(bin.mode | 0o111)
      change.applied = true
      change.after = snapshot(bin.relative, await handle.stat())
    })
  }
}

export async function rollbackLocalBinPermissions(root: string, changes: readonly LocalBinPermissionChange[]): Promise<string[]> {
  const failed: string[] = []
  for (const change of [...changes].reverse()) {
    if (!change.applied) continue
    try {
      if (!change.after) throw new Error('The executable permission update could not be verified.')
      await withBinHandle(root, change.after, (handle) => handle.chmod(change.before.mode))
    } catch { failed.push(change.before.relative) }
  }
  return failed
}
