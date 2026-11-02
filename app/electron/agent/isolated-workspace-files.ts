import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { windowsLocalInput } from './windows-local-input'

export interface WorkspaceEntry {
  kind: 'file' | 'directory' | 'link'
  dev: number
  ino: number
  mode: number
  sha256?: string
  link?: string
  rawLink?: string
  size: number
  nlink: number
}
export type WorkspaceInventory = Map<string, WorkspaceEntry>
export interface WorkspaceReadOptions { externalReadRoots?: string[]; virtualRoot?: string }
const MAX_FILES = 200_000
const MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
export const MAX_WORKSPACE_FILE_BYTES = MAX_TOTAL_BYTES

export function insideWorkspace(root: string, target: string): boolean {
  const relative = path.relative(root, target)
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative))
}

export function protectedGitPath(relative: string): boolean {
  return relative.split(/[\\/]/).some(part => /^\.git(?:[. ]|$)/i.test(part) || /^git~\d+$/i.test(part))
}

export function validateWorkspaceName(name: string): void {
  if (!name || name === '.' || name === '..' || /[\\/:\x00-\x1f]/.test(name) || /[. ]$/.test(name)
    || /^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)/i.test(name)) throw new Error('Workspace contains an unsafe portable filename.')
}

export async function snapshotWorkspaceFile(file: string, destination?: string): Promise<{ sha256: string; size: number }> {
  const handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  let output: Awaited<ReturnType<typeof fs.open>> | undefined
  try {
    output = destination ? await fs.open(destination, 'wx', 0o600) : undefined
    const before = await handle.stat()
    if (!before.isFile() || before.size > MAX_WORKSPACE_FILE_BYTES) throw new Error('Workspace file exceeds the bounded regular-file limit.')
    const buffer = Buffer.alloc(Math.max(1, Math.min(1024 * 1024, before.size))), digest = createHash('sha256')
    let size = 0
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, size)
      if (!bytesRead) break
      size += bytesRead
      if (size > before.size) throw new Error('Workspace file grew while being read.')
      const bytes = buffer.subarray(0, bytesRead)
      digest.update(bytes)
      if (output) {
        let written = 0
        while (written < bytesRead) {
          const count = (await output.write(bytes, written, bytesRead - written)).bytesWritten
          if (!count) throw new Error('Isolated snapshot could not make progress.')
          written += count
        }
      }
    }
    const after = await fs.lstat(file)
    if (size !== before.size || !after.isFile() || before.dev !== after.dev || before.ino !== after.ino
      || before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs) throw new Error('Workspace file changed while being read.')
    return { sha256: digest.digest('hex'), size }
  } finally { await handle.close(); await output?.close() }
}

export async function workspaceFileBytes(file: string): Promise<Buffer> {
  const stat = await fs.lstat(file)
  if (!stat.isFile() || stat.size > 16 * 1024 * 1024) throw new Error('Isolated input metadata exceeds the bounded read limit.')
  const before = await snapshotWorkspaceFile(file)
  const bytes = await fs.readFile(file)
  if (bytes.length !== before.size || workspaceHash(bytes) !== before.sha256) throw new Error('Isolated metadata changed while reading.')
  return bytes
}

export function workspaceHash(bytes: Buffer): string { return createHash('sha256').update(bytes).digest('hex') }

export function sameWorkspaceContent(a?: WorkspaceEntry, b?: WorkspaceEntry): boolean {
  return a?.kind === b?.kind && a?.mode === b?.mode && a?.sha256 === b?.sha256 && a?.link === b?.link
}

export async function scanWorkspace(root: string, copyTo?: string, options: WorkspaceReadOptions = {}): Promise<WorkspaceInventory> {
  const entries: WorkspaceInventory = new Map()
  let total = 0
  const walk = async (relative: string, depth: number): Promise<void> => {
    if (depth > 80) throw new Error('Workspace exceeds the directory depth limit.')
    const names = await fs.readdir(path.join(root, relative))
    const folded = new Set<string>()
    for (const name of names.sort()) {
      validateWorkspaceName(name)
      if (folded.has(name.toLowerCase())) throw new Error('Workspace contains colliding portable filenames.')
      folded.add(name.toLowerCase())
      const rel = relative ? `${relative}/${name}` : name
      if (entries.size >= MAX_FILES) throw new Error('Workspace exceeds the file count limit.')
      const full = path.join(root, rel), target = copyTo ? path.join(copyTo, rel) : null
      const stat = await fs.lstat(full)
      const entry: WorkspaceEntry = {
        kind: stat.isDirectory() ? 'directory' : stat.isSymbolicLink() ? 'link' : 'file',
        dev: stat.dev, ino: stat.ino, mode: stat.mode & 0o777, size: stat.size, nlink: stat.nlink,
      }
      entries.set(rel, entry)
      if (stat.isSymbolicLink()) {
        const rawLink = await fs.readlink(full)
        let destination = path.resolve(path.dirname(full), rawLink)
        if (options.virtualRoot && path.isAbsolute(rawLink) && insideWorkspace(options.virtualRoot, destination)) {
          destination = path.join(root, path.relative(options.virtualRoot, destination))
        }
        const permitted = (target: string) => insideWorkspace(root, target) || (options.externalReadRoots ?? []).some(readRoot => insideWorkspace(readRoot, target))
        if (!permitted(destination)) throw new Error('Workspace symlink leaves its isolated root or read-only SDK roots.')
        if (process.platform === 'win32' && !await windowsLocalInput(destination)) throw new Error('Workspace link has an unsafe reparse target.')
        const resolved = await fs.realpath(destination)
        if (!permitted(resolved) || protectedGitPath(resolved)) throw new Error('Workspace symlink escapes its roots or aliases Git controls.')
        entry.rawLink = rawLink
        entry.link = insideWorkspace(root, resolved)
          ? path.isAbsolute(rawLink) ? path.relative(path.dirname(full), resolved) : rawLink
          : resolved
        if (process.platform === 'win32') {
          const actual = await fs.stat(resolved)
          if (actual.isDirectory()) {
            if (target) await fs.mkdir(target, { mode: 0o700 })
            await walk(rel, depth + 1)
          } else if (actual.isFile()) {
            const bytes = await workspaceFileBytes(resolved)
            total += bytes.length
            if (total > MAX_TOTAL_BYTES) throw new Error('Workspace exceeds the isolated copy size limit.')
            if (target) await fs.writeFile(target, bytes, { flag: 'wx', mode: actual.mode & 0o777 })
          } else throw new Error('Workspace link does not reference a regular file or directory.')
        } else if (target) await fs.symlink(entry.link, target)
      } else if (stat.isDirectory()) {
        if (target) await fs.mkdir(target, { mode: 0o700 })
        await walk(rel, depth + 1)
        if (target) await fs.chmod(target, entry.mode)
      } else if (stat.isFile()) {
        total += stat.size
        if (total > MAX_TOTAL_BYTES) throw new Error('Workspace exceeds the isolated copy size limit.')
        entry.sha256 = (await snapshotWorkspaceFile(full, target ?? undefined)).sha256
        if (target) await fs.chmod(target, entry.mode)
      } else throw new Error('Workspace contains an unsupported special file.')
    }
  }
  await walk('', 0)
  return entries
}
